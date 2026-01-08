/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package main

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/klog/v2"

	nvdev "github.com/NVIDIA/go-nvlib/pkg/nvlib/device"
	"github.com/NVIDIA/go-nvlib/pkg/nvpci"
	"github.com/NVIDIA/go-nvml/pkg/nvml"
	"k8s.io/dynamic-resource-allocation/deviceattribute"

	"github.com/NVIDIA/k8s-dra-driver-gpu/pkg/featuregates"
)

type deviceLib struct {
	nvdev.Interface
	nvmllib           nvml.Interface
	nvpci             nvpci.Interface
	driverLibraryPath string
	devRoot           string
	nvidiaSMIPath     string
	gpuInfosByUUID    map[string]*GpuInfo
	devhandleByUUID   map[string]nvml.Device
}

const (
	procDevicesPath      = "/proc/devices"
	nvidiaCapsDeviceName = "nvidia-caps"
)

type nvcapDeviceInfo struct {
	major  int
	minor  int
	mode   int
	modify int
	path   string
}

// MIG minors are predictable, and can be looked up:
//
// cat /proc/driver/nvidia-caps/mig-minors
//
// ...
// gpu0/gi0/access 3
// gpu0/gi0/ci0/access 4
// gpu0/gi0/ci1/access 5
// gpu0/gi0/ci2/access 6
// ...
// gpu3/gi3/ci3/access 439
// gpu3/gi3/ci4/access 440
// gpu3/gi3/ci5/access 441
// ...
// gpu6/gi11/ci5/access 918
// gpu6/gi11/ci6/access 919
// gpu6/gi11/ci7/access 920
//
// func getNVCapDevNodeInfoForMigMinor(migminor int) (*nvcapDeviceInfo, error) {
// 	major, err := getDeviceMajor(nvidiaCapsDeviceName)
// 	if err != nil {
// 		return nil, fmt.Errorf("error getting device major: %w", err)
// 	}

// 	info := &nvcapDeviceInfo{
// 		major:  major,
// 		minor:  migminor,
// 		mode:   0666,
// 		modify: 0,
// 		path:   fmt.Sprintf("/dev/nvidia-caps/nvidia-cap%d", migminor),
// 	}

// 	return info, nil
// }

func newDeviceLib(driverRoot root) (*deviceLib, error) {
	driverLibraryPath, err := driverRoot.getDriverLibraryPath()
	if err != nil {
		return nil, fmt.Errorf("failed to locate driver libraries: %w", err)
	}

	nvidiaSMIPath, err := driverRoot.getNvidiaSMIPath()
	if err != nil {
		return nil, fmt.Errorf("failed to locate nvidia-smi: %w", err)
	}

	// We construct an NVML library specifying the path to libnvidia-ml.so.1
	// explicitly so that we don't have to rely on the library path.
	nvmllib := nvml.New(
		nvml.WithLibraryPath(driverLibraryPath),
	)
	nvpci := nvpci.New()

	//nvmllib.nvsandboxutilslib

	d := deviceLib{
		Interface:         nvdev.New(nvmllib),
		nvmllib:           nvmllib,
		driverLibraryPath: driverLibraryPath,
		devRoot:           driverRoot.getDevRoot(),
		nvidiaSMIPath:     nvidiaSMIPath,
		nvpci:             nvpci,
		gpuInfosByUUID:    make(map[string]*GpuInfo),
		devhandleByUUID:   make(map[string]nvml.Device),
	}

	// Populate `l.gpuInfosByUUID`.
	//d.VisitAllDevices()
	if err := d.Init(); err != nil {
		return nil, fmt.Errorf("failed to initialize NVML: %w", err)
	}

	return &d, nil
}

// prependPathListEnvvar prepends a specified list of strings to a specified envvar and returns its value.
func prependPathListEnvvar(envvar string, prepend ...string) string {
	if len(prepend) == 0 {
		return os.Getenv(envvar)
	}
	current := filepath.SplitList(os.Getenv(envvar))
	return strings.Join(append(prepend, current...), string(filepath.ListSeparator))
}

// setOrOverrideEnvvar adds or updates an envar to the list of specified envvars and returns it.
func setOrOverrideEnvvar(envvars []string, key, value string) []string {
	var updated []string
	for _, envvar := range envvars {
		pair := strings.SplitN(envvar, "=", 2)
		if pair[0] == key {
			continue
		}
		updated = append(updated, envvar)
	}
	return append(updated, fmt.Sprintf("%s=%s", key, value))
}

// New paradigm, for faster device managemen (getting device handles by UUID can
// take (10 seconds) when done concurrently. Try something slightly more
// difficult instead: maintain long-term state: initialize once at startup,
// cache handles, serialize NVML calls, and implement a clear re-init path on
// NVML errors; this hopefully balances performance and robustness for this
// long-running process.
func (l deviceLib) Init() error {
	klog.Infof("Initializing NVML")
	ret := l.nvmllib.Init()
	if ret != nvml.SUCCESS {
		return fmt.Errorf("error initializing NVML: %v", ret)
	}
	return nil
}

func (l deviceLib) alwaysShutdown() {
	klog.Infof("Shutting down NVML")
	ret := l.nvmllib.Shutdown()
	if ret != nvml.SUCCESS {
		klog.Warningf("error shutting down NVML: %v", ret)
	}
}

// Yield the devices that are allocatable, on this node.
func (l deviceLib) enumerateAllPossibleDevices(config *Config) (AllocatableDevices, PerGPUMinorAllocatableDevices, error) {
	//alldevices := make(AllocatableDevices)

	// TODO: make good decisions about incarnated MIG devices found during
	// program startup. We could
	//
	// 1) assume they are under control of an external  entity, and not announce
	// them. That's likely not true. As hard as we try, _we_ might actually
	// leave a MIG device behind where there should be no more (as of bugs, as
	// of aggressive ops, ...).
	//
	// 2) not do anything: bad idea, we announce availability and the scheduler
	// will assign a job, and prepare() will try to create a specific MIG device
	// (incl placement) and that will fail because that MIG device already
	// exists -- users see something like "prepare devices failed: error
	// creating MIG device: error creating GPU instance for
	// 'gpu-0-mig-1g24gb-0': Insufficient Resources
	//
	// 3) Use the node-local checkpoint as the source of truth. Any MIG device
	// that corresponds to "partially prepared" claims must be destroyed, any
	// MIG device that is not mentioned must be destroyed (only those of
	// completely prepared claims can stay; assuming that the central scheduler
	// state is equivalent).

	// Get the full list of allocatable devices from GPU 0 in this machine
	// (development state, iterate over full GPU devices later)

	//allocatable, err := l.GetPerGpuAllocatableDevices(0)

	perGPUAllocatable, err := l.GetPerGpuAllocatableDevices()
	if err != nil {
		return nil, nil, fmt.Errorf("error enumerating allocatable devices: %w", err)
	}

	if featuregates.Enabled(featuregates.PassthroughSupport) {
		passthroughDevices, err := l.enumerateGpuPciDevices(config, gms)
		if err != nil {
			return nil, fmt.Errorf("error enumerating GPU PCI devices: %w", err)
		}
		for k, v := range passthroughDevices {
			alldevices[k] = v
		}
	}

	all := make(AllocatableDevices)
	for _, devices := range perGPUAllocatable {
		for name, dev := range devices {
			all[name] = dev
		}
	}

	//return alldevices, nil
	return all, perGPUAllocatable, nil
}

// PerGpuAllocatableDevices holds the list of allocatable devices per GPU.
// QQ(JP): what's the int-index good for/

type GPUMinor = int
type PerGPUMinorAllocatableDevices map[GPUMinor]AllocatableDevices

// Discover all (physical, full) GPUs. Assume that the result does not change at
// runtime. Populate `l.gpuInfosByUUID`.
// func (l deviceLib) VisitAllDevices() {
// 	if err := l.Init(); err != nil {
// 		return nil, err
// 	}
// 	defer l.alwaysShutdown()

// 	err := l.VisitDevices(func(i int, d nvdev.Device) error {
// 		gpuInfo, err := l.getGpuInfo(i, d)
// 		if err != nil {
// 			return fmt.Errorf("error getting info for GPU %v: %w", i, err)
// 		}

// 		return nil
// 	})
// 	if err != nil {
// 		return nil, fmt.Errorf("error visiting devices: %w", err)
// 	}
// }

// GetPerGpuAllocatableDevices is called once upon driver startup. It gets the
// set of allocatable devices using NVDeviceLib.  A list of GPU indices can be
// optionally provided to limit the set of allocatable devices to just those
// GPUs. If no indices are provided, the full set of allocatable devices across
// all GPUs are returned. NOTE: Both full GPUs and MIG devices are returned as
// part of this call.
func (l deviceLib) GetPerGpuAllocatableDevices(indices ...int) (PerGPUMinorAllocatableDevices, error) {
	klog.Infof("Traverse GPU devices")

	// if err := l.Init(); err != nil {
	// 	return nil, err
	// }
	// defer l.alwaysShutdown()

	// Flat representation of all allocatable devices
	// allAllocatable := make(AllocatableDevices)
	perGPUAllocatable := make(PerGPUMinorAllocatableDevices)

	err := l.VisitDevices(func(i int, d nvdev.Device) error {
		if indices != nil && !slices.Contains(indices, i) {
			return nil
		}

		// Just those allocatable devices for this physical GPU
		thisGPUAllocatable := make(AllocatableDevices)

		gpuInfo, err := l.getGpuInfo(i, d)
		if err != nil {
			return fmt.Errorf("error getting info for GPU %v: %w", i, err)
		}

		// parentdev := &AllocatableDevice{
		// 	Gpu: gpuInfo,
		// }

		// migdevs, err := l.discoverMigDevicesByGPU(gpuInfo)
		// if err != nil {
		// 	return fmt.Errorf("error discovering MIG devices for GPU %q: %w", gpuInfo.CanonicalName(), err)
		// }
		// if featuregates.Enabled(featuregates.PassthroughSupport) {
		// 	// If no MIG devices are found, allow VFIO devices.
		// 	gpuInfo.vfioEnabled = len(migdevs) == 0
		// }

		// if !gpuInfo.migEnabled {
		// 	klog.Infof("Adding device %s to allocatable devices", gpuInfo.CanonicalName())
		// 	devices[gpuInfo.CanonicalName()] = parentdev
		// 	return nil
		// }

		// // Likely unintentionally stranded capacity (misconfiguration).
		// if len(migdevs) == 0 {
		// 	klog.Warningf("Physical GPU %s has MIG mode enabled but no configured MIG devices", gpuInfo.CanonicalName())
		// }

		// for _, mdev := range migdevs {
		// 	klog.Infof("Adding MIG device %s to allocatable devices (parent: %s)", mdev.CanonicalName(), gpuInfo.CanonicalName())
		// 	devices[mdev.CanonicalName()] = mdev
		// }


		// Store gpuInfo object for later re-use (lookup by UUID).
		l.gpuInfosByUUID[gpuInfo.UUID] = gpuInfo

		// Cache warmup: store mapping between full-GPU UUID and NVML device
		// handle in a map. Ignore failures.
		if _, ret := l.DeviceGetHandleByUUIDCached(gpuInfo.UUID); ret != nvml.SUCCESS {
			klog.Warningf("DeviceGetHandleByUUIDCached failed: %s", ret)
		}

		// For this full device, inspect all MIG profiles and their possible
		// placements. This enriches `gpuInfo` with additional properties (such
		// as the memory slice count, and the maximum capacities as reported by
		// individual MIG profiles).
		migpps, err := l.inspectMigProfilesAndPlacements(gpuInfo, d)
		if err != nil {
			return fmt.Errorf("error getting MIG info for GPU %v: %w", i, err)
		}

		dev := &AllocatableDevice{
			Gpu: gpuInfo,
		}
		//allAllocatable[gpuInfo.CanonicalName()] = dev
		thisGPUAllocatable[gpuInfo.CanonicalName()] = dev

		for _, migpp := range migpps {
			dev := &AllocatableDevice{
				Mig: migpp,
			}
			//allAllocatable[migpp.CanonicalName()] = dev
			thisGPUAllocatable[migpp.CanonicalName()] = dev
		}

		perGPUAllocatable[gpuInfo.minor] = thisGPUAllocatable
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("error visiting devices: %w", err)
	}

	// Should we try to return an object here that retains a stable sort oder
	// over devices, by name? (a normal map does not cut it).
	return perGPUAllocatable, nil
}

func (l deviceLib) discoverMigDevicesByGPU(gpuInfo *GpuInfo) (AllocatableDeviceList, error) {
	var devices AllocatableDeviceList
	migs, err := l.getMigDevices(gpuInfo)
	if err != nil {
		return nil, fmt.Errorf("error getting MIG devices for GPU %q: %w", gpuInfo.CanonicalName(), err)
	}

	for _, migDeviceInfo := range migs {
		mig := &AllocatableDevice{
			Mig: migDeviceInfo,
		}
		devices = append(devices, mig)
	}
	return devices, nil
}

// TODO: Need go-nvlib util for this.
func (l deviceLib) discoverGPUByPCIBusID(pcieBusID string) (*AllocatableDevice, AllocatableDeviceList, error) {
	if err := l.Init(); err != nil {
		return nil, nil, err
	}
	defer l.alwaysShutdown()

	var gpu *AllocatableDevice
	var migs AllocatableDeviceList
	err := l.VisitDevices(func(i int, d nvdev.Device) error {
		gpuPCIBusID, err := d.GetPCIBusID()
		if err != nil {
			return fmt.Errorf("error getting PCIe bus ID for device %d: %w", i, err)
		}
		if gpuPCIBusID != pcieBusID {
			return nil
		}
		gpuInfo, err := l.getGpuInfo(i, d)
		if err != nil {
			return fmt.Errorf("error getting info for GPU %d: %w", i, err)
		}
		migs, err = l.discoverMigDevicesByGPU(gpuInfo)
		if err != nil {
			return fmt.Errorf("error discovering MIG devices for GPU %q: %w", gpuInfo.CanonicalName(), err)
		}
		// If no MIG devices are found, allow VFIO devices.
		gpuInfo.vfioEnabled = len(migs) == 0
		gpu = &AllocatableDevice{
			Gpu: gpuInfo,
		}
		return nil
	})
	if err != nil {
		return nil, nil, fmt.Errorf("error visiting devices: %w", err)
	}
	return gpu, migs, nil
}

// TODO: Need go-nvlib util for this.
func (l deviceLib) discoverVfioDevice(gpuInfo *GpuInfo) (*AllocatableDevice, error) {
	gpus, err := l.nvpci.GetGPUs()
	if err != nil {
		return nil, fmt.Errorf("error getting GPU PCI devices: %w", err)
	}
	for idx, gpu := range gpus {
		if gpu.Address != gpuInfo.pcieBusID {
			continue
		}
		vfioDeviceInfo, err := l.getVfioDeviceInfo(idx, gpu)
		if err != nil {
			return nil, fmt.Errorf("error getting VFIO device info: %w", err)
		}
		vfioDeviceInfo.parent = gpuInfo
		return &AllocatableDevice{
			Vfio: vfioDeviceInfo,
		}, nil
	}
	return nil, fmt.Errorf("error discovering VFIO device by PCIe bus ID: %s", gpuInfo.pcieBusID)
}
// Tear down any MIG devices that are present and don't belong to completed claims.
func (l deviceLib) obliterateStaleMIGDevices(expectedDeviceNames []DeviceName) error {
	// if err := l.Init(); err != nil {
	// 	return err
	// }
	// defer l.alwaysShutdown()

	err := l.VisitDevices(func(i int, d nvdev.Device) error {
		ginfo, err := l.getGpuInfo(i, d)
		if err != nil {
			return fmt.Errorf("error getting info for GPU %d: %w", i, err)
		}

		migs, err := l.getMigDevices(ginfo)
		if err != nil {
			return fmt.Errorf("error getting MIG devices for GPU %d: %w", i, err)
		}

		for _, mdi := range migs {
			// That's the name we announce the device with via DRA, i.e. it's
			// node-unique, and must (by definition) precisely describe a
			// specific device within a node.
			canoname := mdi.CanonicalName()
			expected := slices.Contains(expectedDeviceNames, canoname)
			if !expected {
				klog.Warningf("Found unexpected MIG device (%s), destroy", canoname)
				if err := l.deleteMigDevice(ginfo.UUID, mdi.GIID, mdi.CIID); err != nil {
					return fmt.Errorf("could not delete unexpected MIG device (%s): %w", canoname, err)
				}
			}
		}

		return nil
	})

	if err != nil {
		return fmt.Errorf("error visiting devices: %w", err)
	}
	return nil
}

// This enumerates full GPUs and currently (statically) configured MIG devices
// func (l deviceLib) enumerateGpusAndMigDevices(config *Config) (AllocatableDevices, error) {
// 	if err := l.Init(); err != nil {
// 		return nil, err
// 	}
// 	defer l.alwaysShutdown()

// 	devices := make(AllocatableDevices)
// 	err := l.VisitDevices(func(i int, d nvdev.Device) error {
// 		gpuInfo, err := l.getGpuInfo(i, d)
// 		if err != nil {
// 			return fmt.Errorf("error getting info for GPU %d: %w", i, err)
// 		}

// 		deviceInfo := &AllocatableDevice{
// 			Gpu: gpuInfo,
// 		}
// 		devices[gpuInfo.CanonicalName()] = deviceInfo

// 		migs, err := l.getMigDevices(gpuInfo)
// 		if err != nil {
// 			return fmt.Errorf("error getting MIG devices for GPU %d: %w", i, err)
// 		}

// 		for _, migDeviceInfo := range migs {
// 			deviceInfo := &AllocatableDevice{
// 				Mig: migDeviceInfo,
// 			}
// 			devices[migDeviceInfo.CanonicalName()] = deviceInfo
// 		}

// 		return nil
// 	})
// 	if err != nil {
// 		return nil, fmt.Errorf("error visiting devices: %w", err)
// 	}

// 	return devices, nil
// }

func (l deviceLib) getGpuInfo(index int, device nvdev.Device) (*GpuInfo, error) {
	minor, ret := device.GetMinorNumber()
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("error getting minor number for device %d: %v", index, ret)
	}
	uuid, ret := device.GetUUID()
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("error getting UUID for device %d: %v", index, ret)
	}
	migEnabled, err := device.IsMigEnabled()
	if err != nil {
		return nil, fmt.Errorf("error checking if MIG mode enabled for device %d: %w", index, err)
	}
	memory, ret := device.GetMemoryInfo()
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("error getting memory info for device %d: %v", index, ret)
	}
	productName, ret := device.GetName()
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("error getting product name for device %d: %v", index, ret)
	}
	architecture, err := device.GetArchitectureAsString()
	if err != nil {
		return nil, fmt.Errorf("error getting architecture for device %d: %w", index, err)
	}
	brand, err := device.GetBrandAsString()
	if err != nil {
		return nil, fmt.Errorf("error getting brand for device %d: %w", index, err)
	}
	cudaComputeCapability, err := device.GetCudaComputeCapabilityAsString()
	if err != nil {
		return nil, fmt.Errorf("error getting CUDA compute capability for device %d: %w", index, err)
	}
	driverVersion, ret := l.nvmllib.SystemGetDriverVersion()
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("error getting driver version: %w", err)
	}
	cudaDriverVersion, ret := l.nvmllib.SystemGetCudaDriverVersion()
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("error getting CUDA driver version: %w", err)
	}
	pcieBusID, err := device.GetPCIBusID()
	if err != nil {
		return nil, fmt.Errorf("error getting PCIe bus ID for device %d: %w", index, err)
	}

	var pcieRootAttr *deviceattribute.DeviceAttribute
	if attr, err := deviceattribute.GetPCIeRootAttributeByPCIBusID(pcieBusID); err == nil {
		pcieRootAttr = &attr
	} else {
		klog.Warningf("error getting PCIe root for device %d, continuing without attribute: %v", index, err)
	}

	var migProfiles []*MigProfileInfo
	for i := 0; i < nvml.GPU_INSTANCE_PROFILE_COUNT; i++ {
		giProfileInfo, ret := device.GetGpuInstanceProfileInfo(i)
		if ret == nvml.ERROR_NOT_SUPPORTED {
			continue
		}
		if ret == nvml.ERROR_INVALID_ARGUMENT {
			continue
		}
		if ret != nvml.SUCCESS {
			return nil, fmt.Errorf("error retrieving GpuInstanceProfileInfo for profile %d on GPU %v", i, uuid)
		}

		giPossiblePlacements, ret := device.GetGpuInstancePossiblePlacements(&giProfileInfo)
		if ret == nvml.ERROR_NOT_SUPPORTED {
			continue
		}
		if ret == nvml.ERROR_INVALID_ARGUMENT {
			continue
		}
		if ret != nvml.SUCCESS {
			return nil, fmt.Errorf("error retrieving GpuInstancePossiblePlacements for profile %d on GPU %v", i, uuid)
		}

		var migDevicePlacements []*MigDevicePlacement
		for _, p := range giPossiblePlacements {
			mdp := &MigDevicePlacement{
				GpuInstancePlacement: p,
			}
			migDevicePlacements = append(migDevicePlacements, mdp)
		}

		for j := 0; j < nvml.COMPUTE_INSTANCE_PROFILE_COUNT; j++ {
			for k := 0; k < nvml.COMPUTE_INSTANCE_ENGINE_PROFILE_COUNT; k++ {
				migProfile, err := l.NewMigProfile(i, j, k, giProfileInfo.MemorySizeMB, memory.Total)
				if err != nil {
					return nil, fmt.Errorf("error building MIG profile from GpuInstanceProfileInfo for profile %d on GPU %v", i, uuid)
				}

				if migProfile.GetInfo().G != migProfile.GetInfo().C {
					continue
				}

				profileInfo := &MigProfileInfo{
					profile:    migProfile,
					placements: migDevicePlacements,
				}

				migProfiles = append(migProfiles, profileInfo)
			}
		}
	}

	gpuInfo := &GpuInfo{
		UUID:  uuid,
		minor: minor,
		// index is not initialized here
		migEnabled:            migEnabled,
		memoryBytes:           memory.Total,
		productName:           productName,
		brand:                 brand,
		architecture:          architecture,
		cudaComputeCapability: cudaComputeCapability,
		driverVersion:         driverVersion,
		cudaDriverVersion:     fmt.Sprintf("%v.%v", cudaDriverVersion/1000, (cudaDriverVersion%1000)/10),
		pcieBusID:             pcieBusID,
		pcieRootAttr:          pcieRootAttr,
		migProfiles:           migProfiles,
		Health:                Healthy,
	}

	return gpuInfo, nil
}

func (l deviceLib) enumerateGpuPciDevices(config *Config, gms AllocatableDevices) (AllocatableDevices, error) {
	devices := make(AllocatableDevices)
	gpuPciDevices, err := l.nvpci.GetGPUs()
	if err != nil {
		return nil, fmt.Errorf("error getting GPU PCI devices: %w", err)
	}
	for idx, pci := range gpuPciDevices {
		parent := gms.GetGPUByPCIeBusID(pci.Address)
		if parent == nil || !parent.Gpu.vfioEnabled {
			continue
		}
		vfioDeviceInfo, err := l.getVfioDeviceInfo(idx, pci)
		if err != nil {
			return nil, fmt.Errorf("error getting GPU info from PCI device: %w", err)
		}
		vfioDeviceInfo.parent = parent.Gpu
		devices[vfioDeviceInfo.CanonicalName()] = &AllocatableDevice{
			Vfio: vfioDeviceInfo,
		}
	}
	return devices, nil
}

func (l deviceLib) getVfioDeviceInfo(idx int, device *nvpci.NvidiaPCIDevice) (*VfioDeviceInfo, error) {
	var pcieRootAttr *deviceattribute.DeviceAttribute
	attr, err := deviceattribute.GetPCIeRootAttributeByPCIBusID(device.Address)
	if err == nil {
		pcieRootAttr = &attr
	} else {
		klog.Warningf("error getting PCIe root for device %s, continuing without attribute: %v", device.Address, err)
	}

	_, memoryBytes := device.Resources.GetTotalAddressableMemory(true)

	vfioDeviceInfo := &VfioDeviceInfo{
		UUID:                   uuid.NewSHA1(uuid.NameSpaceDNS, []byte(device.Address)).String(),
		index:                  idx,
		productName:            device.DeviceName,
		pcieBusID:              device.Address,
		pcieRootAttr:           pcieRootAttr,
		deviceID:               fmt.Sprintf("0x%04x", device.Device),
		vendorID:               fmt.Sprintf("0x%04x", device.Vendor),
		numaNode:               device.NumaNode,
		iommuGroup:             device.IommuGroup,
		addressableMemoryBytes: memoryBytes,
	}
	return vfioDeviceInfo, nil
}

func (l deviceLib) getMigDevices(gpuInfo *GpuInfo) (map[string]*MigDeviceInfo, error) {
	if !gpuInfo.migEnabled {
		return nil, nil
	}

	// if err := l.Init(); err != nil {
	// 	return nil, err
	// }
	// defer l.alwaysShutdown()

	//device, ret := l.nvmllib.DeviceGetHandleByUUID(gpuInfo.UUID)
	device, ret := l.DeviceGetHandleByUUIDCached(gpuInfo.UUID)
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("error getting GPU device handle: %v", ret)
	}

	infos := make(map[string]*MigDeviceInfo)
	err := walkMigDevices(device, func(i int, migDevice nvml.Device) error {
		giID, ret := migDevice.GetGpuInstanceId()
		if ret != nvml.SUCCESS {
			return fmt.Errorf("error getting GPU instance ID for MIG device: %v", ret)
		}
		gi, ret := device.GetGpuInstanceById(giID)
		if ret != nvml.SUCCESS {
			return fmt.Errorf("error getting GPU instance for '%v': %v", giID, ret)
		}
		giInfo, ret := gi.GetInfo()
		if ret != nvml.SUCCESS {
			return fmt.Errorf("error getting GPU instance info for '%v': %v", giID, ret)
		}
		ciID, ret := migDevice.GetComputeInstanceId()
		if ret != nvml.SUCCESS {
			return fmt.Errorf("error getting Compute instance ID for MIG device: %v", ret)
		}
		ci, ret := gi.GetComputeInstanceById(ciID)
		if ret != nvml.SUCCESS {
			return fmt.Errorf("error getting Compute instance for '%v': %v", ciID, ret)
		}
		ciInfo, ret := ci.GetInfo()
		if ret != nvml.SUCCESS {
			return fmt.Errorf("error getting Compute instance info for '%v': %v", ciID, ret)
		}
		uuid, ret := migDevice.GetUUID()
		if ret != nvml.SUCCESS {
			return fmt.Errorf("error getting UUID for MIG device: %v", ret)
		}

		var migProfile *MigProfileInfo
		var giProfileInfo *nvml.GpuInstanceProfileInfo
		var ciProfileInfo *nvml.ComputeInstanceProfileInfo
		for _, profile := range gpuInfo.migProfiles {
			profileInfo := profile.profile.GetInfo()
			gipInfo, ret := device.GetGpuInstanceProfileInfo(profileInfo.GIProfileID)
			if ret != nvml.SUCCESS {
				continue
			}
			if giInfo.ProfileId != gipInfo.Id {
				continue
			}
			cipInfo, ret := gi.GetComputeInstanceProfileInfo(profileInfo.CIProfileID, profileInfo.CIEngProfileID)
			if ret != nvml.SUCCESS {
				continue
			}
			if ciInfo.ProfileId != cipInfo.Id {
				continue
			}
			migProfile = profile
			giProfileInfo = &gipInfo
			ciProfileInfo = &cipInfo
		}
		if migProfile == nil {
			return fmt.Errorf("error getting profile info for MIG device: %v", uuid)
		}

		placement := MigDevicePlacement{
			GpuInstancePlacement: giInfo.Placement,
		}

		infos[uuid] = &MigDeviceInfo{
			UUID:          uuid,
			Profile:       migProfile.String(),
			ParentMinor:   gpuInfo.minor,
			ParentUUID:    gpuInfo.UUID,
			CIID:          int(ciInfo.Id),
			GIID:          int(giInfo.Id),
			Placement:     &placement,
			parent:        gpuInfo,
			giProfileInfo: giProfileInfo,
			gIInfo:        &giInfo,
			ciProfileInfo: ciProfileInfo,
			cIInfo:        &ciInfo,
			pcieBusID:     gpuInfo.pcieBusID,
			pcieRootAttr:  gpuInfo.pcieRootAttr,
			Health:        Healthy,
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("error enumerating MIG devices: %w", err)
	}

	if len(infos) == 0 {
		return nil, nil
	}

	return infos, nil
}

func walkMigDevices(d nvml.Device, f func(i int, d nvml.Device) error) error {
	count, ret := nvml.Device(d).GetMaxMigDeviceCount()
	if ret != nvml.SUCCESS {
		return fmt.Errorf("error getting max MIG device count: %v", ret)
	}

	for i := 0; i < count; i++ {
		device, ret := d.GetMigDeviceHandleByIndex(i)
		if ret == nvml.ERROR_NOT_FOUND {
			continue
		}
		if ret == nvml.ERROR_INVALID_ARGUMENT {
			continue
		}
		if ret != nvml.SUCCESS {
			return fmt.Errorf("error getting MIG device handle at index '%v': %v", i, ret)
		}
		err := f(i, device)
		if err != nil {
			return err
		}
	}
	return nil
}

func (l deviceLib) setTimeSlice(uuids []string, timeSlice int) error {
	for _, uuid := range uuids {
		cmd := exec.Command(
			l.nvidiaSMIPath,
			"compute-policy",
			"-i", uuid,
			"--set-timeslice", fmt.Sprintf("%d", timeSlice))

		// In order for nvidia-smi to run, we need update LD_PRELOAD to include the path to libnvidia-ml.so.1.
		cmd.Env = setOrOverrideEnvvar(os.Environ(), "LD_PRELOAD", prependPathListEnvvar("LD_PRELOAD", l.driverLibraryPath))

		output, err := cmd.CombinedOutput()
		if err != nil {
			klog.Errorf("\n%v", string(output))
			return fmt.Errorf("error running nvidia-smi: %w", err)
		}
	}
	return nil
}

func (l deviceLib) setComputeMode(uuids []string, mode string) error {
	for _, uuid := range uuids {
		cmd := exec.Command(
			l.nvidiaSMIPath,
			"-i", uuid,
			"-c", mode)

		// In order for nvidia-smi to run, we need update LD_PRELOAD to include the path to libnvidia-ml.so.1.
		cmd.Env = setOrOverrideEnvvar(os.Environ(), "LD_PRELOAD", prependPathListEnvvar("LD_PRELOAD", l.driverLibraryPath))

		output, err := cmd.CombinedOutput()
		if err != nil {
			klog.Errorf("\n%v", string(output))
			return fmt.Errorf("error running nvidia-smi: %w", err)
		}
	}
	return nil
}

// This is meant to only be used for physical, full GPUs (not MIG devices).
func (l deviceLib) DeviceGetHandleByUUIDCached(uuid string) (nvml.Device, nvml.Return) {
	dev, exists := l.devhandleByUUID[uuid]
	if exists {
		return dev, nvml.SUCCESS
	}

	klog.V(6).Infof("DeviceGetHandleByUUIDCached called for %s, cache miss", uuid)
	// Note(JP): in theory here we need a 'single‑flight' (request coalescing)
	// strategy. Otherwise, cache stampede is a thing in practice: a burst of
	// requests with the same UUID might be incoming in a timeframe much shorter
	// than it takes for the call below to succeed. All cache missed then end up
	// doing this expensive lookup, although it only needs to be performed once.
	// For now, I opt for addressing this by warming up the cache during program
	// startup. Given that the set of full GPUs is static and that we have no
	// expiry (but a long-lived map), that will work.
	dev, ret := l.nvmllib.DeviceGetHandleByUUID(uuid)

	if ret != nvml.SUCCESS {
		return nil, ret
	}
	l.devhandleByUUID[uuid] = dev
	return dev, ret
}

func (l deviceLib) createMigDevice(migpp *MigInfo) (*MigDeviceInfo, error) {
	gpu := migpp.Parent
	profile := migpp.Profile
	placement := &migpp.MemorySlices

	// tcmigdev0 := time.Now()
	// if err := l.Init(); err != nil {
	// 	return nil, err
	// }
	// defer l.alwaysShutdown()
	// klog.V(6).Infof("t_prep_create_mig_dev_init %.3f s", time.Since(tcmigdev0).Seconds())

	tdhbu0 := time.Now()
	// I've seen this to take up to 10 seconds, also four six 8 seconds.
	// Can this be cached?
	//device, ret := l.nvmllib.DeviceGetHandleByUUID(gpu.UUID)
	device, ret := l.DeviceGetHandleByUUIDCached(gpu.UUID)
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("error getting GPU device handle: %v", ret)
	}
	klog.V(6).Infof("t_prep_create_mig_dev_get_dev_handle %.3f s", time.Since(tdhbu0).Seconds())

	tnd0 := time.Now()
	ndev, err := l.NewDevice(device)
	if err != nil {
		return nil, fmt.Errorf("error instantiating nvml dev: %w", err)
	}
	klog.V(6).Infof("t_prep_create_mig_dev_new_dev %.3f s", time.Since(tnd0).Seconds())

	// nvml GetMigMode distinguishes between current and pending -- not exposed
	// in go-nvlib yet. Maybe that distinction is important here.
	// migModeCurrent, migModePending, err := device.GetMigMode()
	// https://github.com/NVIDIA/go-nvlib/blame/7d260da4747c220a6972ebc83e4eb7116fc9b89a/pkg/nvlib/device/device.go#L225
	tcme0 := time.Now()
	migEnabled, err := ndev.IsMigEnabled()
	if err != nil {
		return nil, fmt.Errorf("error checking if MIG mode enabled for device %s: %w", ndev, err)
	}
	klog.V(6).Infof("t_prep_create_mig_dev_check_mig_enabled %.3f s", time.Since(tcme0).Seconds())

	logpfx := fmt.Sprintf("Create %s", migpp.CanonicalName())

	if !migEnabled {
		klog.V(6).Infof("%s: Attempting to enable MIG mode for to-be parent %s", logpfx, gpu.String())
		// If this is newer than A100 and if device unused: enable MIG.
		tem0 := time.Now()
		ret, activationStatus := device.SetMigMode(nvml.DEVICE_MIG_ENABLE)
		if ret != nvml.SUCCESS {
			// activationStatus would return the appropriate error code upon unsuccessful activation
			klog.Warningf("%s: SetMigMode activationStatus (device %s): %s", logpfx, gpu.String(), activationStatus)
			return nil, fmt.Errorf("error enabling MIG mode for device %s: %v", gpu.String(), ret)
		}
		klog.V(1).Infof("%s: MIG mode now enabled for device %s, t_enable_mig %.3f s", logpfx, gpu.String(), time.Since(tem0).Seconds())
	} else {
		klog.V(6).Infof("%s: MIG mode already enabled for device %s", logpfx, gpu.String())
	}

	profileInfo := profile.GetInfo()

	tcgigi0 := time.Now()
	giProfileInfo, ret := device.GetGpuInstanceProfileInfo(profileInfo.GIProfileID)
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("error getting GPU instance profile info for '%v': %v", profile, ret)
	}

	gi, ret := device.CreateGpuInstanceWithPlacement(&giProfileInfo, placement)
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("error creating GPU instance for '%s': %v", migpp.CanonicalName(), ret)
	}

	giInfo, ret := gi.GetInfo()
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("error getting GPU instance info for '%s': %v", migpp.CanonicalName(), ret)
	}

	ciProfileInfo, ret := gi.GetComputeInstanceProfileInfo(profileInfo.CIProfileID, profileInfo.CIEngProfileID)
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("error getting Compute instance profile info for '%v': %v", profile, ret)
	}

	ci, ret := gi.CreateComputeInstance(&ciProfileInfo)
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("error creating Compute instance for '%v': %v", profile, ret)
	}

	ciInfo, ret := ci.GetInfo()
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("error getting GPU instance info for '%v': %v", profile, ret)
	}
	klog.V(6).Infof("t_prep_create_mig_dev_cigi %.3f s", time.Since(tcgigi0).Seconds())

	// For obtaining the UUID, we initially walked through all MIG devices on
	// the parent GPU to identify the one that matches the CIID and GIID of the
	// MIG device that was just created; we then extracted the UUID from there.
	// Under load, this 'walk all MIG devices' took up to ten seconds. This can
	// be simplified by getting the MIG device handle from the CI and then
	// calling the UUID API on that handle. A MIG device handle maps 1:1 to a CI
	// in NVML, so once the CI is known, the MIG device handle and its UUID can
	// be retrieved directly without scanning through indices.
	uuid, ret := ciInfo.Device.GetUUID()
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("error getting UUID from CI info/device for CI %d: %v", ciInfo.Id, ret)
	}

	// Should use two types here, one just for the 3-tuple, and then the rest of
	// the info. Things get confusing.
	migDevInfo := &MigDeviceInfo{
		UUID:       uuid,
		CIID:       int(ciInfo.Id),
		GIID:       int(giInfo.Id),
		ParentUUID: gpu.UUID,
		Profile:    profile.String(),
		Placement: &MigDevicePlacement{
			GpuInstancePlacement: *placement,
		},
		gIInfo: &giInfo,
		cIInfo: &ciInfo,
		parent: gpu,
	}

	klog.V(6).Infof("%s: MIG device created on %s: %s (%s)", logpfx, gpu.String(), migDevInfo.CanonicalName(), migDevInfo.UUID)
	return migDevInfo, nil
}

// TODO(JP): the three function arguments provided represent that fundamental
// 3-tuple for precisely describing a concrete MIG device. Introduce a data type
// for just that, and pass it into the function. Notably, the MIG device's UUID
// does not need to be known here (I hope -- should the MIG device UUID from the
// allocated claim be checked against the to-be-deleted MIG dqevice? Is there
// _any_ chance that the "same" MIG device was re-created in the meantime? In
// that case it might have the same 3-tuple, but a different UUID. Further
// questions: are MIG UUIDs random? Can a MIG UUID be looked up given the
// 3-tuple?
func (l deviceLib) deleteMigDevice(parentUUID string, giId int, ciId int) error {
	migStr := fmt.Sprintf("MIG(%s, %d, %d)", parentUUID, giId, ciId)
	klog.V(6).Infof("Delete MIG device: %s", migStr)

	// if err := l.Init(); err != nil {
	// 	return err
	// }
	// defer l.alwaysShutdown()

	//parentNvmlDev, ret := l.nvmllib.DeviceGetHandleByUUID(parentUUID)
	parentNvmlDev, ret := l.DeviceGetHandleByUUIDCached(parentUUID)
	if ret != nvml.SUCCESS {
		return fmt.Errorf("error getting device from UUID '%v': %v", parentUUID, ret)
	}

	// The order of destroying 1) compute instance and 2) GPU instance matters.
	// These resources are hierarchical: compute instances are created inside a
	// GPU instance, so the parent GPU instance cannot be destroyed while
	// children (compute instances) still exist.
	gi, gires := parentNvmlDev.GetGpuInstanceById(giId)

	// Ref docs document this error with "If device doesn't have MIG mode
	// enabled" -- for the unlikely case that we end up in this state (MIG mode
	// was disabled out-of-band?), this should be treated as deletion success.
	if gires == nvml.ERROR_NOT_SUPPORTED {
		klog.Infof("Delete %s: GetGpuInstanceById yielded ERROR_NOT_SUPPORTED: MIG disabled, treat as success", migStr)
		return nil
	}

	// UNINITIALIZED, INVALID_ARGUMENT, NO_PERMISSION
	if gires != nvml.SUCCESS && gires != nvml.ERROR_NOT_FOUND {
		return fmt.Errorf("error getting GPU instance handle for MIG device: %v", ret)
	}

	if gires == nvml.ERROR_NOT_FOUND {
		// In this case assume that no compute instances exist (as of the GI>CI
		// hierarchy) and proceed with attempt-to-disable-MIG-mode
		klog.Infof("Delete %s: GI was not found skip CI cleanup", migStr)
		if err := l.maybeDisableMigMode(parentUUID, parentNvmlDev); err != nil {
			return fmt.Errorf("failed maybeDisableMigMode: %w", err)
		}
		return nil
	}

	// Remainder, with `gi` actually being valid.
	ci, cires := gi.GetComputeInstanceById(ciId)

	// Can never be `ERROR_NOT_SUPPORTED` at this point. Can be UNINITIALIZED,
	// INVALID_ARGUMENT, NO_PERMISSION: for those three, it's worth erroring out
	// here (to be retried later).
	if cires != nvml.SUCCESS && cires != nvml.ERROR_NOT_FOUND {
		return fmt.Errorf("error getting Compute instance handle for MIG device %s: %v", migStr, ret)
	}

	// A previous, partial cleanup may actually have already deleted that. Seen
	// in practice. Ignore, and proceed with deleting GPU instance below.
	if cires == nvml.ERROR_NOT_FOUND {
		klog.Infof("Delete %s: CI not found, ignore", migStr)
	} else {
		ret := ci.Destroy()
		if ret != nvml.SUCCESS {
			return fmt.Errorf("error destroying Compute instance: %v", ret)
		}
	}

	// That can for example fail with "In use by another client", in which case
	// we may have performed only a partical cleanup (CI already destroyed; seen
	// in practice).

	// Note that this operation may take O(1 s). In a machine supporting many
	// MIG devices and significant job throughput, this may become noticeable.
	// In a stressing test, I have seen the prep/unprep lock acquisition time
	// out after 10 seconds, when requests pile up.
	ret = gi.Destroy()
	if ret != nvml.SUCCESS {
		return fmt.Errorf("error destroying GPU Instance: %v", ret)
	}

	if err := l.maybeDisableMigMode(parentUUID, parentNvmlDev); err != nil {
		return fmt.Errorf("failed maybeDisableMigMode: %w", err)
	}

	return nil
}

func (l deviceLib) maybeDisableMigMode(uuid string, nvmldev nvml.Device) error {
	// Expect the parent GPU to be represented in in `l.gpuInfosByUUID`
	gpu, ok := l.gpuInfosByUUID[uuid]
	if !ok {
		// TODO: this is a programming error -- panic instead
		return fmt.Errorf("uuid not in gpuInfosByUUID: %s", uuid)
	}

	migs, err := l.getMigDevices(gpu)
	if err != nil {
		return fmt.Errorf("error getting MIG devices for %s: %w", gpu.String(), err)
	}

	if len(migs) > 0 {
		klog.V(6).Infof("Leaving MIG mode enabled for device %s (currently present MIG devices: %d)", gpu.String(), len(migs))
		return nil
	}

	klog.V(6).Infof("Attempting to disable MIG mode for device %s", gpu.String())
	ret, activationStatus := nvmldev.SetMigMode(nvml.DEVICE_MIG_DISABLE)
	if ret != nvml.SUCCESS {
		// activationStatus would return the appropriate error code upon unsuccessful activation
		klog.Warningf("SetMigMode activationStatus (device %s): %s", gpu.String(), activationStatus)
		// We could also log this as an error and proceed, and hope for the
		// state machine to clean this up in the future. Probably not a good
		// idea.
		return fmt.Errorf("error disabling MIG mode for device %s: %v", gpu.String(), ret)
	}
	klog.V(1).Infof("MIG mode now disabled for device %s", gpu.String())
	return nil
}

// Returns a flat list of all potentially possible incarnated MIG devices.
// Specifically, it discovers all possible abstract profiles (device types),
// then determines each specific placement for each profile.
func (l deviceLib) inspectMigProfilesAndPlacements(gpuInfo *GpuInfo, device nvdev.Device) ([]*MigInfo, error) {
	var infos []*MigInfo

	maxCapacities := make(PartCapacityMap)
	maxMemSlicesConsumed := 0

	err := device.VisitMigProfiles(func(migProfile nvdev.MigProfile) error {
		if migProfile.GetInfo().C != migProfile.GetInfo().G {
			return nil
		}

		if migProfile.GetInfo().CIProfileID == nvml.COMPUTE_INSTANCE_PROFILE_1_SLICE_REV1 {
			return nil
		}

		giProfileInfo, ret := device.GetGpuInstanceProfileInfo(migProfile.GetInfo().GIProfileID)
		if ret == nvml.ERROR_NOT_SUPPORTED {
			return nil
		}
		if ret == nvml.ERROR_INVALID_ARGUMENT {
			return nil
		}
		if ret != nvml.SUCCESS {
			return fmt.Errorf("error getting GI Profile info for MIG profile %v: %w", migProfile, ret)
		}

		giPlacements, ret := device.GetGpuInstancePossiblePlacements(&giProfileInfo)
		if ret == nvml.ERROR_NOT_SUPPORTED {
			return nil
		}
		if ret == nvml.ERROR_INVALID_ARGUMENT {
			return nil
		}
		if ret != nvml.SUCCESS {
			return fmt.Errorf("error getting GI possible placements for MIG profile %v: %w", migProfile, ret)
		}

		for _, giPlacement := range giPlacements {
			mi := &MigInfo{
				Parent:        gpuInfo,
				Profile:       migProfile,
				GIProfileInfo: giProfileInfo,
				MemorySlices:  giPlacement,
			}
			infos = append(infos, mi)

			// Assume that the largest MIG profile consumes all memory slices,
			// and hence we can infer the memory slice count by looking at the
			// Size property of all MigPP objects, and picking the maximum.
			maxMemSlicesConsumed = max(maxMemSlicesConsumed, int(giPlacement.Size))

			// Across all MIG profiles, identify the largest value for each
			// capacity dimension. They probably all corresponding to the same
			// profile.
			caps := mi.PartCapacities()
			for name, cap := range caps {
				setMax(maxCapacities, name, cap)
			}
		}
		return nil
	})

	klog.Infof("Per-capacity maximum across all MIG profiles+placements: %v", maxCapacities)
	klog.Infof("Largest MIG placement size seen (maxMemSlicesConsumed): %d", maxMemSlicesConsumed)

	if err != nil {
		return nil, fmt.Errorf("error visiting MIG profiles: %w", err)
	}

	// Mutate the full-device information container `gpuInfo`; enrich it with
	// detail obtained from walking MIG devices. Assume that the largest MIG
	// profile seen consumes all memory slices; equate maxMemSlicesConsumed =
	// memSliceCount.
	gpuInfo.AddDetailAfterWalkingMigProfiles(maxCapacities, maxMemSlicesConsumed)
	return infos, nil
}

// Mutate map `m` in-place: insert into map if the current QualifiedName does
// not yet exist as a key. Otherwise, update item in map if the incoming value
// `v` is larger than the one currenty stored in the map.
func setMax(m map[resourceapi.QualifiedName]resourceapi.DeviceCapacity, k resourceapi.QualifiedName, v resourceapi.DeviceCapacity) {
	if cur, ok := m[k]; !ok || v.Value.Value() > cur.Value.Value() {
		m[k] = v
	}
}

// copied straight from CD plugin
// that for examplex returns 510 when /proc/devices contains
// 510 nvidia-caps
func getDeviceMajor(name string) (int, error) {

	re := regexp.MustCompile(
		// The `(?s)` flag makes `.` match newlines. The greedy modifier in
		// `.*?` ensures to pick the first match after "Character devices".
		// Extract the number as capture group (the first and only group).
		"(?s)Character devices:.*?" +
			"([0-9]+) " + regexp.QuoteMeta(name) +
			// Require `name` to be newline-terminated (to not match on a device
			// that has `name` as prefix).
			"\n.*Block devices:",
	)

	data, err := os.ReadFile(procDevicesPath)
	if err != nil {
		return -1, fmt.Errorf("error reading '%s': %w", procDevicesPath, err)
	}

	// Expect precisely one match: first element is the total match, second
	// element corresponds to first capture group within that match (i.e., the
	// number of interest).
	matches := re.FindStringSubmatch(string(data))
	if len(matches) != 2 {
		return -1, fmt.Errorf("error parsing '%s': unexpected regex match: %v", procDevicesPath, matches)
	}

	// Convert capture group content to integer. Perform upper bound check:
	// value must fit into 32-bit integer (it's then also guaranteed to fit into
	// a 32-bit unsigned integer, which is the type that must be passed to
	// unix.Mkdev()).
	major, err := strconv.ParseInt(matches[1], 10, 32)
	if err != nil {
		return -1, fmt.Errorf("int conversion failed for '%v': %w", matches[1], err)
	}

	// ParseInt() always returns an integer of explicit type `int64`. We have
	// performed an upper bound check so it's safe to convert this to `int`
	// (which is documented as "int is a signed integer type that is at least 32
	// bits in size", so in theory it could be smaller than int64).
	return int(major), nil
}

func parseNVCapDeviceInfo(nvcapsFilePath string) (*nvcapDeviceInfo, error) {
	// klog.V(6).Infof("Parse %s", nvcapsFilePath)

	file, err := os.Open(nvcapsFilePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	info := &nvcapDeviceInfo{}

	major, err := getDeviceMajor(nvidiaCapsDeviceName)
	if err != nil {
		return nil, fmt.Errorf("error getting device major: %w", err)
	}

	klog.V(7).Infof("Got major for %s: %d", nvidiaCapsDeviceName, major)
	info.major = major

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.SplitN(line, ":", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		value := strings.TrimSpace(parts[1])

		switch key {
		case "DeviceFileMinor":
			_, _ = fmt.Sscanf(value, "%d", &info.minor)
		case "DeviceFileMode":
			_, _ = fmt.Sscanf(value, "%d", &info.mode)
		case "DeviceFileModify":
			_, _ = fmt.Sscanf(value, "%d", &info.modify)
		}
	}
	info.path = fmt.Sprintf("/dev/nvidia-caps/nvidia-cap%d", info.minor)

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return info, nil
}
