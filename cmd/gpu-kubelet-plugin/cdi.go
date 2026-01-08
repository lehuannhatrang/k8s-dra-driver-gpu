/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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
	"fmt"
	"os"
	"time"

	"github.com/davecgh/go-spew/spew"
	"github.com/sirupsen/logrus"

	nvdevice "github.com/NVIDIA/go-nvlib/pkg/nvlib/device"
	"github.com/NVIDIA/go-nvml/pkg/nvml"
	"github.com/NVIDIA/nvidia-container-toolkit/pkg/nvcdi"
	"github.com/NVIDIA/nvidia-container-toolkit/pkg/nvcdi/spec"
	transformroot "github.com/NVIDIA/nvidia-container-toolkit/pkg/nvcdi/transform/root"
	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"

	utilcache "k8s.io/apimachinery/pkg/util/cache"
	cdiapi "tags.cncf.io/container-device-interface/pkg/cdi"
	cdiparser "tags.cncf.io/container-device-interface/pkg/parser"
	cdispec "tags.cncf.io/container-device-interface/specs-go"

	"github.com/NVIDIA/k8s-dra-driver-gpu/internal/common"
	"github.com/NVIDIA/k8s-dra-driver-gpu/pkg/featuregates"
)

const (
	cdiVendor = "k8s." + DriverName

	cdiDeviceClass = "device"
	cdiDeviceKind  = cdiVendor + "/" + cdiDeviceClass
	cdiClaimClass = "claim"
	cdiClaimKind  = cdiVendor + "/" + cdiClaimClass

	cdiBaseSpecIdentifier = "base"
	cdiVfioSpecIdentifier = "vfio"

	defaultCDIRoot = "/var/run/cdi"
	procNvCapsPath = "/proc/driver/nvidia/capabilities"
)

type CDIHandler struct {
	logger   *logrus.Logger
	nvml     nvml.Interface
	nvdevice nvdevice.Interface
	nvcdiDevice       nvcdi.Interface
	nvcdiClaim        nvcdi.Interface
	cache             *cdiapi.Cache
	driverRoot        string
	devRoot           string
	targetDriverRoot  string
	nvidiaCDIHookPath string

	specCache *utilcache.Expiring

	cdiRoot string
	vendor  string
	//deviceClass string
	claimClass string
}

func NewCDIHandler(opts ...cdiOption) (*CDIHandler, error) {
	h := &CDIHandler{}
	for _, opt := range opts {
		opt(h)
	}

	if h.logger == nil {
		h.logger = logrus.New()
		// h.logger.SetOutput(io.Discard)
	}
	if h.nvml == nil {
		h.nvml = nvml.New()
	}
	if h.cdiRoot == "" {
		h.cdiRoot = defaultCDIRoot
	}
	if h.nvdevice == nil {
		h.nvdevice = nvdevice.New(h.nvml)
	}
	if h.vendor == "" {
		h.vendor = cdiVendor
	}
	// if h.deviceClass == "" {
	// 	h.deviceClass = cdiDeviceClass
	// }
	if h.claimClass == "" {
		h.claimClass = cdiClaimClass
	}
	if h.nvcdiDevice == nil {
		nvcdilib, err := nvcdi.New(
			nvcdi.WithDeviceLib(h.nvdevice),
			nvcdi.WithDriverRoot(h.driverRoot),
			nvcdi.WithDevRoot(h.devRoot),
			nvcdi.WithLogger(h.logger),
			nvcdi.WithNvmlLib(h.nvml),
			nvcdi.WithMode("nvml"),
			nvcdi.WithVendor(h.vendor),
			nvcdi.WithClass(h.deviceClass),
			nvcdi.WithNVIDIACDIHookPath(h.nvidiaCDIHookPath),
		)
		if err != nil {
			return nil, fmt.Errorf("unable to create CDI library for devices: %w", err)
		}
		h.nvcdiDevice = nvcdilib
	}
	if h.nvcdiClaim == nil {
		nvcdilib, err := nvcdi.New(
			nvcdi.WithDeviceLib(h.nvdevice),
			nvcdi.WithDriverRoot(h.driverRoot),
			nvcdi.WithDevRoot(h.devRoot),
			nvcdi.WithLogger(h.logger),
			nvcdi.WithNvmlLib(h.nvml),
			nvcdi.WithMode("nvml"),
			nvcdi.WithVendor(h.vendor),
			nvcdi.WithClass(h.claimClass),
			nvcdi.WithNVIDIACDIHookPath(h.nvidiaCDIHookPath),
			nvcdi.WithFeatureFlags(nvcdi.FeatureDisableNvsandboxUtils),
			//vcdi.WithNvsandboxuitilsLib(nil),
		)
		if err != nil {
			return nil, fmt.Errorf("unable to create CDI library for claims: %w", err)
		}
		h.nvcdiClaim = nvcdilib
	}
	if h.cache == nil {
		cache, err := cdiapi.NewCache(
			cdiapi.WithSpecDirs(h.cdiRoot),
		)
		if err != nil {
			return nil, fmt.Errorf("unable to create a new CDI cache: %w", err)
		}
		h.cache = cache
	}

	h.specCache = utilcache.NewExpiring()

	return h, nil
}

func (cdi *CDIHandler) writeSpec(spec spec.Interface, specName string) error {
	// Transform the spec to make it aware that it is running inside a container.
	err := transformroot.New(
		transformroot.WithRoot(cdi.driverRoot),
		transformroot.WithTargetRoot(cdi.targetDriverRoot),
		transformroot.WithRelativeTo("host"),
	).Transform(spec.Raw())
	if err != nil {
		return fmt.Errorf("failed to transform driver root in CDI spec: %w", err)
	}

	// Update the spec to include only the minimum version necessary.
	minVersion, err := cdispec.MinimumRequiredVersion(spec.Raw())
	if err != nil {
		return fmt.Errorf("failed to get minimum required CDI spec version: %w", err)
	}
	spec.Raw().Version = minVersion

	// Write the spec out to disk.
	return cdi.cache.WriteSpec(spec.Raw(), specName)

}

func (cdi *CDIHandler) CreateStandardDeviceSpecFile(allocatable AllocatableDevices) error {
	if err := cdi.createStandardNvidiaDeviceSpecFile(allocatable); err != nil {
		klog.Errorf("failed to create standard nvidia device spec file: %v", err)
		return err
	}

	if featuregates.Enabled(featuregates.PassthroughSupport) {
		if err := cdi.createStandardVfioDeviceSpecFile(allocatable); err != nil {
			klog.Errorf("failed to create standard vfio device spec file: %v", err)
			return err
		}
	}
	return nil
}

func (cdi *CDIHandler) createStandardVfioDeviceSpecFile(allocatable AllocatableDevices) error {
	commonEdits := GetVfioCommonCDIContainerEdits()
	var deviceSpecs []cdispec.Device
	for _, device := range allocatable {
		if device.Type() != VfioDeviceType {
			continue
		}
		edits := GetVfioCDIContainerEdits(device.Vfio)
		dspec := cdispec.Device{
			Name:           device.CanonicalName(),
			ContainerEdits: *edits.ContainerEdits,
		}
		deviceSpecs = append(deviceSpecs, dspec)
	}

	if len(deviceSpecs) == 0 {
		return nil
	}

	spec, err := spec.New(
		spec.WithVendor(cdiVendor),
		spec.WithClass(cdiDeviceClass),
		spec.WithDeviceSpecs(deviceSpecs),
		spec.WithEdits(*commonEdits.ContainerEdits),
	)
	if err != nil {
		return fmt.Errorf("failed to creat CDI spec: %w", err)
	}

	specName := cdiapi.GenerateTransientSpecName(cdiVendor, cdiDeviceClass, cdiVfioSpecIdentifier)
	klog.Infof("Writing vfio spec for %s to %s", specName, cdi.cdiRoot)
	return cdi.writeSpec(spec, specName)
}

func (cdi *CDIHandler) createStandardNvidiaDeviceSpecFile(allocatable AllocatableDevices) error {
	// Initialize NVML in order to get the device edits.
	if r := cdi.nvml.Init(); r != nvml.SUCCESS {
		return fmt.Errorf("failed to initialize NVML: %v", r)
	}
	defer func() {
		if r := cdi.nvml.Shutdown(); r != nvml.SUCCESS {
			klog.Warningf("failed to shutdown NVML: %v", r)
		}
	}()

	// Generate the set of common edits.
	commonEdits, err := cdi.nvcdiDevice.GetCommonEdits()
	if err != nil {
		return fmt.Errorf("failed to get common CDI spec edits: %w", err)
	}

	// Make sure that NVIDIA_VISIBLE_DEVICES is set to void to avoid the
	// nvidia-container-runtime honoring it in addition to the underlying
	// runtime honoring CDI.
	commonEdits.Env = append(
		commonEdits.Env,
		"NVIDIA_VISIBLE_DEVICES=void")

	// Generate device specs for all full GPUs and MIG devices.
	var deviceSpecs []cdispec.Device
	for _, device := range allocatable {
		if device.Type() == VfioDeviceType {
			continue
		}

		uuid := device.UUID()
		if device.Type() == MigDeviceType {
			// Goal: inject parent dev node. Other dev nodes specific to this
			// MIG device are injected 'manually' further below. That is because
			// currently `nvcdiDevice.GetDeviceSpecsByID()` may yield an
			// incomplete spec for MIG devices, see
			// https://github.com/NVIDIA/k8s-dra-driver-gpu/issues/787. Instead,
			// manually create the required dev node spec for the consuming
			// container via GetDevNodesForMigDevice() below.
			uuid = device.Mig.Parent.UUID
		}

		dspecs, err := cdi.nvcdiDevice.GetDeviceSpecsByID(uuid)
		if err != nil {
			return fmt.Errorf("unable to get device spec for %s: %w", device.CanonicalName(), err)
		}
		dspecs[0].Name = device.CanonicalName()

		if device.Type() == MigDeviceType {
			devnodesForMig, err := cdi.GetDevNodesForMigDevice(device.Mig.parent.minor, int(device.Mig.giInfo.Id), int(device.Mig.ciInfo.Id))
			if err != nil {
				return fmt.Errorf("failed to construct MIG device DeviceNode edits: %w", err)
			}
			klog.V(7).Infof("CDI spec: appending MIG device nodes")
			dspecs[0].ContainerEdits.DeviceNodes = append(dspecs[0].ContainerEdits.DeviceNodes, devnodesForMig...)
		}

		deviceSpecs = append(deviceSpecs, dspecs[0])
	}

	// Generate base spec from commonEdits and deviceEdits.
	spec, err := spec.New(
		spec.WithVendor(cdiVendor),
		spec.WithClass(cdiDeviceClass),
		spec.WithDeviceSpecs(deviceSpecs),
		spec.WithEdits(*commonEdits.ContainerEdits),
	)
	if err != nil {
		return fmt.Errorf("failed to creat CDI spec: %w", err)
	}

	specName := cdiapi.GenerateTransientSpecName(cdiVendor, cdiDeviceClass, cdiBaseSpecIdentifier)
	klog.Infof("Writing spec for %s to %s", specName, cdi.cdiRoot)
	return cdi.writeSpec(spec, specName)
}

// Construct and return the CDI `deviceNodes` specification for the two
// character devices `/dev/nvidia-caps/nvidia-cap<CIm>` and
// `/dev/nvidia-caps/nvidia-cap<GIm>` for a specific MIG device.
//
// Context: for containerized workload to see and use a specific MIG device, it
// needs to be able to open three character device nodes:
//
// 1) `/dev/nvidia<Pm>`, with <Pm> referring to the parent's minor. This exists
// on the host.
//
// 2) /dev/nvidia-caps/nvidia-cap<CIm> and /dev/nvidia-caps/nvidia-cap<GIm>,
// with <GIm> and <CIm> referring to the MIG GPU instance's and Compute
// instance's minor, respectively. For the the latter two device nodes it is
// sufficient to create them in the container (with proper cgroups permissions),
// without actually requiring the same device nodes to be explicitly created on
// the host. That is what is achieved below with the structure created in
// cdiDevNodeFromNVCapDevInfo().
func (cdi *CDIHandler) GetDevNodesForMigDevice(mig *MigDeviceInfo) ([]*cdispec.DeviceNode, error) {
	gpath := fmt.Sprintf("%s/gpu%d/mig/gi%d/access", procNvCapsPath, mig.parent.minor, mig.GIID)
	giCapsInfo, err := parseNVCapDeviceInfo(gpath)
	if err != nil {
		return nil, fmt.Errorf("failed to parse GI capabilities file %s: %w", gpath, err)
	}

	cpath := fmt.Sprintf("%s/gpu%d/mig/gi%d/ci%d/access", procNvCapsPath, mig.parent.minor, mig.GIID, mig.CIID)
	ciCapsInfo, err := parseNVCapDeviceInfo(cpath)
	if err != nil {
		return nil, fmt.Errorf("failed to parse CI capabilities file %s: %w", cpath, err)
	}

	devnodes := []*cdispec.DeviceNode{cdiCharDevNode(giCapsInfo), cdiCharDevNode(ciCapsInfo)}
	spew.Printf("%s=%#v\n", "devnodes", devnodes)
	return devnodes, nil
}

// Construct and return a CDI `deviceNodes` entry. Adding this to a CDI
// container specification at the high level has the purpose of granting cgroup
// access to the containerized application for being able to access (open) a
// certain character device node as identified by `i.path`. The special device
// type "c" below specifically instructs the container runtime to create (mknod)
// the character device (for the container, accessible from within the
// container, not visible on the host), and to grant the cgroup privilege to the
// container to open that device. References:
// https://github.com/opencontainers/runtime-spec/blob/main/config-linux.md#allowed-device-list
// https://www.kernel.org/doc/Documentation/cgroup-v1/devices.txt
func cdiCharDevNode(i *nvcapDeviceInfo) *cdispec.DeviceNode {
	return &cdispec.DeviceNode{
		Path:     i.path,
		Type:     "c",
		FileMode: ptr.To(os.FileMode(i.mode)),
		Major:    int64(i.major),
		Minor:    int64(i.minor),
	}
}

func (cdi *CDIHandler) GetCommonEditsCached() (*cdiapi.ContainerEdits, error) {
	key := "commonEdits"
	if v, ok := cdi.specCache.Get(key); ok {
		edits := v.(*cdiapi.ContainerEdits)
		// Return a shallow copy so that cache entry consumer is less likely to
		// mutate the cache entry.
		clone := *edits
		return &clone, nil
	}

	// Initialize NVML in order to get the device edits.
	// if r := cdi.nvml.Init(); r != nvml.SUCCESS {
	// 	return nil, fmt.Errorf("failed to initialize NVML: %v", r)
	// }
	// defer func() {
	// 	if r := cdi.nvml.Shutdown(); r != nvml.SUCCESS {
	// 		klog.Warningf("failed to shutdown NVML: %v", r)
	// 	}
	// }()

	t0 := time.Now()
	v, err := cdi.nvcdiClaim.GetCommonEdits()
	klog.V(6).Infof("t_cdi_get_common_edits %.3f s", time.Since(t0).Seconds())

	if err != nil {
		return nil, err
	}
	cdi.specCache.Set(key, v, time.Duration(5*time.Minute))
	// Return a shallow copy, see above.
	clone := *v
	return &clone, nil
}

func (cdi *CDIHandler) WarmupDevSpecCache(uuids []string) {
	for _, uuid := range uuids {
		_, err := cdi.GetDeviceSpecsByUUIDCached(uuid)
		if err != nil {
			klog.Warningf("Ignore error during cache warmup: GetDeviceSpecsByUUIDCached() failed: %s", err)
		}
	}
}

func (cdi *CDIHandler) GetDeviceSpecsByUUIDCached(uuid string) ([]cdispec.Device, error) {
	key := uuid
	if v, ok := cdi.specCache.Get(key); ok {
		devs := v.([]cdispec.Device)
		clone := make([]cdispec.Device, len(devs))
		copy(clone, devs)
		return clone, nil
	}

	// Initialize NVML in order to get the device edits.
	// if r := cdi.nvml.Init(); r != nvml.SUCCESS {
	// 	return nil, fmt.Errorf("failed to initialize NVML: %v", r)
	// }
	// defer func() {
	// 	if r := cdi.nvml.Shutdown(); r != nvml.SUCCESS {
	// 		klog.Warningf("failed to shutdown NVML: %v", r)
	// 	}
	// }()

	t0 := time.Now()
	// This has been called 28 times
	devs, err := cdi.nvcdiClaim.GetDeviceSpecsByID(uuid)
	klog.Infof("GetDeviceSpecsByID() called for %s", uuid)
	klog.V(6).Infof("t_cdi_get_specs_for_uuid %.3f s", time.Since(t0).Seconds())
	if err != nil {
		return nil, err
	}
	cdi.specCache.Set(key, devs, time.Duration(5*time.Minute))
	clone := make([]cdispec.Device, len(devs))
	copy(clone, devs)
	return clone, nil
}

func (cdi *CDIHandler) CreateClaimSpecFile(claimUID string, preparedDevices PreparedDevices) error {
	// Generate those parts of the container spec that are note device-specific.
	// To inject things like driver library mounts and meta devices.
	// commonEdits, err := cdi.nvcdiDevice.GetCommonEdits() this may initialize
	// nvsandboxutilslib under the hood cdi.nvcdiClaim.GetCommonEdits()
	commonEdits, err := cdi.GetCommonEditsCached()
	if err != nil {
		return fmt.Errorf("failed to get common CDI spec edits: %w", err)
	}

	// Make sure that NVIDIA_VISIBLE_DEVICES is set to void to avoid the
	// nvidia-container-runtime honoring it in addition to the underlying
	// runtime honoring CDI.
	commonEdits.Env = append(
		commonEdits.Env,
		"NVIDIA_VISIBLE_DEVICES=void")

	var deviceSpecs []cdispec.Device
	for _, device := range allocatable {
		if device.Type() == VfioDeviceType {
			continue
		}

		uuid := device.UUID()
		if device.Type() == MigDeviceType {
			// Goal: inject parent dev node. Other dev nodes specific to this
			// MIG device are injected 'manually' further below. That is because
			// currently `nvcdiDevice.GetDeviceSpecsByID()` may yield an
			// incomplete spec for MIG devices, see
			// https://github.com/NVIDIA/k8s-dra-driver-gpu/issues/787. Instead,
			// manually create the required dev node spec for the consuming
			// container via GetDevNodesForMigDevice() below.
			uuid = device.Mig.parent.UUID
		}

		dspecs, err := cdi.nvcdiDevice.GetDeviceSpecsByID(uuid)
		if err != nil {
			return fmt.Errorf("unable to get device spec for %s: %w", device.CanonicalName(), err)
		}
		dspecs[0].Name = device.CanonicalName()

		if device.Type() == MigDeviceType {
			devnodesForMig, err := cdi.GetDevNodesForMigDevice(device.Mig.parent.minor, int(device.Mig.giInfo.Id), int(device.Mig.ciInfo.Id))
			if err != nil {
				return fmt.Errorf("failed to construct MIG device DeviceNode edits: %w", err)
			}
			klog.V(7).Infof("CDI spec: appending MIG device nodes")
			dspecs[0].ContainerEdits.DeviceNodes = append(dspecs[0].ContainerEdits.DeviceNodes, devnodesForMig...)
		}

		deviceSpecs = append(deviceSpecs, dspecs[0])
	}

	// Generate base spec from commonEdits and deviceEdits.
	spec, err := spec.New(
		spec.WithVendor(cdiVendor),
		spec.WithClass(cdiDeviceClass),
		spec.WithDeviceSpecs(deviceSpecs),
		spec.WithEdits(*commonEdits.ContainerEdits),
	)
	if err != nil {
		return fmt.Errorf("failed to creat CDI spec: %w", err)
	}

	specName := cdiapi.GenerateTransientSpecName(cdiVendor, cdiDeviceClass, cdiBaseSpecIdentifier)
	klog.Infof("Writing spec for %s to %s", specName, cdi.cdiRoot)
	return cdi.writeSpec(spec, specName)
}

func (cdi *CDIHandler) CreateClaimSpecFile(claimUID string, preparedDevices PreparedDevices) error {
	// Generate claim specific specs for each device.
	var deviceSpecs []cdispec.Device

	for _, group := range preparedDevices {
		for _, dev := range group.Devices {
			duuid := ""

			// Construct claim-specific CDI device name in accordance with the
			// naming convention is encoded in GetClaimDeviceName() below.
			dname := fmt.Sprintf("%s-%s", claimUID, dev.CanonicalName())

			if dev.Mig != nil {
				// Inject parent dev node
				// Other dev nodes specific to this MIG device are injected 'manually' below.
				//duuid = dev.Mig.Created.UUID
				duuid = dev.Mig.Created.parent.UUID
			} else {
				duuid = dev.Gpu.Info.UUID
			}

			// For a just-created MIG device I see this emit a msg on stderr:
			//
			// ERROR: migGetDevFileInfo 212 result=11ERROR: init 310
			// result=11ERROR: migGetDevFileInfo 212 result=11ERROR: init 310
			// result=11
			//
			// Evan said that "Seems like nvsandboxutils " "There is a feature
			// flag in the CDI API to disable it, but you may need a code change
			// in the driver ..." Probably triggered here:
			// https://github.com/NVIDIA/nvidia-container-toolkit/blob/e03ac3644d63ec30849dffebd0170811e4903e78/internal/platform-support/dgpu/nvsandboxutils.go#L67
			//
			//klog.V(7).Infof("Call nvcdiDevice.GetDeviceSpecsByID(%s)", duuid)

			// Get (copy of) cached device spec (is safe to be mutated below,
			// w/o compromising cache).
			dspecs, err := cdi.GetDeviceSpecsByUUIDCached(duuid)
			if err != nil {
				return fmt.Errorf("unable to get device spec for %s: %w", dname, err)
			}

			// Note(JP): for a regular GPU, this canonical name is for example
			// `gpu-0`, with the numerical suffix as of the time of writing
			// reflecting the device minor. NVMLs' DeviceSetMigMode() is
			// documented with 'This API may unbind or reset the device to
			// activate the requested mode. Thus, the attributes associated with
			// the device, such as minor number, might change. The caller of
			// this API is expected to query such attributes again.' -- if the
			// minor is indeed not necessarily stable, there may be problems
			// associating this spec _long-term_ with that name. That is an
			// argument for always dynamically generating also full-GPU CDI spec
			// during prepare().
			dspecs[0].Name = dname

			//devnodes := dspecs[0].ContainerEdits.DeviceNodes
			//json.Marshal(devnodes)

			// I've found a situation where `GetDeviceSpecsByID(duuid)` returned
			// just one instead of three dev nodes; just the dev node for the
			// full device. That was because the mig device-specific nodes e.g.
			// `nvidia-cap66  nvidia-cap67` were not in /dev/nvidia-caps. Then,
			// we would proceed with just the full GPU gets injected (which
			// hopefully cannot actually be used to launch workload on). It's
			// yet unclear to me what the exact conditions were for that to
			// happen, but we should error out in that case. The dev nodes were
			// present in the container nvidia-mig-manager-zz5h8 at
			// /dev/nvidia-caps but not at /driver-root/dev/nvidia-caps
			//
			// Later I saw, with CDI library logging enabled:
			// time="2025-11-02T21:38:00Z" level=info msg="Selecting /driver-root/dev/nvidia0 as /dev/nvidia0"
			// time="2025-11-02T21:38:00Z" level=warning msg="Could not locate /dev/nvidia-caps/nvidia-cap66: pattern /dev/nvidia-caps/nvidia-cap66 not found"
			// time="2025-11-02T21:38:00Z" level=warning msg="Could not locate /dev/nvidia-caps/nvidia-cap67: pattern /dev/nvidia-caps/nvidia-cap67 not found"
			// editsForDevice, err := edits.FromDiscoverer(deviceNodes)
			//
			// if dev.Mig != nil && len(devnodes) < 3 {
			// 	klog.Warningf("insufficent number of dev nodes returned for MIG device by GetDeviceSpecsByID(): %d", len(devnodes))
			// 	//return fmt.Errorf("bad result from GetDeviceSpecsByID() for MIG device -- restart GPU Operator?")
			// }
			if dev.Mig != nil {
				devnodesForMig, err := cdi.GetDevNodesForMigDevice(dev.Mig.Created)
				if err != nil {
					return fmt.Errorf("failed to construct MIG device DeviceNode edits: %w", err)
				}
				klog.V(7).Infof("CDI spec: appending MIG device nodes")
				dspecs[0].ContainerEdits.DeviceNodes = append(dspecs[0].ContainerEdits.DeviceNodes, devnodesForMig...)
			}

			deviceSpecs = append(deviceSpecs, dspecs[0])

			klog.V(7).Infof("Number of device nodes about to inject for device %s: %d", dname, len(dspecs[0].ContainerEdits.DeviceNodes))

			// spew.Printf("%s=%#v\n", "deviceSpecs", deviceSpecs)
			// If there edits passed as part of the device config state (set on
			// the group), add them to the spec of each device in that group.
			if group.ConfigState.containerEdits != nil {
				deviceSpec := cdispec.Device{
					Name:           fmt.Sprintf("%s-%s", claimUID, dname),
					ContainerEdits: *group.ConfigState.containerEdits.ContainerEdits,
				}
				deviceSpecs = append(deviceSpecs, deviceSpec)
			}
		}
	}

	tws0 := time.Now()

	spec, err := spec.New(
		spec.WithVendor(cdiVendor),
		spec.WithClass(cdiClaimClass),
		spec.WithDeviceSpecs(deviceSpecs),
		spec.WithEdits(*commonEdits.ContainerEdits),
	)
	if err != nil {
		return fmt.Errorf("failed to create CDI spec: %w", err)
	}

	// Transform the spec to make it aware that it is running inside a container.
	err = transformroot.New(
		transformroot.WithRoot(cdi.driverRoot),
		transformroot.WithTargetRoot(cdi.targetDriverRoot),
		transformroot.WithRelativeTo("host"),
	).Transform(spec.Raw())
	if err != nil {
		return fmt.Errorf("failed to transform driver root in CDI spec: %w", err)
	}

	minVersion, err := cdispec.MinimumRequiredVersion(spec.Raw())
	if err != nil {
		return fmt.Errorf("failed to get minimum required CDI spec version: %w", err)
	}
	spec.Raw().Version = minVersion

	// Write the per-claim spec that was generated above to the filesystem. As
	// it is bound to a DRA ResourceClaim, it's transient (bound to the lifetime
	// of a container). Hence, Use the "transient spec" concept from CDI.
	specName := cdiapi.GenerateTransientSpecName(cdiVendor, cdiClaimClass, claimUID)
	klog.V(6).Infof("Writing CDI spec '%s' for claim '%s'", specName, claimUID)
	result := cdi.cache.WriteSpec(spec.Raw(), specName)

	klog.V(6).Infof("t_gen_write_cdi_spec %.3f s", time.Since(tws0).Seconds())

	return result

	// I still wonder -- how is this discovered when starting the container?
	// Need to understand that information flow better.
}

func (cdi *CDIHandler) DeleteClaimSpecFile(claimUID string) error {
	specName := cdiapi.GenerateTransientSpecName(cdiVendor, cdiClaimClass, claimUID)
	klog.V(6).Infof("Delete CDI spec file: '%s', claim '%s'", specName, claimUID)
	return cdi.cache.RemoveSpec(specName)
}

// Philosophy: all devices to be injected into a container are defined in a
// single, transient CDI spec. This function returns the fully qualified
// identifier for a device defined in that spec. Example:
// k8s.gpu.nvidia.com/claim=dab5ab50-d59a-42a6-af16-cfd4628c0f7a-gpu-0
// That identifier can be used elsewhere, and _points to the spec_.
func (cdi *CDIHandler) GetClaimDeviceName(claimUID string, device *AllocatableDevice, containerEdits *cdiapi.ContainerEdits) string {
	return cdiparser.QualifiedName(cdiVendor, cdiClaimClass, fmt.Sprintf("%s-%s", claimUID, device.CanonicalName()))
}

// Construct and return the CDI `deviceNodes` specification for the two
// character devices `/dev/nvidia-caps/nvidia-cap<CIm>` and
// `/dev/nvidia-caps/nvidia-cap<GIm>` for a specific MIG device.
//
// Context: for containerized workload to see and use a specific MIG device, it
// needs to be able to open three character device nodes:
//
// 1) `/dev/nvidia<Pm>`, with <Pm> referring to the parent's minor. This exists
// on the host.
//
// 2) /dev/nvidia-caps/nvidia-cap<CIm> and /dev/nvidia-caps/nvidia-cap<GIm>,
// with <GIm> and <CIm> referring to the MIG GPU instance's and Compute
// instance's minor, respectively. For the the latter two device nodes it is
// sufficient to create them in the container (with proper cgroups permissions),
// without actually requiring the same device nodes to be explicitly created on
// the host. That is what is achieved below with the structure created in
// cdiDevNodeFromNVCapDevInfo().
func (cdi *CDIHandler) GetDevNodesForMigDevice(parentMinor int, giId int, ciId int) ([]*cdispec.DeviceNode, error) {
	gipath := fmt.Sprintf("%s/gpu%d/mig/gi%d/access", procNvCapsPath, parentMinor, giId)
	cipath := fmt.Sprintf("%s/gpu%d/mig/gi%d/ci%d/access", procNvCapsPath, parentMinor, giId, ciId)

	giCapsInfo, err := common.ParseNVCapDeviceInfo(gipath)
	if err != nil {
		return nil, fmt.Errorf("failed to parse GI capabilities file %s: %w", gipath, err)
	}

	ciCapsInfo, err := common.ParseNVCapDeviceInfo(cipath)
	if err != nil {
		return nil, fmt.Errorf("failed to parse CI capabilities file %s: %w", cipath, err)
	}

	devnodes := []*cdispec.DeviceNode{giCapsInfo.CDICharDevNode(), ciCapsInfo.CDICharDevNode()}
	return devnodes, nil
}
