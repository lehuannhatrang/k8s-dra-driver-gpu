/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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
	"context"
	"fmt"
	"io"
	"path/filepath"
	"slices"
	"sync"
	"time"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/dynamic-resource-allocation/kubeletplugin"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	cdiapi "tags.cncf.io/container-device-interface/pkg/cdi"

	"github.com/sirupsen/logrus"

	configapi "github.com/NVIDIA/k8s-dra-driver-gpu/api/nvidia.com/resource/v1beta1"
	"github.com/NVIDIA/k8s-dra-driver-gpu/pkg/featuregates"
	"github.com/NVIDIA/k8s-dra-driver-gpu/pkg/flock"
)

type OpaqueDeviceConfig struct {
	Requests []string
	Config   runtime.Object
}

type DeviceConfigState struct {
	MpsControlDaemonID string `json:"mpsControlDaemonID"`
	containerEdits     *cdiapi.ContainerEdits
}

type DeviceState struct {
	sync.Mutex
	cdi            *CDIHandler
	tsManager      *TimeSlicingManager
	mpsManager     *MpsManager
	vfioPciManager *VfioPciManager
	allocatable    AllocatableDevices
	config         *Config

	// Same set of allocatable devices as stored in `allocatable`, but grouped
	// by physical GPU. This is useful for grouped announcement (e.g., when
	// announcing one ResourceSlice per physical GPU).
	perGPUAllocatable PerGPUMinorAllocatableDevices

	nvdevlib          *deviceLib
	checkpointManager checkpointmanager.CheckpointManager
	// Checkpoint read/write lock, file-based for multi-process synchronization.

	cplock *flock.Flock
}

func NewDeviceState(ctx context.Context, config *Config) (*DeviceState, error) {
	containerDriverRoot := root(config.flags.containerDriverRoot)
	devRoot := containerDriverRoot.getDevRoot()
	klog.Infof("Using devRoot=%v", devRoot)

	nvdevlib, err := newDeviceLib(containerDriverRoot)
	if err != nil {
		return nil, fmt.Errorf("failed to create device library: %w", err)
	}

	allocatable, perGPUAllocatable, err := nvdevlib.enumerateAllPossibleDevices(config)
	if err != nil {
		return nil, fmt.Errorf("error enumerating all possible devices: %w", err)
	}

	hostDriverRoot := config.flags.hostDriverRoot

	// Let nvcdi logs see the light of day (emit to standard streams) when we've
	// been configured with verbosity level 7 or higher.
	cdilogger := logrus.New()
	if config.flags.klogVerbosity < 7 {
		klog.Infof("Muting CDI logger (verbosity is smaller 7: %d)", config.flags.klogVerbosity)
		cdilogger.SetOutput(io.Discard)
	}

	cdi, err := NewCDIHandler(
		WithNvml(nvdevlib.nvmllib),
		WithDeviceLib(nvdevlib),
		WithDriverRoot(string(containerDriverRoot)),
		WithDevRoot(devRoot),
		WithTargetDriverRoot(hostDriverRoot),
		WithNVIDIACDIHookPath(config.flags.nvidiaCDIHookPath),
		WithCDIRoot(config.flags.cdiRoot),
		WithVendor(cdiVendor),
		WithLogger(cdilogger),
	)
	if err != nil {
		return nil, fmt.Errorf("unable to create CDI handler: %w", err)
	}

	var fullGPUuuids []string
	for _, dev := range allocatable {
		if dev.Gpu != nil {
			fullGPUuuids = append(fullGPUuuids, dev.Gpu.UUID)
		}
	}
	klog.V(2).Infof("Warming up CDI device spec cache for GPUs %v", fullGPUuuids)
	cdi.WarmupDevSpecCache(fullGPUuuids)

	var tsManager *TimeSlicingManager
	if featuregates.Enabled(featuregates.TimeSlicingSettings) {
		tsManager = NewTimeSlicingManager(nvdevlib)
	}

	var mpsManager *MpsManager
	if featuregates.Enabled(featuregates.MPSSupport) {
		mpsManager = NewMpsManager(config, nvdevlib, hostDriverRoot, MpsControlDaemonTemplatePath)
	}

	var vfioPciManager *VfioPciManager
	if featuregates.Enabled(featuregates.PassthroughSupport) {
		vfioPciManager = NewVfioPciManager(string(containerDriverRoot), string(hostDriverRoot), nvdevlib, true /* nvidiaEnabled */)
	}

	// Validate passthrough support if feature gate is enabled.
	if featuregates.Enabled(featuregates.PassthroughSupport) {
		if err := vfioPciManager.ValidatePassthroughSupport(); err != nil {
			klog.Fatalf("Failed to validate passthrough support: %v", err)
		}
	}

	// if err := cdi.CreateStandardDeviceSpecFile(allocatable); err != nil {
	// 	return nil, fmt.Errorf("unable to create base CDI spec file: %v", err)
	// }

	checkpointManager, err := checkpointmanager.NewCheckpointManager(config.DriverPluginPath())
	if err != nil {
		return nil, fmt.Errorf("unable to create checkpoint manager: %v", err)
	}

	cpLockPath := filepath.Join(config.DriverPluginPath(), "cp.lock")

	state := &DeviceState{
		cdi:               cdi,
		tsManager:         tsManager,
		mpsManager:        mpsManager,
		vfioPciManager:    vfioPciManager,
		allocatable:       allocatable,
		perGPUAllocatable: perGPUAllocatable,
		config:            config,
		nvdevlib:          nvdevlib,
		checkpointManager: checkpointManager,
		cplock:            flock.NewFlock(cpLockPath),
	}
	state.allocatable = allocatable

	checkpoints, err := state.checkpointManager.ListCheckpoints()
	if err != nil {
		return nil, fmt.Errorf("unable to list checkpoints: %v", err)
	}

	for _, c := range checkpoints {
		if c == DriverPluginCheckpointFileBasename {
			return state, nil
		}
	}

	if err := state.createCheckpoint(ctx, &Checkpoint{}); err != nil {
		return nil, fmt.Errorf("unable to create fresh checkpoint: %v", err)
	}

	return state, nil
}

func (s *DeviceState) Prepare(ctx context.Context, claim *resourceapi.ResourceClaim) ([]kubeletplugin.Device, error) {
	// tplock0 := time.Now()
	// s.Lock()
	// defer s.Unlock()
	// klog.V(6).Infof("t_prep_state_lock_acq %.3f s", time.Since(tplock0).Seconds())

	claimUID := string(claim.UID)

	tgcp0 := time.Now()
	cp, err := s.getCheckpoint(ctx)
	if err != nil {
		return nil, fmt.Errorf("unable to get checkpoint: %v", err)
	}
	klog.V(6).Infof("t_prep_gcp %.3f s", time.Since(tgcp0).Seconds())

	// Check for existing 'completed' claim preparation before updating the
	// checkpoint with 'PrepareStarted'. Otherwise, we effectively mark a
	// perfectly prepared claim as only partially prepared, which may have
	// negative side effects during Unprepare() (currently a noop in this case:
	// unprepare noop: claim preparation started but not completed).
	preparedClaim, exists := cp.V2.PreparedClaims[claimUID]
	if exists && preparedClaim.CheckpointState == ClaimCheckpointStatePrepareCompleted {
		// Make this a noop. Associated device(s) has/ave been prepared by us.
		// Prepare() must be idempotent, as it may be invoked more than once per
		// claim (and actual device preparation must happen at most once).
		klog.V(6).Infof("skip prepare: claim %v found in checkpoint", claimUID)
		return preparedClaim.PreparedDevices.GetDevices(), nil
	}
	
	tucp0 := time.Now()
	err = s.updateCheckpoint(func(checkpoint *Checkpoint) {
		checkpoint.V2.PreparedClaims[claimUID] = PreparedClaim{
			CheckpointState: ClaimCheckpointStatePrepareStarted,
			Status:          claim.Status,
		}
	})
	if err != nil {
		return nil, fmt.Errorf("unable to update checkpoint: %w", err)
	}
	klog.V(6).Infof("checkpoint updated for claim %v", claimUID)
	klog.V(6).Infof("t_prep_ucp %.3f s", time.Since(tucp0).Seconds())

	tprep0 := time.Now()
	preparedDevices, err := s.prepareDevices(ctx, claim)
	klog.V(6).Infof("t_prep_core %.3f s (claim %s)", time.Since(tprep0).Seconds(), ResourceClaimToString(claim))
	if err != nil {
		return nil, fmt.Errorf("prepare devices failed: %w", err)
	}

	if featuregates.Enabled(featuregates.PassthroughSupport) {
		for _, device := range preparedDevices.GetDevices() {
			allocatableDevice, ok := s.allocatable[device.DeviceName]
			if !ok {
				klog.Warningf("allocatable not found for device: %v", device.DeviceName)
				continue
			}
			s.allocatable.RemoveSiblingDevices(allocatableDevice)
		}
	}

	tccsf0 := time.Now()
	if err := s.cdi.CreateClaimSpecFile(claimUID, preparedDevices); err != nil {
		return nil, fmt.Errorf("unable to create CDI spec file for claim: %w", err)
	}
	klog.V(6).Infof("t_prep_ccsf %.3f s", time.Since(tccsf0).Seconds())

	tucp20 := time.Now()
	err = s.updateCheckpoint(ctx, func(cp *Checkpoint) {
		cp.V2.PreparedClaims[claimUID] = PreparedClaim{
			CheckpointState: ClaimCheckpointStatePrepareCompleted,
			Status:          claim.Status,
			PreparedDevices: preparedDevices,
		}
	})
	if err != nil {
		return nil, fmt.Errorf("unable to update checkpoint: %w", err)
	}
	klog.V(6).Infof("checkpoint updated for claim %v", claimUID)
	klog.V(6).Infof("t_prep_ucp2 %.3f s", time.Since(tucp20).Seconds())

	return preparedDevices.GetDevices(), nil
}

// Quick&dirty; call me only once during startup for now -- before starting the
// driver logic (before accepting requests from the kubelet). This needs to be
// thought through properly.
func (s *DeviceState) DestroyUnknownMIGDevices(ctx context.Context) {
	logpfx := "Destroy unknown MIG devices"
	cp, err := s.getCheckpoint(ctx)
	if err != nil {
		klog.Errorf("%s: unable to get checkpoint: %s", logpfx, err)
		return
	}

	// Get checkpointed claims in PrepareCompleted state (explicitly not
	// PrepareStarted - those are of course good cleanup candidates in general).
	filtered := make(PreparedClaimsByUIDV2)
	for uid, claim := range cp.V2.PreparedClaims {
		if claim.CheckpointState == ClaimCheckpointStatePrepareCompleted {
			filtered[uid] = claim
		}
	}

	var expectedDeviceNames []DeviceName
	for _, cpclaim := range filtered {
		expectedDeviceNames = append(expectedDeviceNames, cpclaim.Status.Allocation.Devices.Results[0].Device)
	}

	if err := s.nvdevlib.obliterateStaleMIGDevices(expectedDeviceNames); err != nil {
		klog.Errorf("%s: obliterateStaleMIGDevices failed: %s", logpfx, err)
	}

	klog.Infof("%s: done", logpfx)
}

func (s *DeviceState) Unprepare(ctx context.Context, claimRef kubeletplugin.NamespacedObject) error {
	s.Lock()
	defer s.Unlock()
	klog.V(6).Infof("Unprepare() for claim '%s'", claimRef.String())

	checkpoint, err := s.getCheckpoint(ctx)
	if err != nil {
		return fmt.Errorf("unable to get checkpoint: %v", err)
	}

	claimUID := string(claimRef.UID)
	pc, exists := checkpoint.V2.PreparedClaims[claimUID]
	if !exists {
		// Not an error: if this claim UID is not in the checkpoint then this
		// device was never prepared or has already been unprepared (assume that
		// Prepare+Checkpoint are done transactionally). Note that
		// claimRef.String() contains namespace, name, UID.
		klog.V(2).Infof("Unprepare noop: claim not found in checkpoint data: %v", claimRef.String())
		return nil
	}

	switch pc.CheckpointState {
	case ClaimCheckpointStatePrepareStarted:
		// TODO: revert potential state mutations -- e.g. disable MIG mode,
		// destroy any potential MIG device -- (but: only by MIG UUID
		// identification -- deletion by just CI and GI id may attempt
		// destruction underneath another claim)
		//
		// Currently, we only store this:
		//
		// "checkpointState": "PrepareStarted", "status": {
		//   "allocation": {
		//     "devices": {
		//       "results": [
		//         {
		//           "request": "mig-1g",
		//           "driver": "gpu.nvidia.com",
		//           "pool": "gb-nvl-027-compute06",
		//           "device": "gpu-0-mig-1g24gb-0"
		//         }
		//       ]
		//     },
		//
		// `gpu-0-mig-1g24gb-0` does uniquely identify a profile/placement that
		// we could attempt to delete here -- but that doesn't seem safe.
		klog.Infof("unprepare noop: claim preparation started but not completed for claim '%s' (devices: %v)", claimRef.String(), pc.Status.Allocation.Devices.Results)
		return nil
	case ClaimCheckpointStatePrepareCompleted:
		if err := s.unprepareDevices(ctx, claimUID, pc.PreparedDevices); err != nil {
			return fmt.Errorf("unprepare devices failed for claim %s: %w", claimRef.String(), err)
		}
	default:
		return fmt.Errorf("unsupported ClaimCheckpointState: %v", pc.CheckpointState)
	}

	if featuregates.Enabled(featuregates.PassthroughSupport) {
		for _, device := range pc.PreparedDevices.GetDevices() {
			allocatableDevice, ok := s.allocatable[device.DeviceName]
			if !ok {
				klog.Warningf("allocatable not found for device: %v", device.DeviceName)
				continue
			}
			err := s.discoverSiblingAllocatables(allocatableDevice)
			if err != nil {
				return fmt.Errorf("error discovering sibling allocatables: %w", err)
			}
		}
	}
	if err := s.cdi.DeleteClaimSpecFile(claimUID); err != nil {
		return fmt.Errorf("unable to delete CDI spec file for claim %s: %w", claimRef.String(), err)
	}

	// Mutate checkpoint reflecting that all devices for this claim have been
	// unprepared, by virtue of removing its UID from the PreparedClaims map.
	err = s.deleteClaimFromCheckpoint(ctx, claimRef)
	if err != nil {
		return fmt.Errorf("error deleting claim from checkpoint: %w", err)
	}
	return nil
}

func (s *DeviceState) createCheckpoint(ctx context.Context, cp *Checkpoint) error {
	klog.V(6).Info("acquire cplock (create cp)")
	release, err := s.cplock.Acquire(ctx, flock.WithTimeout(10*time.Second))
	if err != nil {
		return fmt.Errorf("error acquiring cplock: %w", err)
	}
	defer release()
	klog.V(6).Info("acquired cplock")
	err = s.checkpointManager.CreateCheckpoint(DriverPluginCheckpointFileBasename, cp)
	klog.V(6).Info("create cp: done")
	return err
}

func (s *DeviceState) getCheckpoint(ctx context.Context) (*Checkpoint, error) {
	klog.V(6).Info("acquire cplock (get cp)")
	release, err := s.cplock.Acquire(ctx, flock.WithTimeout(10*time.Second))
	if err != nil {
		return nil, fmt.Errorf("error acquiring cplock: %w", err)
	}
	defer release()
	klog.V(6).Info("acquired cplock")

	checkpoint := &Checkpoint{}
	if err := s.checkpointManager.GetCheckpoint(DriverPluginCheckpointFileBasename, checkpoint); err != nil {
		return nil, err
	}
	klog.V(6).Info("cp read")
	return checkpoint.ToLatestVersion(), nil
}

// Read checkpoint from store, perform mutation, and write checkpoint back. Any
// mutation of the checkpoint must go through this function. The
// read-mutate-write sequence must be performed under a lock: we must be
// conceptually certain that multiple read-mutate-write actions never overlap.
func (s *DeviceState) updateCheckpoint(ctx context.Context, mutate func(*Checkpoint)) error {
	tucp0 := time.Now()
	klog.V(6).Info("acquire cplock (update cp)")
	release, err := s.cplock.Acquire(ctx, flock.WithTimeout(10*time.Second))
	if err != nil {
		return fmt.Errorf("error acquiring cplock: %w", err)
	}
	defer release()
	klog.V(6).Info("acquired cplock")

	// get checkpoint w/o acquiring lock (we have it already)
	checkpoint := &Checkpoint{}
	if err := s.checkpointManager.GetCheckpoint(DriverPluginCheckpointFileBasename, checkpoint); err != nil {
		return err
	}
	if err != nil {
		return fmt.Errorf("unable to get checkpoint: %w", err)
	}

	mutate(checkpoint)

	// create w/o lock acqu, we already have the lock
	err = s.checkpointManager.CreateCheckpoint(DriverPluginCheckpointFileBasename, checkpoint)
	if err != nil {
		return fmt.Errorf("unable to create checkpoint: %w", err)
	}
	klog.V(6).Info("cp updated")
	klog.V(6).Infof("t_checkpoint_update_total %.3f s", time.Since(tucp0).Seconds())
	return nil
}

func (s *DeviceState) deleteClaimFromCheckpoint(ctx context.Context, claimRef kubeletplugin.NamespacedObject) error {
	err := s.updateCheckpoint(ctx, func(cp *Checkpoint) {
		delete(cp.V2.PreparedClaims, string(claimRef.UID))
	})
	if err != nil {
		return fmt.Errorf("unable to update checkpoint: %w", err)
	}
	klog.V(6).Infof("Deleted claim from checkpoint: %s", claimRef.String())
	return nil
}

func (s *DeviceState) prepareDevices(ctx context.Context, claim *resourceapi.ResourceClaim) (PreparedDevices, error) {
	if claim.Status.Allocation == nil {
		return nil, fmt.Errorf("claim not yet allocated")
	}

	klog.V(6).Infof("Preparing devices for claim %s", ResourceClaimToString(claim))

	// Retrieve the full set of device configs for the driver.
	configs, err := GetOpaqueDeviceConfigs(
		configapi.StrictDecoder,
		DriverName,
		claim.Status.Allocation.Devices.Config,
	)
	if err != nil {
		return nil, fmt.Errorf("error getting opaque device configs: %v", err)
	}

	// Add the default GPU and MIG device Configs to the front of the config
	// list with the lowest precedence. This guarantees there will be at least
	// one of each config in the list with len(Requests) == 0 for the lookup below.
	configs = slices.Insert(configs, 0, &OpaqueDeviceConfig{
		Requests: []string{},
		Config:   configapi.DefaultGpuConfig(),
	})
	configs = slices.Insert(configs, 0, &OpaqueDeviceConfig{
		Requests: []string{},
		Config:   configapi.DefaultMigDeviceConfig(),
	})
	if featuregates.Enabled(featuregates.PassthroughSupport) {
		configs = slices.Insert(configs, 0, &OpaqueDeviceConfig{
			Requests: []string{},
			Config:   configapi.DefaultVfioDeviceConfig(),
		})
	}

	// Look through the configs and figure out which one will be applied to
	// each device allocation result based on their order of precedence and type.
	configResultsMap := make(map[runtime.Object][]*resourceapi.DeviceRequestAllocationResult)
	for _, result := range claim.Status.Allocation.Devices.Results {
		if result.Driver != DriverName {
			continue
		}
		device, exists := s.allocatable[result.Device]
		if !exists {
			return nil, fmt.Errorf("requested device is not allocatable: %v", result.Device)
		}
		// only proceed with config mapping if device is healthy.
		if featuregates.Enabled(featuregates.NVMLDeviceHealthCheck) {
			if !device.IsHealthy() {
				return nil, fmt.Errorf("requested device is not healthy: %v", result.Device)
			}
		}
		for _, c := range slices.Backward(configs) {
			if slices.Contains(c.Requests, result.Request) {
				if _, ok := c.Config.(*configapi.GpuConfig); ok && device.Type() != GpuDeviceType {
					return nil, fmt.Errorf("cannot apply GPU config to request: %v", result.Request)
				}
				if _, ok := c.Config.(*configapi.MigDeviceConfig); ok && device.Type() != MigDeviceType {
					return nil, fmt.Errorf("cannot apply MIG device config to request: %v", result.Request)
				}
				if _, ok := c.Config.(*configapi.VfioDeviceConfig); ok && device.Type() != VfioDeviceType {
					return nil, fmt.Errorf("cannot apply VFIO device config to request: %v", result.Request)
				}
				configResultsMap[c.Config] = append(configResultsMap[c.Config], &result)
				break
			}
			if len(c.Requests) == 0 {
				if _, ok := c.Config.(*configapi.GpuConfig); ok && device.Type() != GpuDeviceType {
					continue
				}
				if _, ok := c.Config.(*configapi.MigDeviceConfig); ok && device.Type() != MigDeviceType {
					continue
				}
				if _, ok := c.Config.(*configapi.VfioDeviceConfig); ok && device.Type() != VfioDeviceType {
					continue
				}
				configResultsMap[c.Config] = append(configResultsMap[c.Config], &result)
				break
			}
		}
	}

	// Normalize, validate, and apply all configs associated with devices that
	// need to be prepared. Track device group configs generated from applying the
	// config to the set of device allocation results.
	preparedDeviceGroupConfigState := make(map[runtime.Object]*DeviceConfigState)
	for c, results := range configResultsMap {
		// Cast the opaque config to a configapi.Interface type
		var config configapi.Interface
		switch castConfig := c.(type) {
		case *configapi.GpuConfig:
			config = castConfig
		case *configapi.MigDeviceConfig:
			config = castConfig
		case *configapi.VfioDeviceConfig:
			config = castConfig
		default:
			return nil, fmt.Errorf("runtime object is not a recognized configuration")
		}

		// Normalize the config to set any implied defaults.
		if err := config.Normalize(); err != nil {
			return nil, fmt.Errorf("error normalizing GPU config: %w", err)
		}

		// Validate the config to ensure its integrity.
		if err := config.Validate(); err != nil {
			return nil, fmt.Errorf("error validating GPU config: %w", err)
		}

		// Apply the config to the list of results associated with it.
		configState, err := s.applyConfig(ctx, config, claim, results)
		if err != nil {
			return nil, fmt.Errorf("error applying GPU config: %w", err)
		}

		// Capture the prepared device group config in the map.
		preparedDeviceGroupConfigState[c] = configState
	}

	// Walk through each config and its associated device allocation results
	// and construct the list of prepared devices to return.
	var preparedDevices PreparedDevices
	for c, results := range configResultsMap {
		preparedDeviceGroup := PreparedDeviceGroup{
			ConfigState: *preparedDeviceGroupConfigState[c],
		}

		for _, result := range results {
			cdiDevices := []string{}
			// The claim-based CDI spec is generated soon; Expect it to be the
			// complete CDI spec gererated freshly, with all devices specified
			// in that spec -- a CDI spec of kind `k8s.gpu.nvidia.com/claim`
			if d := s.cdi.GetClaimDeviceName(string(claim.UID), s.allocatable[result.Device], preparedDeviceGroupConfigState[c].containerEdits); d != "" {
				cdiDevices = append(cdiDevices, d)
			}

			device := &kubeletplugin.Device{
				Requests:     []string{result.Request},
				PoolName:     result.Pool,
				DeviceName:   result.Device,
				CDIDeviceIDs: cdiDevices,
			}

			var preparedDevice PreparedDevice
			switch s.allocatable[result.Device].Type() {
			case GpuDeviceType:
				preparedDevice.Gpu = &PreparedGpu{
					Info:   s.allocatable[result.Device].Gpu,
					Device: device,
				}
			case MigDeviceType:
				devinfo := s.allocatable[result.Device]
				// Maybe: persist anything to disk that may be useful for
				// cleaning up a partial prepare. Here, we have valuable
				// information: claim UID, mig device placement (also encoded by
				// the canonical MIG device name). Note that the
				// `PrepareStarted` checkpoint entry really only stores e.g.
				// `gpu-3-mig-1g24gb-5` as the only tangible piece of
				// information that we can use to delete a specific device.
				// However, deleting a MIG device requires parrent UUID, CI and
				// GI ID. Those are 'hard' (not impossible) to reconstruct based
				// on just that name.
				tcmig0 := time.Now()
				migdev, err := s.nvdevlib.createMigDevice(devinfo.Mig)
				klog.V(6).Infof("t_prep_create_mig_dev %.3f s (claim %s)", time.Since(tcmig0).Seconds(), ResourceClaimToString(claim))
				if err != nil {
					return nil, fmt.Errorf("error creating MIG device: %w", err)
				}

				preparedDevice.Mig = &PreparedMigDevice{
					// Abstract, allocatable device
					// encodes a specific mig/profile/placement combination.
					RequestedCanonicalName: devinfo.Mig.CanonicalName(),
					// Specifc, created device
					Created: migdev,
					// DRA device object
					Device: device,
				}
			case VfioDeviceType:
				preparedDevice.Vfio = &PreparedVfioDevice{
					Info:   s.allocatable[result.Device].Vfio,
					Device: device,
				}
			}

			klog.V(6).Infof("Prepared device for claim '%s': %s", ResourceClaimToString(claim), device.DeviceName)
			// TODO: here is maybe a decent opportunity to update the
			// checkpoint, reflecting the device preparation (still within
			// PrepareStarted state) -- there is potential for crashes and early
			// termination between here and final claim preparation; and we need
			// to retain the opportunity to revert changes, based on the
			// checkpoint ('unprepare previously partially prepared claims').
			// For example, a prepare devices failed: `error creating MIG
			// device: error creating GPU instance for 'gpu-0-mig-1g24gb-0':
			// Insufficient Resources,},},}`` may leave an enabled MIG mode
			// behind (not the most extreme example, can be resolved by other
			// means -- but just an example)
			preparedDeviceGroup.Devices = append(preparedDeviceGroup.Devices, preparedDevice)
		}

		preparedDevices = append(preparedDevices, &preparedDeviceGroup)
	}

	return preparedDevices, nil
}

func (s *DeviceState) unprepareDevices(ctx context.Context, claimUID string, devices PreparedDevices) error {
	klog.V(6).Infof("Unpreparing claim '%s', previously prepared devices from checkpoint: %v", claimUID, devices.GetDeviceNames())
	for _, group := range devices {
		// Unconfigure the vfio-pci devices.
		if featuregates.Enabled(featuregates.PassthroughSupport) {
			err := s.unprepareVfioDevices(ctx, group.Devices.VfioDevices())
			if err != nil {
				return err
			}
		}

		// Dynamically delete MIG devices, if applicable.
		// Do this before or after MPS/TimeSlicing primitive teardown?
		// EDIT: must be done after (TODO)
		for _, device := range group.Devices {
			switch device.Type() {
			case GpuDeviceType:
				klog.V(4).Infof("Unprepare: regular GPU: noop (GPU %s)", device.Gpu.Info.String())
			case MigDeviceType:
				mig := device.Mig.Created
				klog.V(4).Infof("Unprepare: tear down MIG device '%s' for claim '%s'", mig.UUID, claimUID)
				err := s.nvdevlib.deleteMigDevice(mig.ParentUUID, mig.GIID, mig.CIID)
				if err != nil {
					// Such errors are expected, but they also are somewhat
					// worrisome. This may for example be 'error destroying GPU
					// Instance: In use by another client' and resolve itself
					// soon. Log an explicit warning, at least.
					klog.Warningf("Error deleting MIG device %s: %s", device.Mig.Created.CanonicalName(), err)
					return fmt.Errorf("error deleting MIG device %s: %w", device.Mig.Created.CanonicalName(), err)
				}
			}
		}

		// Stop any MPS control daemons started for each group of prepared devices.
		if featuregates.Enabled(featuregates.MPSSupport) {
			mpsControlDaemon := s.mpsManager.NewMpsControlDaemon(claimUID, group)
			if err := mpsControlDaemon.Stop(ctx); err != nil {
				return fmt.Errorf("error stopping MPS control daemon: %w", err)
			}
		}

		// Go back to default time-slicing for all full GPUs.
		if featuregates.Enabled(featuregates.TimeSlicingSettings) {
			tsc := configapi.DefaultGpuConfig().Sharing.TimeSlicingConfig
			if err := s.tsManager.SetTimeSlice(group.Devices.Gpus(), tsc); err != nil {
				return fmt.Errorf("error setting timeslice for devices: %w", err)
			}
		}

	}
	return nil
}

func (s *DeviceState) getAllocatableVfioDevice(uuid string) (*AllocatableDevice, error) {
	for _, allocatable := range s.allocatable {
		if allocatable.Type() != VfioDeviceType {
			continue
		}
		if allocatable.Vfio.UUID == uuid {
			return allocatable, nil
		}
	}
	return nil, fmt.Errorf("allocatable device not found for vfio device: %v", uuid)
}

func (s *DeviceState) unprepareVfioDevices(ctx context.Context, devices PreparedDeviceList) error {
	for _, device := range devices {
		vfioAllocatable, err := s.getAllocatableVfioDevice(device.Vfio.Info.UUID)
		if err != nil {
			return fmt.Errorf("error getting allocatable device for vfio device: %w", err)
		}
		if err := s.vfioPciManager.Unconfigure(ctx, vfioAllocatable.Vfio); err != nil {
			return fmt.Errorf("error unconfiguring vfio device: %w", err)
		}
	}
	return nil
}

func (s *DeviceState) discoverSiblingAllocatables(device *AllocatableDevice) error {
	switch device.Type() {
	case GpuDeviceType:
		if !device.Gpu.vfioEnabled {
			return nil
		}
		vfio, err := s.nvdevlib.discoverVfioDevice(device.Gpu)
		if err != nil {
			return fmt.Errorf("error discovering vfio device: %w", err)
		}
		s.allocatable[vfio.CanonicalName()] = vfio
	case VfioDeviceType:
		gpu, migs, err := s.nvdevlib.discoverGPUByPCIBusID(device.Vfio.pcieBusID)
		if err != nil {
			return fmt.Errorf("error discovering gpu by pci bus id: %w", err)
		}
		s.allocatable[gpu.CanonicalName()] = gpu
		device.Vfio.parent = gpu.Gpu
		for _, mig := range migs {
			s.allocatable[mig.CanonicalName()] = mig
		}
	case MigDeviceType:
		// TODO: Implement once dynamic MIG is supported.
		return nil
	}
	return nil
}

func (s *DeviceState) applyConfig(ctx context.Context, config configapi.Interface, claim *resourceapi.ResourceClaim, results []*resourceapi.DeviceRequestAllocationResult) (*DeviceConfigState, error) {
	switch castConfig := config.(type) {
	case *configapi.GpuConfig:
		return s.applySharingConfig(ctx, castConfig.Sharing, claim, results)
	case *configapi.MigDeviceConfig:
		return s.applySharingConfig(ctx, castConfig.Sharing, claim, results)
	case *configapi.VfioDeviceConfig:
		return s.applyVfioDeviceConfig(ctx, castConfig, claim, results)
	default:
		return nil, fmt.Errorf("unknown config type: %T", castConfig)
	}
}

func (s *DeviceState) applySharingConfig(ctx context.Context, config configapi.Sharing, claim *resourceapi.ResourceClaim, results []*resourceapi.DeviceRequestAllocationResult) (*DeviceConfigState, error) {
	// Get the list of claim requests this config is being applied over.
	var requests []string
	for _, r := range results {
		requests = append(requests, r.Request)
	}

	// Get the list of allocatable devices this config is being applied over.
	allocatableDevices := make(AllocatableDevices)
	for _, r := range results {
		allocatableDevices[r.Device] = s.allocatable[r.Device]
	}

	// Declare a device group state object to populate.
	var configState DeviceConfigState

	// Apply time-slicing settings (if available and feature gate enabled).
	if featuregates.Enabled(featuregates.TimeSlicingSettings) && config.IsTimeSlicing() {
		// tsc, err := config.GetTimeSlicingConfig()
		// if err != nil {
		// 	return nil, fmt.Errorf("error getting timeslice config for requests '%v' in claim '%v': %w", requests, claim.UID, err)
		// }
		// if tsc != nil {
		// 	err = s.tsManager.SetTimeSlice(allocatableDevices, tsc)
		// 	if err != nil {
		// 		return nil, fmt.Errorf("error setting timeslice config for requests '%v' in claim '%v': %w", requests, claim.UID, err)
		// 	}
		// }
	}

	// Apply MPS settings (if available and feature gate enabled).
	if featuregates.Enabled(featuregates.MPSSupport) && config.IsMps() {
		// mpsc, err := config.GetMpsConfig()
		// if err != nil {
		// 	return nil, fmt.Errorf("error getting MPS configuration: %w", err)
		// }
		// mpsControlDaemon := s.mpsManager.NewMpsControlDaemon(string(claim.UID), allocatableDevices)
		// if err := mpsControlDaemon.Start(ctx, mpsc); err != nil {
		// 	return nil, fmt.Errorf("error starting MPS control daemon: %w", err)
		// }
		// if err := mpsControlDaemon.AssertReady(ctx); err != nil {
		// 	return nil, fmt.Errorf("MPS control daemon is not yet ready: %w", err)
		// }
		// configState.MpsControlDaemonID = mpsControlDaemon.GetID()
		// configState.containerEdits = mpsControlDaemon.GetCDIContainerEdits()
	}

	return &configState, nil
}

func (s *DeviceState) applyVfioDeviceConfig(ctx context.Context, config *configapi.VfioDeviceConfig, claim *resourceapi.ResourceClaim, results []*resourceapi.DeviceRequestAllocationResult) (*DeviceConfigState, error) {
	if !featuregates.Enabled(featuregates.PassthroughSupport) {
		return nil, nil
	}
	var configState DeviceConfigState

	// Configure the vfio-pci devices.
	for _, r := range results {
		info := s.allocatable[r.Device]
		err := s.vfioPciManager.Configure(ctx, info.Vfio)
		if err != nil {
			return nil, err
		}
	}

	return &configState, nil
}

// GetOpaqueDeviceConfigs returns an ordered list of the configs contained in possibleConfigs for this driver.
//
// Configs can either come from the resource claim itself or from the device
// class associated with the request. Configs coming directly from the resource
// claim take precedence over configs coming from the device class. Moreover,
// configs found later in the list of configs attached to its source take
// precedence over configs found earlier in the list for that source.
//
// All of the configs relevant to the driver from the list of possibleConfigs
// will be returned in order of precedence (from lowest to highest). If no
// configs are found, nil is returned.
func GetOpaqueDeviceConfigs(
	decoder runtime.Decoder,
	driverName string,
	possibleConfigs []resourceapi.DeviceAllocationConfiguration,
) ([]*OpaqueDeviceConfig, error) {
	// Collect all configs in order of reverse precedence.
	var classConfigs []resourceapi.DeviceAllocationConfiguration
	var claimConfigs []resourceapi.DeviceAllocationConfiguration
	var candidateConfigs []resourceapi.DeviceAllocationConfiguration
	for _, config := range possibleConfigs {
		switch config.Source {
		case resourceapi.AllocationConfigSourceClass:
			classConfigs = append(classConfigs, config)
		case resourceapi.AllocationConfigSourceClaim:
			claimConfigs = append(claimConfigs, config)
		default:
			return nil, fmt.Errorf("invalid config source: %v", config.Source)
		}
	}
	candidateConfigs = append(candidateConfigs, classConfigs...)
	candidateConfigs = append(candidateConfigs, claimConfigs...)

	// Decode all configs that are relevant for the driver.
	var resultConfigs []*OpaqueDeviceConfig
	for _, config := range candidateConfigs {
		// If this is nil, the driver doesn't support some future API extension
		// and needs to be updated.
		if config.Opaque == nil {
			return nil, fmt.Errorf("only opaque parameters are supported by this driver")
		}

		// Configs for different drivers may have been specified because a
		// single request can be satisfied by different drivers. This is not
		// an error -- drivers must skip over other driver's configs in order
		// to support this.
		if config.Opaque.Driver != driverName {
			continue
		}

		decodedConfig, err := runtime.Decode(decoder, config.Opaque.Parameters.Raw)
		if err != nil {
			return nil, fmt.Errorf("error decoding config parameters: %w", err)
		}

		resultConfig := &OpaqueDeviceConfig{
			Requests: config.Requests,
			Config:   decodedConfig,
		}

		resultConfigs = append(resultConfigs, resultConfig)
	}

	return resultConfigs, nil
}

func (s *DeviceState) UpdateDeviceHealthStatus(d *AllocatableDevice, hs HealthStatus) {
	s.Lock()
	defer s.Unlock()

	switch d.Type() {
	case GpuDeviceType:
		d.Gpu.Health = hs
	case MigDeviceType:
		d.Mig.Health = hs
	default:
		klog.V(6).Infof("Cannot update health status for unknown device type: %s", d.Type())
		return
	}
	klog.V(4).Infof("Updated device: %s health status to %s", d.UUID(), hs)
}

// TODO: Dynamic MIG is not yet supported with structured parameters.
// Refactor this to allow for the allocation of statically partitioned MIG
// devices.
//
// func (s *DeviceState) prepareMigDevices(claimUID string, allocated *nascrd.AllocatedMigDevices) (*PreparedMigDevices, error) {
// 	prepared := &PreparedMigDevices{}
//
// 	for _, device := range allocated.Devices {
// 		if _, exists := s.allocatable[device.ParentUUID]; !exists {
// 			return nil, fmt.Errorf("allocated GPU does not exist: %v", device.ParentUUID)
// 		}
//
// 		parent := s.allocatable[device.ParentUUID]
//
// 		if !parent.migEnabled {
// 			return nil, fmt.Errorf("cannot prepare a GPU with MIG mode disabled: %v", device.ParentUUID)
// 		}
//
// 		if _, exists := parent.migProfiles[device.Profile]; !exists {
// 			return nil, fmt.Errorf("MIG profile %v does not exist on GPU: %v", device.Profile, device.ParentUUID)
// 		}
//
// 		placement := nvml.GpuInstancePlacement{
// 			Start: uint32(device.Placement.Start),
// 			Size:  uint32(device.Placement.Size),
// 		}
//
// 		migInfo, err := s.nvdevlib.createMigDevice(parent.GpuInfo, parent.migProfiles[device.Profile].profile, &placement)
// 		if err != nil {
// 			return nil, fmt.Errorf("error creating MIG device: %w", err)
// 		}
//
// 		prepared.Devices = append(prepared.Devices, migInfo)
// 	}
//
// 	return prepared, nil
// }
//
// func (s *DeviceState) unprepareMigDevices(claimUID string, devices *PreparedDevices) error {
// 	for _, device := range devices.Mig.Devices {
// 		err := s.nvdevlib.deleteMigDevice(device)
// 		if err != nil {
// 			return fmt.Errorf("error deleting MIG device for %v: %w", device.uuid, err)
// 		}
// 	}
// 	return nil
//}
