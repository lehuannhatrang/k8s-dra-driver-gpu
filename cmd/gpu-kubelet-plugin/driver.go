/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
	"maps"
	"path/filepath"
	"slices"
	"time"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/runtime"
	coreclientset "k8s.io/client-go/kubernetes"
	"k8s.io/dynamic-resource-allocation/kubeletplugin"
	"k8s.io/dynamic-resource-allocation/resourceslice"
	"k8s.io/klog/v2"

	"github.com/NVIDIA/k8s-dra-driver-gpu/pkg/flock"
)

// DriverPrepUprepFlockPath is the path to a lock file used to make sure
// that calls to nodePrepareResource() / nodeUnprepareResource() never
// interleave, node-globally.
const DriverPrepUprepFlockFileName = "pu.lock"

type driver struct {
	client       coreclientset.Interface
	pluginhelper *kubeletplugin.Helper
	state        *DeviceState
	pulock       *flock.Flock
	healthcheck  *healthcheck
}

func NewDriver(ctx context.Context, config *Config) (*driver, error) {
	state, err := NewDeviceState(ctx, config)
	if err != nil {
		return nil, err
	}

	// Could be done in NewDeviceState, but I want to make sure that the
	// checkpoint logic is intact -- that's most obvious here.
	state.DestroyUnknownMIGDevices(ctx)

	puLockPath := filepath.Join(config.DriverPluginPath(), DriverPrepUprepFlockFileName)

	driver := &driver{
		client: config.clientsets.Core,
		state:  state,
		pulock: flock.NewFlock(puLockPath),
	}

	helper, err := kubeletplugin.Start(
		ctx,
		driver,
		kubeletplugin.KubeClient(driver.client),
		kubeletplugin.NodeName(config.flags.nodeName),
		kubeletplugin.DriverName(DriverName),
		kubeletplugin.Serialize(false),
		kubeletplugin.RegistrarDirectoryPath(config.flags.kubeletRegistrarDirectoryPath),
		kubeletplugin.PluginDataDirectoryPath(config.DriverPluginPath()),
	)
	if err != nil {
		return nil, err
	}
	driver.pluginhelper = helper

	resources := driver.GenerateDriverResources(config.flags.nodeName)

	healthcheck, err := startHealthcheck(ctx, config, helper)
	if err != nil {
		return nil, fmt.Errorf("start healthcheck: %w", err)
	}
	driver.healthcheck = healthcheck

	// Note(JP): from KEP 4815: "we will add client-side validation in the
	// ResourceSlice controller helper, so that any errors in the ResourceSlices
	// will be caught before they even are applied to the APIServer" -- the
	// helper below is being referred to.
	//
	// This will be a TODO soon:
	// https://github.com/kubernetes/kubernetes/commit/a171795e313ee9f407fef4897c1a1e2052120991
	if err := driver.pluginhelper.PublishResources(ctx, resources); err != nil {
		return nil, err
	}

	klog.V(4).Infof("Current kubelet plugin registration status: %s", helper.RegistrationStatus())

	return driver, nil
}

// GenerateDriverResources() returns the set of DRA ResourceSlices announced by
// this DRA driver to the system.
func (d *driver) GenerateDriverResources(nodeName string) resourceslice.DriverResources {
	// Note(JP): I first considered model 1, and then implemented model model 2.
	//
	// Model 1) Create G+1 resource slices, given G physical GPUs. 1) one slice
	// with shared counters, for each full GPU 2) for each full GPU, a slice
	// with the full-GPU-device and all potential MIG profile/placement devices.
	// That model is inspired by KEP 4815 which talks about "require that
	// ResourceSlice objects can only contain either SharedCounters or Devices".
	// In practice, that does not seem to be a requirement (yet?); in fact, it
	// doesn't seem to work to consume from a shared counter _not_ specified in
	// the same ResourceSlice. Which brings me to model 2.
	//
	// Model 2) Create G resource slices, given G physical GPUs. Each slice
	// describes the abstract full device capacity by definining _one_ counter
	// set as part of `SharedCounters` (there could be 8 counter sets in total,
	// but try to get away using one). Then, in the same resource slice define
	// all devices allocatable for that physical GPU. That is, M possible MIG
	// devices, and 1 device representing the full GPU. Hence:
	//
	// - G resource slices
	// - Each resource slice:
	//       - Defines `SharedCounters` with 1 counter set
	//       - Defines M+1 devices
	//
	//
	//
	// Specific quote from KEP 4815 does does not make sense to be right now:
	// "The decision to require that ResourceSlice objects can only contain
	// either SharedCounters or Devices was made to prevent having to enforce
	// overly strict validation to make sure that ResourceSlice objects can't
	// exceed the etcd limit."

	var gpuslices []resourceslice.Slice

	// Iterate through map in predictable order, so that the slices get
	// published in predictable order.
	for _, minor := range slices.Sorted(maps.Keys(d.state.perGPUAllocatable)) {
		allocatable := d.state.perGPUAllocatable[minor]
		var slice resourceslice.Slice
		countersets := []resourceapi.CounterSet{}

		// Stable sort order by devicename -- makes the order of devices
		// presented in a resource slice predictable. Good for debuggability /
		// readability, and leads to a minimal slice diff during kubelet plugin
		// restart (the slice diff is logged).
		for _, devname := range slices.Sorted(maps.Keys(allocatable)) {
			device := allocatable[devname]
			klog.V(4).Infof("About to announce device %s", devname)

			// Full GPU: take note of countersets, indicating absolute capacity.
			// For now this is expected to be one counter set.
			if device.Gpu != nil {
				countersets = append(countersets, device.Gpu.PartSharedCounterSets()...)
			}

			// Add all allocatable devices for this physical GPU to this slice.
			// This includes not-yet-manifested MIG devices, and the physical
			// GPU itself.
			slice.Devices = append(slice.Devices, device.PartGetDevice())
		}
		slice.SharedCounters = countersets
		gpuslices = append(gpuslices, slice)
	}

	return resourceslice.DriverResources{
		Pools: map[string]resourceslice.Pool{
			nodeName: {Slices: gpuslices},
		},
	}
}

func (d *driver) Shutdown() error {
	if d == nil {
		return nil
	}

	if d.healthcheck != nil {
		d.healthcheck.Stop()
	}

	d.state.nvdevlib.alwaysShutdown()

	d.pluginhelper.Stop()
	return nil
}

func (d *driver) PrepareResourceClaims(ctx context.Context, claims []*resourceapi.ResourceClaim) (map[types.UID]kubeletplugin.PrepareResult, error) {

	if len(claims) == 0 {
		// That's probably the health check, log that on higher verbosity level
		klog.V(7).Infof("PrepareResourceClaims called with %d claim(s)", len(claims))
	} else {
		// Log canonical string representation for each claim injected here --
		// we've noticed that this can greatly facilitate debugging.
		klog.V(6).Infof("Prepare called for: %v", ClaimsToStrings(claims))
	}

	results := make(map[types.UID]kubeletplugin.PrepareResult)

	for _, claim := range claims {
		results[claim.UID] = d.nodePrepareResource(ctx, claim)
	}

	return results, nil
}

func (d *driver) UnprepareResourceClaims(ctx context.Context, claimRefs []kubeletplugin.NamespacedObject) (map[types.UID]error, error) {
	klog.V(6).Infof("Unprepare called for: %v", ClaimRefsToStrings(claimRefs))
	results := make(map[types.UID]error)
	for _, claimRef := range claimRefs {
		results[claimRef.UID] = d.nodeUnprepareResource(ctx, claimRef)
	}

	return results, nil
}

func (d *driver) HandleError(ctx context.Context, err error, msg string) {
	// For now we just follow the advice documented in the DRAPlugin API docs.
	// See: https://pkg.go.dev/k8s.io/apimachinery/pkg/util/runtime#HandleErrorWithContext
	runtime.HandleErrorWithContext(ctx, err, msg)
}

func (d *driver) nodePrepareResource(ctx context.Context, claim *resourceapi.ResourceClaim) kubeletplugin.PrepareResult {
	cs := ResourceClaimToString(claim)
	// queue things a little longer than 10 seconds.
	t0 := time.Now()
	release, err := d.pulock.Acquire(ctx, flock.WithTimeout(300*time.Second))
	if err != nil {
		return kubeletplugin.PrepareResult{
			Err: fmt.Errorf("error acquiring prep/unprep lock: %w", err),
		}
	}
	defer release()
	klog.V(6).Infof("t_prep_lock_acq %.3f s", time.Since(t0).Seconds())

	tprep0 := time.Now()
	devs, err := d.state.Prepare(ctx, claim)
	klog.V(6).Infof("t_prep %.3f s (claim %s)", time.Since(tprep0).Seconds(), cs)
	klog.V(6).Infof("t_prep_total %.3f s (claim %s)", time.Since(t0).Seconds(), cs)

	if err != nil {
		return kubeletplugin.PrepareResult{
			Err: fmt.Errorf("error preparing devices for claim %s: %w", cs, err),
		}
	}

	klog.Infof("Returning newly prepared devices for claim '%s': %v", cs, devs)
	return kubeletplugin.PrepareResult{Devices: devs}
}

func (d *driver) nodeUnprepareResource(ctx context.Context, claimRef kubeletplugin.NamespacedObject) error {
	cs := claimRef.String()
	t0 := time.Now()
	release, err := d.pulock.Acquire(ctx, flock.WithTimeout(300*time.Second))
	if err != nil {
		return fmt.Errorf("error acquiring prep/unprep lock: %w", err)
	}
	defer release()
	klog.V(6).Infof("t_unprep_lock_acq %.3f s", time.Since(t0).Seconds())

	tunprep0 := time.Now()
	err = d.state.Unprepare(ctx, claimRef)
	klog.V(6).Infof("t_unprep %.3f s (claim %s)", time.Since(tunprep0).Seconds(), cs)
	klog.V(6).Infof("t_unprep_total %.3f s (claim %s)", time.Since(t0).Seconds(), cs)

	if err != nil {
		return fmt.Errorf("error unpreparing devices for claim %v: %w", claimRef.String(), err)
	}

	return nil
}

// TODO: implement loop to remove CDI files from the CDI path for claimUIDs
//       that have been removed from the AllocatedClaims map.
// func (d *driver) cleanupCDIFiles(wg *sync.WaitGroup) chan error {
// 	errors := make(chan error)
// 	return errors
// }
//
// TODO: implement loop to remove mpsControlDaemon folders from the mps
//       path for claimUIDs that have been removed from the AllocatedClaims map.
// func (d *driver) cleanupMpsControlDaemonArtifacts(wg *sync.WaitGroup) chan error {
// 	errors := make(chan error)
// 	return errors
// }
