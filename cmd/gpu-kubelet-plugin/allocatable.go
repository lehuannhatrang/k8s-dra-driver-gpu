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
	"fmt"
	"regexp"
	"slices"
	"strings"

	"github.com/Masterminds/semver"
	nvdev "github.com/NVIDIA/go-nvlib/pkg/nvlib/device"
	"github.com/NVIDIA/go-nvml/pkg/nvml"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/api/validate/constraints"
	"k8s.io/utils/ptr"
)

// Naming convention: this name is used for announcement (device name announced
// in a DRA ResourceSlice), and it's also played back to us upon a request,
// which is when we look it up in the AllocatableDevices map. Conceptually, this
// is hence the same as kubeletplugin.Device.DeviceName (documented with
// 'DeviceName identifies the device inside that pool')
type DeviceName = string

type AllocatableDevices map[DeviceName]*AllocatableDevice
type HealthStatus string
//type AllocatableDevices []AllocatableDevice

// AllocatableDevice represents an individual device that can be allocated.
// This can either be a full GPU or MIG device, but not both.
type AllocatableDevice struct {
	Gpu *GpuInfo
	Mig *MigInfo
	Vfio *VfioDeviceInfo
}

// MigInfo describes an abstract MIG device with precise specification of parent
// GPU, MIG profile and physcical placement on the parent GPU.
//
// Does not necessarily encode a _concrete_ (materialized / created /
// incarnated) device. In terms of abstract MIG device descriptions, there are
// many useful levels of precision / abstraction / refinement. For example:
//
// 1) specify MIG profile; do not specify parent, placement 2) specify MIG
// profile, parent; do not specify placement 3) specify MIG profile, parent,
// placement.
//
// This type is case (3); it leaves no degrees of freedom.
//
// TODO(JP): maybe rename to AbstractMigInfo or AbstractMigDevice or MigPP.
// TODO2(JP): clarify how CI id and GI id are not orthogonal to placement. Maybe
// add a method that translates between the two. I don't think we need to wait
// for creation to then know the CI ID and GI ID. TODO3(JP): clarify the
// relevance of the three-tuple of parent,ciid,giid for precisely describing a
// MIG device, and how that's a fundamental data structure. Maybe define its own
// type for it, and use it elsewhere.
type MigInfo struct {
	UUID          string `json:"uuid"`
	Parent        *GpuInfo
	Profile       nvdev.MigProfile
	GIProfileInfo nvml.GpuInstanceProfileInfo
	// TODO(JP): rename to Placement?
	MemorySlices nvml.GpuInstancePlacement
	Health        HealthStatus
}

func (i *MigInfo) CanonicalName() DeviceName {
	return migppCanonicalName(i.Parent.minor, i.Profile.String(), &i.MemorySlices)
}

// TODO(JP): add memory slices?
//
// Note(JP): for now, I feel like we may want to keep capacity agnostic to
// placement. That would imply not announcing specific memory slices as part of
// capacity (specific memory slices encode placement). That makes sense to me,
// but I may of course miss something here.
//
// I have noted that in an example spec in KEP 4815 we _do_ enumerate memory
// slices in a partition's capacity. Example:
//
//   - name: gpu-0-mig-2g.10gb-0-1
//     attributes:
//     ...
//     capacity:
//     ...
//     decoders:
//     value: "1"
//     encoders:
//     value: "0"
//     ...
//     memorySlice0:
//     value: "1"
//     memorySlice1:
//     value: "1"
//     multiprocessors:
//     value: "28"
//     ...
//
// 1) There, we only announce those slices with value 1 but we do _not_ announce
// memory slices not consumed (value: 0). That's inconsistent with other
// capacity dimensions which (in the example above) are enumerated despite
// having a value of zero (e.g. `encoders` above).
//
// 2) Semantically, to me, capacity I think can (and should?) be
// placement-agnostic. I am happy to be convinced otherwise.
//
// 3) If `capacity“ is in our case always encoding the _same_ information as
// `consumesCounters` then that's duplication and feels like an API design flaw
// or API usage flow. It feels like these parts of the resource slice device
// spec serve a different meaning, and hence allow for different information
// content. Again, I may miss something.
func (i MigInfo) PartCapacities() PartCapacityMap {
	p := i.GIProfileInfo
	return PartCapacityMap{
		"multiprocessors": intcap(p.MultiprocessorCount),
		"copyEngines":     intcap(p.CopyEngineCount),
		"decoders":        intcap(p.DecoderCount),
		"encoders":        intcap(p.EncoderCount),
		"jpegEngines":     intcap(p.JpegCount),
		"ofaEngines":      intcap(p.OfaCount),
		// In the k8s world, we love announcing unit-less memory :-). However,
		// it's important to know that `memory` here must be announced with the
		// unit "Bytes". The MIG profile's MemorySizeMB` property comes straight
		// from NVML and is documented in the public API docs with "Memory size
		// in MBytes". As it says "MB" and not MiB", one could assume that unit
		// to be 10^6 Bytes. However Update: in nvml.h this type's property is
		// documented with `memorySizeMB; //!< Device memory size (in MiB)`.
		// Hence, the unit is 2^20 Bytes (1024 * 1024 Bytes).
		"memory": intcap(int64(p.MemorySizeMB * 1024 * 1024)),
	}
}

func (i MigInfo) PartAttributes() map[resourceapi.QualifiedName]resourceapi.DeviceAttribute {
	return map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
		"type": {
			StringValue: ptr.To(MigDeviceType),
		},
		"parentUUID": {
			StringValue: &i.Parent.UUID,
		},
		// "parentIndex": {
		// 	IntValue: ptr.To(int64(i.Parent.index)), // TODO: really expose?
		// },
		"parentMinor": {
			IntValue: ptr.To(int64(i.Parent.minor)), // TODO: really expose?
		},
		"profile": {
			StringValue: ptr.To(i.Profile.String()),
		},
		"productName": {
			StringValue: &i.Parent.productName,
		},
		"brand": {
			StringValue: &i.Parent.brand,
		},
		"architecture": {
			StringValue: &i.Parent.architecture,
		},
		"cudaComputeCapability": {
			VersionValue: ptr.To(semver.MustParse(i.Parent.cudaComputeCapability).String()),
		},
		"driverVersion": {
			VersionValue: ptr.To(semver.MustParse(i.Parent.driverVersion).String()),
		},
		"cudaDriverVersion": {
			VersionValue: ptr.To(semver.MustParse(i.Parent.cudaDriverVersion).String()),
		},
		"pcieBusID": {
			StringValue: &i.Parent.pcieBusID,
		},
	}
}

func capacitiesToCounters(m PartCapacityMap) map[string]resourceapi.Counter {
	counters := make(map[string]resourceapi.Counter)
	for name, cap := range m {
		// Automatically derive counter name from capacity to ensure consistency.
		counters[toRFC1123Compliant(string(name))] = resourceapi.Counter{Value: cap.Value}
	}
	return counters
}

// This device is a partition of a physical GPU. The
//
// Construct `DeviceCounterConsumption`, describing which fractions precisely
// this partition consumes of the the full device.
//
// Each entry in capacity is modeled as a counter (consuming from the parent
// device). In addition, this MIG device, if allocated, consumes at least one
// specific memory slice. Each memory slice is modeled with its own counter
// (capacity: 1). Note that for example on a B200 GPU, the `3g.90gb` device
// consumes 4 out of 8 memory slices in total, but only 3 out of seven SMs. That
// is, with two `3g.90gb` devices allocated all memory slices are consumed, and
// one SM -- while unallocated -- cannot be used anymore. The parent is a full
// GPU device.
//
// When this device is allocated, it consumes from the parent's CounterSet. The
// parent's counter set is referred to by name. Use a naming convention:
// currently, a full GPU has precisely one counter set associated with it, and
// its name has the form 'gpu-%d-counter-set' where the placeholder is the GPU
// index (change to UUID)?
func (i MigInfo) PartConsumesCounters() []resourceapi.DeviceCounterConsumption {
	return []resourceapi.DeviceCounterConsumption{{
		CounterSet: i.Parent.GetSharedCounterSetName(),
		Counters:   addCountersForMemSlices(capacitiesToCounters(i.PartCapacities()), int(i.MemorySlices.Start), int(i.MemorySlices.Size)),
	}}
}

// Insert one counter for each memory slice consumed, as given by the `start`
// and `size` parameters (from a nvml.GpuInstancePlacement). Mutate the input
// map in place, and (also) return it.
func addCountersForMemSlices(counters map[string]resourceapi.Counter, start int, size int) map[string]resourceapi.Counter {
	for i := start; i < start+size; i++ {
		counters[memsliceCounterName(i)] = resourceapi.Counter{Value: *resource.NewQuantity(1, resource.BinarySI)}
	}
	return counters
}

func (i *MigInfo) PartGetDevice() resourceapi.Device {
	d := resourceapi.Device{
		Name:             i.CanonicalName(),
		Attributes:       i.PartAttributes(),
		Capacity:         i.PartCapacities(),
		ConsumesCounters: i.PartConsumesCounters(),
	}
	return d
}

func (d AllocatableDevice) Type() string {
	if d.Gpu != nil {
		return GpuDeviceType
	}
	if d.Mig != nil {
		return MigDeviceType
	}
	if d.Vfio != nil {
		return VfioDeviceType
	}
	return UnknownDeviceType
}

func (d *AllocatableDevice) CanonicalName() string {
	switch d.Type() {
	case GpuDeviceType:
		return d.Gpu.CanonicalName()
	case MigDeviceType:
		return d.Mig.CanonicalName()
	case VfioDeviceType:
		return d.Vfio.CanonicalName()
	}
	panic("unexpected type for AllocatableDevice")
}

func (d *AllocatableDevice) PartGetDevice() resourceapi.Device {
	switch d.Type() {
	case GpuDeviceType:
		return d.Gpu.PartGetDevice()
	case MigDeviceType:
		return d.Mig.PartGetDevice()
	case VfioDeviceType:
		return d.Vfio.GetDevice()
	}
	panic("unexpected type for AllocatableDevice")
}

// This concept will have to change for abstract allocatable devices that do not
// have a UUID before actualization.
func (d AllocatableDevice) UUID() string {
	if d.Gpu != nil {
		return d.Gpu.UUID
	}
	if d.Mig != nil {
		// Review: this can only be done after dynamic MIG device creation, once
		// this ID exists.
		return "foo"
	}
	if d.Vfio != nil {
		return d.Vfio.UUID
	}
	panic("unexpected type for AllocatableDevice")
}

type AllocatableDeviceList []*AllocatableDevice

func (d AllocatableDevices) getDevicesByGPUPCIBusID(pcieBusID string) AllocatableDeviceList {
	var devices AllocatableDeviceList
	for _, device := range d {
		switch device.Type() {
		case GpuDeviceType:
			if device.Gpu.pcieBusID == pcieBusID {
				devices = append(devices, device)
			}
		case MigDeviceType:
			if device.Mig.Parent.pcieBusID == pcieBusID {
				devices = append(devices, device)
			}
		case VfioDeviceType:
			if device.Vfio.pcieBusID == pcieBusID {
				devices = append(devices, device)
			}
		}
	}
	return devices
}

func (d AllocatableDevices) GetGPUByPCIeBusID(pcieBusID string) *AllocatableDevice {
	for _, device := range d {
		if device.Type() != GpuDeviceType {
			continue
		}
		if device.Gpu.pcieBusID == pcieBusID {
			return device
		}
	}
	return nil
}

func (d AllocatableDevices) GetGPUs() AllocatableDeviceList {
	var devices AllocatableDeviceList
	for _, device := range d {
		if device.Type() == GpuDeviceType {
			devices = append(devices, device)
		}
	}
	return devices
}

func (d AllocatableDevices) GetMigDevices() AllocatableDeviceList {
	var devices AllocatableDeviceList
	for _, device := range d {
		if device.Type() == MigDeviceType {
			devices = append(devices, device)
		}
	}
	return devices
}

func (d AllocatableDevices) GetVfioDevices() AllocatableDeviceList {
	var devices AllocatableDeviceList
	for _, device := range d {
		if device.Type() == VfioDeviceType {
			devices = append(devices, device)
		}
	}
	return devices
}

func (d AllocatableDevices) GpuUUIDs() []string {
	var uuids []string
	for _, device := range d {
		if device.Type() == GpuDeviceType {
			uuids = append(uuids, device.Gpu.UUID)
		}
	}
	slices.Sort(uuids)
	return uuids
}

func (d AllocatableDevices) PossibleMigDeviceNames() []string {
	var uuids []string
	for _, device := range d {
		if device.Type() == MigDeviceType {
			uuids = append(uuids, device.Mig.CanonicalName())
		}
	}
	slices.Sort(uuids)
	return uuids
}

func (d AllocatableDevices) VfioDeviceUUIDs() []string {
	var uuids []string
	for _, device := range d {
		if device.Type() == VfioDeviceType {
			uuids = append(uuids, device.Vfio.UUID)
		}
	}
	slices.Sort(uuids)
	return uuids
}

func (d AllocatableDevices) Names() []string {
	names := append(d.GpuUUIDs(), d.PossibleMigDeviceNames()...)
	slices.Sort(names)
	return names
}

// toRFC1123Compliant converts the input to a DNS name compliant with RFC 1123.
// Note that a device name in DRA must not contain dots either (this function
// does not always return a string that can be used as a device name).
func toRFC1123Compliant(name string) string {
	name = strings.ToLower(name)
	re := regexp.MustCompile(`[^a-z0-9-.]`)
	name = re.ReplaceAllString(name, "-")
	name = strings.Trim(name, "-")
	name = strings.TrimSuffix(name, ".")

	// Can this ever hurt? Should we error out?
	if len(name) > 253 {
		name = name[:253]
	}

	return name
}

func placementString(p *nvml.GpuInstancePlacement) string {
	sfx := fmt.Sprintf("%d", p.Start)
	if p.Size > 1 {
		sfx = fmt.Sprintf("%s-%d", sfx, p.Start+p.Size-1)
	}
	// first, I had `placement-` in there -- but that may be too bulky
	//return toRFC1123Compliant(fmt.Sprintf("%s", sfx))
	return toRFC1123Compliant(sfx)
}

// `profile string` must be what's returned by profile.String(), the
// classical/canonical profile string notation, with a dot. This must not crash
// when fed with data from a MigDeviceInfo object deserialized from JSON.
func migppCanonicalName(parentMinor int, profile string, p *nvml.GpuInstancePlacement) string {
	placementSuffix := placementString(p)
	// Remove the dot in e.g. `4g.95gb` -- device names must not contain dots,
	// and this is used in a device name.
	profname := strings.ReplaceAll(profile, ".", "")
	return toRFC1123Compliant(fmt.Sprintf("gpu-%d-mig-%s-%s", parentMinor, profname, placementSuffix))
}

// Return canonical name for memory slice (placement) `i` (a zero-based index).
// Note that this name must be used for memslice-N counters in a SharedCounters
// counter set, and for corresponding counters in a ConsumesCounters counter
// set. Counters (as opposed to capacities) are allowed to have hyphens in their
// name.
func memsliceCounterName(i int) string {
	return fmt.Sprintf("memory-slice-%d", i)
}

// Helper for creating an integer-based DeviceCapacity. Accept any integer type.
func intcap[T constraints.Integer](i T) resourceapi.DeviceCapacity {
	return resourceapi.DeviceCapacity{Value: *resource.NewQuantity(int64(i), resource.BinarySI)}
}

func (d AllocatableDevices) MigDeviceUUIDs() []string {
	var uuids []string
	for _, device := range d {
		if device.Type() == MigDeviceType {
			uuids = append(uuids, device.Mig.UUID)
		}
	}
	slices.Sort(uuids)
	return uuids
}

func (d AllocatableDevices) UUIDs() []string {
	uuids := append(d.GpuUUIDs(), d.MigDeviceUUIDs()...)
	uuids = append(uuids, d.VfioDeviceUUIDs()...)
	slices.Sort(uuids)
	return uuids
}

func (d AllocatableDevices) RemoveSiblingDevices(device *AllocatableDevice) {
	var pciBusID string
	switch device.Type() {
	case GpuDeviceType:
		pciBusID = device.Gpu.pcieBusID
	case VfioDeviceType:
		pciBusID = device.Vfio.pcieBusID
	case MigDeviceType:
		// TODO: Implement once dynamic MIG is supported.
		return
	}

	siblings := d.getDevicesByGPUPCIBusID(pciBusID)
	for _, sibling := range siblings {
		if sibling.Type() == device.Type() {
			continue
		}
		switch sibling.Type() {
		case GpuDeviceType:
			delete(d, sibling.Gpu.CanonicalName())
		case VfioDeviceType:
			delete(d, sibling.Vfio.CanonicalName())
		case MigDeviceType:
			// TODO: Implement once dynamic MIG is supported.
			continue
		}
	}
}

func (d *AllocatableDevice) IsHealthy() bool {
	switch d.Type() {
	case GpuDeviceType:
		return d.Gpu.Health == Healthy
	case MigDeviceType:
		return d.Mig.Health == Healthy
	}
	panic("unexpected type for AllocatableDevice")
}
