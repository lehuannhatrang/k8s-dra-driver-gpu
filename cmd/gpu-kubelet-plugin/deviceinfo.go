/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

	"github.com/Masterminds/semver"
	nvdev "github.com/NVIDIA/go-nvlib/pkg/nvlib/device"
	"github.com/NVIDIA/go-nvml/pkg/nvml"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/dynamic-resource-allocation/deviceattribute"
	"k8s.io/utils/ptr"
)


const (
	Healthy HealthStatus = "Healthy"
	// With NVMLDeviceHealthCheck, Unhealthy means that there are critcal xid errors on the device.
	Unhealthy HealthStatus = "Unhealthy"
)

// Represents a specific, full, physical GPU device.
type GpuInfo struct {
	UUID string `json:"uuid"`
	//index                 int
	minor                 int
	migEnabled            bool
	vfioEnabled           bool
	memoryBytes           uint64
	productName           string
	brand                 string
	architecture          string
	cudaComputeCapability string
	driverVersion         string
	cudaDriverVersion     string
	pcieBusID             string
	pcieRootAttr          *deviceattribute.DeviceAttribute
	migProfiles           []*MigProfileInfo
	Health                HealthStatus
	MigCapable            bool

	// Properties that can only be known after inspecting MIG profiles
	maxCapacities PartCapacityMap
	memSliceCount int
}

type PartCapacityMap map[resourceapi.QualifiedName]resourceapi.DeviceCapacity

// Represents a specific (concrete, incarnated, created) MIG device.
type MigDeviceInfo struct {
	UUID    string `json:"uuid"`
	Profile string `json:"profile"`

	// Selectively serialize details to checkpoint JSON file, needed mainly
	// for controlled deletion.
	ParentUUID  string `json:"parentUUID"`
	ParentMinor int    `json:"parentMinor"`
	CIID        int    `json:"ciId"`
	GIID        int    `json:"giId"`

	// Is the placement encoded already in CIID and GIID? In any case, for now,
	// store this in the JSON checkpoint because in CanonicalName() we rely on
	// this -- and this must work after JSON deserialization. TODO: introduce
	// cleaner data type carrying precisely (and just) what we want to store
	// about a created MIG device in the checkpoint JSON.
	Placement *MigDevicePlacement `json:"placement"`

	gIInfo        *nvml.GpuInstanceInfo
	cIInfo        *nvml.ComputeInstanceInfo
	parent        *GpuInfo
	giProfileInfo *nvml.GpuInstanceProfileInfo
	ciProfileInfo *nvml.ComputeInstanceProfileInfo
	pcieBusID     string
	pcieRootAttr  *deviceattribute.DeviceAttribute
	Health        HealthStatus
}

type VfioDeviceInfo struct {
	UUID                   string `json:"uuid"`
	deviceID               string
	vendorID               string
	index                  int
	parent                 *GpuInfo
	productName            string
	pcieBusID              string
	pcieRootAttr           *deviceattribute.DeviceAttribute
	numaNode               int
	iommuGroup             int
	addressableMemoryBytes uint64
}

type MigProfileInfo struct {
	profile    nvdev.MigProfile
	placements []*MigDevicePlacement
}

type MigDevicePlacement struct {
	nvml.GpuInstancePlacement
}

func (p MigProfileInfo) String() string {
	return p.profile.String()
}

func (d *GpuInfo) CanonicalName() DeviceName {
	return fmt.Sprintf("gpu-%d", d.minor)
}

// String() contains both the GPU index/minor for easy recognizability, but also
// the UUID for precision. It is intended for usage in log messages.
func (d *GpuInfo) String() string {
	return fmt.Sprintf("%s-%s", d.CanonicalName(), d.UUID)
}

func (d *MigDeviceInfo) CanonicalName() string {
	return migppCanonicalName(d.ParentMinor, d.Profile, &d.Placement.GpuInstancePlacement)
}

func (d *GpuInfo) PartDevAttributes() map[resourceapi.QualifiedName]resourceapi.DeviceAttribute {
	return map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
		"type": {
			StringValue: ptr.To(GpuDeviceType),
		},
		"uuid": {
			StringValue: &d.UUID,
		},
		"productName": {
			StringValue: &d.productName,
		},
		"brand": {
			StringValue: &d.brand,
		},
		"architecture": {
			StringValue: &d.architecture,
		},
		"cudaComputeCapability": {
			VersionValue: ptr.To(semver.MustParse(d.cudaComputeCapability).String()),
		},
		"driverVersion": {
			VersionValue: ptr.To(semver.MustParse(d.driverVersion).String()),
		},
		"cudaDriverVersion": {
			VersionValue: ptr.To(semver.MustParse(d.cudaDriverVersion).String()),
		},
		"pcieBusID": {
			StringValue: &d.pcieBusID,
		},
	}
}

// Full device capacity, based on looking at all MIG profiles.
func (d *GpuInfo) PartCapacities() PartCapacityMap {
	return d.maxCapacities
}

func (d *GpuInfo) GetSharedCounterSetName() string {
	return toRFC1123Compliant(fmt.Sprintf("gpu-%d-counter-set", d.minor))
}

// For now, define exactly one counter set per full GPU device. Individual
// partitions consume from that.
//
// In that counter set, define one counter per device capacity dimension, and
// add one counter (capacity 1) per memory slice.
func (d *GpuInfo) PartSharedCounterSets() []resourceapi.CounterSet {
	return []resourceapi.CounterSet{{
		Name:     d.GetSharedCounterSetName(),
		Counters: addCountersForMemSlices(capacitiesToCounters(d.maxCapacities), 0, d.memSliceCount),
	}}
}

func (d *GpuInfo) PartConsumesCounters() []resourceapi.DeviceCounterConsumption {
	// Let the full device consume everything. Goals: 1) when the full device is
	// allocated, all available counters drop to zero. 2) when the smallest
	// partition gets allocated, the full device cannot be allocated anymore.
	return []resourceapi.DeviceCounterConsumption{{
		CounterSet: d.GetSharedCounterSetName(),
		Counters:   addCountersForMemSlices(capacitiesToCounters(d.maxCapacities), 0, d.memSliceCount),
	}}
}

// For the new partitionable devices API, return the 'full' device announcement.
func (d *GpuInfo) PartGetDevice() resourceapi.Device {
	dev := resourceapi.Device{
		Name:             d.CanonicalName(),
		Attributes:       d.PartDevAttributes(),
		Capacity:         d.PartCapacities(),
		ConsumesCounters: d.PartConsumesCounters(),
	}

	// Not available in all environments, enrich advertised device only
	// conditionally.
	if d.pcieRootAttr != nil {
		dev.Attributes[d.pcieRootAttr.Name] = d.pcieRootAttr.Value
	}

	return dev
}

// Add detail that is only known after inspecting the individual MIG profiles.
func (d *GpuInfo) AddDetailAfterWalkingMigProfiles(maxcap PartCapacityMap, memSliceCount int) {
	d.maxCapacities = maxcap
	d.memSliceCount = memSliceCount
}

func (d *VfioDeviceInfo) CanonicalName() string {
	return fmt.Sprintf("gpu-vfio-%d", d.index)
}

func (d *GpuInfo) GetDevice() resourceapi.Device {
	device := resourceapi.Device{
		Name:       d.CanonicalName(),
		Attributes: d.PartDevAttributes(),
		Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
			"memory": {
				Value: *resource.NewQuantity(int64(d.memoryBytes), resource.BinarySI),
			},
		},
	}
	if d.pcieRootAttr != nil {
		device.Attributes[d.pcieRootAttr.Name] = d.pcieRootAttr.Value
	}
	return device
}

func (d *MigDeviceInfo) GetDevice() resourceapi.Device {
	device := resourceapi.Device{
		Name: d.CanonicalName(),
		Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
			"type": {
				StringValue: ptr.To(MigDeviceType),
			},
			"uuid": {
				StringValue: &d.UUID,
			},
			"parentUUID": {
				StringValue: &d.parent.UUID,
			},
			"profile": {
				StringValue: &d.Profile,
			},
			"productName": {
				StringValue: &d.parent.productName,
			},
			"brand": {
				StringValue: &d.parent.brand,
			},
			"architecture": {
				StringValue: &d.parent.architecture,
			},
			"cudaComputeCapability": {
				VersionValue: ptr.To(semver.MustParse(d.parent.cudaComputeCapability).String()),
			},
			"driverVersion": {
				VersionValue: ptr.To(semver.MustParse(d.parent.driverVersion).String()),
			},
			"cudaDriverVersion": {
				VersionValue: ptr.To(semver.MustParse(d.parent.cudaDriverVersion).String()),
			},
			"pcieBusID": {
				StringValue: &d.pcieBusID,
			},
		},
		Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
			"multiprocessors": {
				Value: *resource.NewQuantity(int64(d.giProfileInfo.MultiprocessorCount), resource.BinarySI),
			},
			"copyEngines": {Value: *resource.NewQuantity(int64(d.giProfileInfo.CopyEngineCount), resource.BinarySI)},
			"decoders":    {Value: *resource.NewQuantity(int64(d.giProfileInfo.DecoderCount), resource.BinarySI)},
			"encoders":    {Value: *resource.NewQuantity(int64(d.giProfileInfo.EncoderCount), resource.BinarySI)},
			"jpegEngines": {Value: *resource.NewQuantity(int64(d.giProfileInfo.JpegCount), resource.BinarySI)},
			"ofaEngines":  {Value: *resource.NewQuantity(int64(d.giProfileInfo.OfaCount), resource.BinarySI)},
			// Should we expose this as memoryBytes instead? Seemingly, in the
			// k8s landscape, that ship has long saileed: container limits for
			// example also use `memory`.
			"memory": {Value: *resource.NewQuantity(int64(d.giProfileInfo.MemorySizeMB*1024*1024), resource.BinarySI)},
		},
	}

	// Note(JP): noted elsewhere; what's the purpose of announcing memory slices
	// as capacity? Are users interested? That effectively shows 'placement' to
	// users. Does it also allow users to request placement? Do we want to allow
	// users to request specific placement?
	for i := d.Placement.Start; i < d.Placement.Start+d.Placement.Size; i++ {
		// TODO: review memorySlice (legacy) vs memory-slice -- I believe I
		// prefer memory-slice because that works for counters. Do we even need
		// to announce the slices as capacity?
		capacity := resourceapi.QualifiedName(fmt.Sprintf("memorySlice%d", i))
		device.Capacity[capacity] = resourceapi.DeviceCapacity{
			Value: *resource.NewQuantity(1, resource.BinarySI),
		}
	}
	if d.pcieRootAttr != nil {
		device.Attributes[d.pcieRootAttr.Name] = d.pcieRootAttr.Value
	}
	return device
}

func (d *VfioDeviceInfo) GetDevice() resourceapi.Device {
	device := resourceapi.Device{
		Name: d.CanonicalName(),
		Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
			"type": {
				StringValue: ptr.To(VfioDeviceType),
			},
			"uuid": {
				StringValue: &d.UUID,
			},
			"deviceID": {
				StringValue: &d.deviceID,
			},
			"vendorID": {
				StringValue: &d.vendorID,
			},
			"numa": {
				IntValue: ptr.To(int64(d.numaNode)),
			},
			"pcieBusID": {
				StringValue: &d.pcieBusID,
			},
			"productName": {
				StringValue: &d.productName,
			},
		},
		Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
			"addressableMemory": {
				Value: *resource.NewQuantity(int64(d.addressableMemoryBytes), resource.BinarySI),
			},
		},
	}
	if d.pcieRootAttr != nil {
		device.Attributes[d.pcieRootAttr.Name] = d.pcieRootAttr.Value
	}
	return device
}
