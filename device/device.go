// Package device models a whole PC as measured resources: processing
// units (CPU performance/efficiency core clusters, GPUs of any vendor),
// memory domains (per-GPU VRAM, system RAM), storage devices and the
// links between memory domains.
//
// Every performance number in this package is meant to be a MEASUREMENT
// (microbenchmark result), never a spec-sheet guess: estimates may be
// used to order candidates, but the tuner (package tuner) replaces them
// with measurements on the real machine. This is the central lesson of
// the Alloy heterogeneous-GPU investigation (see docs/ALGORITHM.md).
package device

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// UnitKind classifies a processing unit.
type UnitKind string

const (
	KindCPUPerf UnitKind = "cpu-performance" // performance cores (cluster)
	KindCPUEff  UnitKind = "cpu-efficiency"  // efficiency cores (cluster)
	KindCPU     UnitKind = "cpu"             // undifferentiated CPU
	KindGPU     UnitKind = "gpu"
)

// Unit is one processing unit (a core cluster counts as one unit; the
// planner assigns work to clusters, not individual cores).
type Unit struct {
	ID   string   `json:"id"`
	Name string   `json:"name"`
	Kind UnitKind `json:"kind"`

	// ComputeRate is the measured sustained-peak throughput in abstract
	// work units per second (workload.Compute is expressed in the same
	// units; FLOPs are the natural choice). Measured, per phase class if
	// the machine profile distinguishes phases — see Rates.
	ComputeRate float64 `json:"compute_rate"`
	// SustainedRate is the compute rate once the unit is thermally
	// soaked (<= ComputeRate). 0 means "same as ComputeRate".
	SustainedRate float64 `json:"sustained_rate,omitempty"`
	// ThermalTauSec is the thermal time constant: a workload shorter
	// than this runs at peak rate; a much longer one runs at the
	// sustained rate. The cost model blends between the two.
	ThermalTauSec float64 `json:"thermal_tau_sec,omitempty"`

	// MemoryID is the unit's local memory domain (VRAM for a GPU, RAM
	// for a CPU). Data resident elsewhere is streamed over a link.
	MemoryID string `json:"memory_id"`

	PowerActiveW float64 `json:"power_active_w,omitempty"`
	PowerIdleW   float64 `json:"power_idle_w,omitempty"`
}

// EffectiveRate returns the compute rate to charge for a run whose
// total duration is horizonSec: peak for short bursts, blending toward
// the sustained (thermally soaked) rate as the horizon grows past the
// thermal time constant.
func (u Unit) EffectiveRate(horizonSec float64) float64 {
	peak := u.ComputeRate
	sust := u.SustainedRate
	if sust <= 0 || sust >= peak {
		return peak
	}
	if u.ThermalTauSec <= 0 || horizonSec <= u.ThermalTauSec {
		return peak
	}
	// Fraction of the run spent soaked grows toward 1 as horizon >> tau.
	soak := 1 - u.ThermalTauSec/horizonSec
	return peak - (peak-sust)*soak
}

// Memory is one memory domain: a capacity plus a shared bandwidth.
// Bandwidth is shared by every unit and task using the domain — that
// sharing is the machine's main interference channel.
type Memory struct {
	ID                string  `json:"id"`
	Name              string  `json:"name"`
	CapacityBytes     int64   `json:"capacity_bytes"`
	BandwidthBytesSec float64 `json:"bandwidth_bytes_sec"`
	LatencySec        float64 `json:"latency_sec,omitempty"`
}

// Storage is one storage device (NVMe, SATA SSD, HDD).
type Storage struct {
	ID            string  `json:"id"`
	Name          string  `json:"name"`
	ReadBytesSec  float64 `json:"read_bytes_sec"`
	WriteBytesSec float64 `json:"write_bytes_sec"`
	CapacityBytes int64   `json:"capacity_bytes,omitempty"`
}

// Link is a data path between two memory domains (PCIe, Infinity
// Fabric, a memory-bus hop). Transfers between domains with no link
// fall back to host staging through RAM at DefaultLinkRate.
type Link struct {
	FromMemory        string  `json:"from_memory"`
	ToMemory          string  `json:"to_memory"`
	BandwidthBytesSec float64 `json:"bandwidth_bytes_sec"`
	LatencySec        float64 `json:"latency_sec,omitempty"`
}

// DefaultLinkRate is the fallback transfer bandwidth (bytes/sec) for
// domain pairs without an explicit link: host-staged PCIe-class copies.
const DefaultLinkRate = 8e9

// Machine is a whole PC.
type Machine struct {
	Name     string    `json:"name"`
	Units    []Unit    `json:"units"`
	Memories []Memory  `json:"memories"`
	Storages []Storage `json:"storages,omitempty"`
	Links    []Link    `json:"links,omitempty"`
	// PowerCapW is the whole-machine sustained power budget (PSU /
	// chassis / user-set cap). 0 means unconstrained.
	PowerCapW float64 `json:"power_cap_w,omitempty"`

	units    map[string]*Unit
	memories map[string]*Memory
	links    map[[2]string]Link
}

func (m *Machine) index() {
	// The maps cache pointers into the slices. A copied Machine whose
	// slices were then extended (the natural way to model "same PC,
	// one more device") must not keep serving the stale index, so
	// rebuild whenever the lengths disagree.
	if m.units != nil && len(m.units) == len(m.Units) && len(m.memories) == len(m.Memories) {
		return
	}
	m.units = map[string]*Unit{}
	for i := range m.Units {
		m.units[m.Units[i].ID] = &m.Units[i]
	}
	m.memories = map[string]*Memory{}
	for i := range m.Memories {
		m.memories[m.Memories[i].ID] = &m.Memories[i]
	}
	m.links = map[[2]string]Link{}
	for _, l := range m.Links {
		m.links[[2]string{l.FromMemory, l.ToMemory}] = l
		m.links[[2]string{l.ToMemory, l.FromMemory}] = l
	}
}

// Unit returns the unit with the given ID.
func (m *Machine) Unit(id string) (Unit, bool) {
	m.index()
	u, ok := m.units[id]
	if !ok {
		return Unit{}, false
	}
	return *u, true
}

// Memory returns the memory domain with the given ID.
func (m *Machine) Memory(id string) (Memory, bool) {
	m.index()
	mem, ok := m.memories[id]
	if !ok {
		return Memory{}, false
	}
	return *mem, true
}

// Transfer returns (bandwidth, latency) for moving data between two
// memory domains. Same domain: effectively free (bandwidth = +Inf is
// reported as 0 with ok flag semantics handled by callers via
// SameDomain).
func (m *Machine) Transfer(fromMem, toMem string) (bw float64, latency float64) {
	if fromMem == toMem {
		return math.Inf(1), 0
	}
	m.index()
	if l, ok := m.links[[2]string{fromMem, toMem}]; ok {
		return l.BandwidthBytesSec, l.LatencySec
	}
	return DefaultLinkRate, 0
}

// FastestStorageRead is the best storage read bandwidth (bytes/sec),
// 0 when the machine has no storage model.
func (m *Machine) FastestStorageRead() float64 {
	best := 0.0
	for _, s := range m.Storages {
		if s.ReadBytesSec > best {
			best = s.ReadBytesSec
		}
	}
	return best
}

// ActivePower sums the active power draw of the given unit IDs.
func (m *Machine) ActivePower(unitIDs []string) float64 {
	m.index()
	sum := 0.0
	seen := map[string]bool{}
	for _, id := range unitIDs {
		if seen[id] {
			continue
		}
		seen[id] = true
		if u, ok := m.units[id]; ok {
			sum += u.PowerActiveW
		}
	}
	return sum
}

// Signature identifies a machine configuration for tuner records: the
// sorted unit IDs with capacities and (rounded) measured rates, plus
// memory capacities and the power cap. A tuned winner measured on one
// signature must never be applied to another — when hardware or its
// measured profile changes, planning falls back to estimates.
func (m *Machine) Signature() string {
	m.index()
	var parts []string
	ids := make([]string, 0, len(m.units))
	for id := range m.units {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		u := m.units[id]
		memCap := int64(0)
		if mem, ok := m.memories[u.MemoryID]; ok {
			memCap = mem.CapacityBytes
		}
		parts = append(parts, fmt.Sprintf("%s:%s:cap=%d:rate=%.0f:sust=%.0f",
			u.ID, u.Kind, memCap, u.ComputeRate, u.EffectiveRate(math.Inf(1))))
	}
	return strings.Join(parts, "|") + fmt.Sprintf(":capW=%.0f", m.PowerCapW)
}

// Microbench is one measured device profile: the import format for
// benchmark results (e.g. from llama-bench-style sweeps, STREAM,
// fio, or a custom harness). Fields left at zero keep the unit's
// existing values when applied with ApplyMicrobench.
type Microbench struct {
	UnitID          string  `json:"unit_id"`
	ComputeRate     float64 `json:"compute_rate,omitempty"`
	SustainedRate   float64 `json:"sustained_rate,omitempty"`
	MemoryBandwidth float64 `json:"memory_bandwidth_bytes_sec,omitempty"` // of the unit's local domain
	StorageRead     float64 `json:"storage_read_bytes_sec,omitempty"`
	MeasuredAt      string  `json:"measured_at,omitempty"` // RFC3339, informational
	Source          string  `json:"source,omitempty"`      // harness that produced it
}

// ApplyMicrobench folds measured values into the machine, returning a
// copy (the receiver is not mutated). Unknown unit IDs are ignored.
func (m Machine) ApplyMicrobench(results []Microbench) Machine {
	out := m
	out.units, out.memories, out.links = nil, nil, nil
	out.Units = append([]Unit(nil), m.Units...)
	out.Memories = append([]Memory(nil), m.Memories...)
	out.Storages = append([]Storage(nil), m.Storages...)
	byUnit := map[string]*Unit{}
	for i := range out.Units {
		byUnit[out.Units[i].ID] = &out.Units[i]
	}
	memByID := map[string]*Memory{}
	for i := range out.Memories {
		memByID[out.Memories[i].ID] = &out.Memories[i]
	}
	for _, r := range results {
		u, ok := byUnit[r.UnitID]
		if !ok {
			continue
		}
		if r.ComputeRate > 0 {
			u.ComputeRate = r.ComputeRate
		}
		if r.SustainedRate > 0 {
			u.SustainedRate = r.SustainedRate
		}
		if r.MemoryBandwidth > 0 {
			if mem, ok := memByID[u.MemoryID]; ok {
				mem.BandwidthBytesSec = r.MemoryBandwidth
			}
		}
	}
	return out
}

// Validate reports structural problems (dangling references, negative
// values). Planning on an invalid machine is meaningless, so callers
// should surface these rather than plan around them.
func (m *Machine) Validate() []string {
	m.index()
	var problems []string
	seenUnits := map[string]bool{}
	for _, u := range m.Units {
		if seenUnits[u.ID] {
			problems = append(problems, fmt.Sprintf("duplicate unit ID %q", u.ID))
		}
		seenUnits[u.ID] = true
	}
	seenMems := map[string]bool{}
	for _, mem := range m.Memories {
		if seenMems[mem.ID] {
			problems = append(problems, fmt.Sprintf("duplicate memory ID %q", mem.ID))
		}
		seenMems[mem.ID] = true
	}
	for _, u := range m.Units {
		if u.ComputeRate <= 0 {
			problems = append(problems, fmt.Sprintf("unit %s: compute rate must be > 0", u.ID))
		}
		if _, ok := m.memories[u.MemoryID]; !ok {
			problems = append(problems, fmt.Sprintf("unit %s: unknown memory domain %q", u.ID, u.MemoryID))
		}
	}
	for _, mem := range m.Memories {
		if mem.CapacityBytes < 0 {
			problems = append(problems, fmt.Sprintf("memory %s: negative capacity", mem.ID))
		}
		if mem.BandwidthBytesSec <= 0 {
			problems = append(problems, fmt.Sprintf("memory %s: bandwidth must be > 0", mem.ID))
		}
	}
	return problems
}
