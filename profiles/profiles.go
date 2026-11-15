// Package profiles provides measured machine profiles and example
// tasks. The Alloy rig profile carries the numbers from the Alloy
// heterogeneous-GPU investigation (docs/ALGORITHM.md lists which are
// measured and which are nominal): memory bandwidths and VRAM sizes
// are the report's measured values for the three cards; compute
// rates are nominal spec-sheet figures, present so the model can
// rank prefill, and explicitly marked as the first thing a
// microbenchmark pass should replace.
package profiles

import (
	"yosegi/device"
	"yosegi/workload"
)

const gib = 1 << 30

// AlloyRig is the PC Alloy was built for: RTX A2000 12GB + RX 6400
// 4GB + Arc A310 4GB on a Ryzen 5 5600 host with 16GB RAM.
// Bandwidths (288 / 128 / 124 GB/s) and VRAM are the investigation's
// measured figures; compute rates are nominal (A2000 FP16 tensor
// ~33 TFLOPS dense; RX 6400 ~5.3 TFLOPS FP16 with NO matrix path
// under Vulkan — its effective matmul rate is far lower, which the
// SustainedRate field encodes for long compute-bound runs; A310
// ~13 TFLOPS FP16 class; Ryzen 5600 ~0.4 TFLOPS FP32 AVX2).
func AlloyRig() device.Machine {
	return device.Machine{
		Name: "alloy-rig",
		Units: []device.Unit{
			{ID: "a2000", Name: "NVIDIA RTX A2000 12GB", Kind: device.KindGPU,
				ComputeRate: 33e12, SustainedRate: 30e12, ThermalTauSec: 120,
				MemoryID: "vram-a2000", PowerActiveW: 70},
			{ID: "rx6400", Name: "AMD Radeon RX 6400", Kind: device.KindGPU,
				ComputeRate: 5.3e12, SustainedRate: 2.0e12, ThermalTauSec: 60,
				MemoryID: "vram-rx6400", PowerActiveW: 53},
			{ID: "a310", Name: "Intel Arc A310", Kind: device.KindGPU,
				ComputeRate: 13e12, SustainedRate: 11e12, ThermalTauSec: 60,
				MemoryID: "vram-a310", PowerActiveW: 50},
			{ID: "cpu", Name: "AMD Ryzen 5 5600 (6 cores)", Kind: device.KindCPU,
				ComputeRate: 0.4e12, SustainedRate: 0.38e12, ThermalTauSec: 300,
				MemoryID: "ram", PowerActiveW: 65},
		},
		Memories: []device.Memory{
			{ID: "vram-a2000", Name: "A2000 VRAM", CapacityBytes: 12 * gib, BandwidthBytesSec: 288e9},
			{ID: "vram-rx6400", Name: "RX 6400 VRAM", CapacityBytes: 4 * gib, BandwidthBytesSec: 128e9},
			{ID: "vram-a310", Name: "A310 VRAM", CapacityBytes: 4 * gib, BandwidthBytesSec: 124e9},
			{ID: "ram", Name: "System RAM", CapacityBytes: 16 * gib, BandwidthBytesSec: 45e9},
		},
		Storages: []device.Storage{
			{ID: "nvme", Name: "NVMe SSD", ReadBytesSec: 3.5e9, WriteBytesSec: 3e9},
		},
		Links: []device.Link{
			{FromMemory: "vram-a2000", ToMemory: "ram", BandwidthBytesSec: 25e9, LatencySec: 2e-6},
			{FromMemory: "vram-rx6400", ToMemory: "ram", BandwidthBytesSec: 12e9, LatencySec: 2e-6},
			{FromMemory: "vram-a310", ToMemory: "ram", BandwidthBytesSec: 12e9, LatencySec: 2e-6},
		},
	}
}

// WholePC is a modern desktop with P/E-core CPU, one large GPU, fast
// NVMe and a chassis power cap — the generalized Yosegi target.
func WholePC() device.Machine {
	return device.Machine{
		Name: "whole-pc",
		Units: []device.Unit{
			{ID: "cpu-p", Name: "CPU performance cores x8", Kind: device.KindCPUPerf,
				ComputeRate: 0.9e12, SustainedRate: 0.75e12, ThermalTauSec: 90,
				MemoryID: "ram", PowerActiveW: 125},
			{ID: "cpu-e", Name: "CPU efficiency cores x16", Kind: device.KindCPUEff,
				ComputeRate: 0.5e12, SustainedRate: 0.48e12, ThermalTauSec: 90,
				MemoryID: "ram", PowerActiveW: 35},
			{ID: "gpu0", Name: "Discrete GPU 16GB", Kind: device.KindGPU,
				ComputeRate: 80e12, SustainedRate: 72e12, ThermalTauSec: 180,
				MemoryID: "vram0", PowerActiveW: 320},
			{ID: "igpu", Name: "Integrated GPU", Kind: device.KindGPU,
				ComputeRate: 3e12, SustainedRate: 2.8e12, ThermalTauSec: 120,
				MemoryID: "ram", PowerActiveW: 25},
		},
		Memories: []device.Memory{
			{ID: "ram", Name: "DDR5 RAM", CapacityBytes: 64 * gib, BandwidthBytesSec: 90e9},
			{ID: "vram0", Name: "GPU VRAM", CapacityBytes: 16 * gib, BandwidthBytesSec: 900e9},
		},
		Storages: []device.Storage{
			{ID: "nvme", Name: "NVMe Gen4", ReadBytesSec: 7e9, WriteBytesSec: 6e9},
		},
		Links: []device.Link{
			{FromMemory: "vram0", ToMemory: "ram", BandwidthBytesSec: 50e9, LatencySec: 1.5e-6},
		},
		PowerCapW: 450,
	}
}

// LLM14B is a 14B-class dense model at Q4_K_M (the Alloy rig's
// realistic ceiling): ~8.5GB of weights.
func LLM14B() workload.ModelSpec {
	return workload.ModelSpec{
		Name: "llama-14b-q4km", Params: 14e9, Layers: 40, HiddenSize: 5120,
		KVHeads: 8, HeadDim: 128, Quant: "Q4_K_M", KVType: "q8_0", ContextLen: 4096,
	}
}

// LLM32B is a 32B-class dense model at Q4_K_M: ~19GB of weights —
// too big for the A2000 alone, the capacity case for splitting.
func LLM32B() workload.ModelSpec {
	return workload.ModelSpec{
		Name: "llama-32b-q4km", Params: 32e9, Layers: 64, HiddenSize: 5120,
		KVHeads: 8, HeadDim: 128, Quant: "Q4_K_M", KVType: "q8_0", ContextLen: 4096,
	}
}

// MoE35B is a 35B-A3B-class MoE at Q4_K_M: ~21GB total, ~3B active
// per token, 85% of weight bytes in routed experts — the offload case.
func MoE35B() workload.ModelSpec {
	return workload.ModelSpec{
		Name: "qwen-35b-a3b-q4km", Params: 35e9, Layers: 48, HiddenSize: 4096,
		KVHeads: 4, HeadDim: 128, Quant: "Q4_K_M", KVType: "q8_0", ContextLen: 4096,
		ExpertFraction: 0.85, ActiveExpertRatio: 0.03,
	}
}

// TranscodeTask is a video transcode: two large splittable stages,
// compute-bound with real storage I/O.
func TranscodeTask() workload.Task {
	return workload.Task{
		Name:    "transcode-4k",
		Repeats: 600, // frames in flight batches
		Stages: []workload.Stage{
			{Name: "decode", Compute: 2.5e12, MemBytes: 800 << 20, IOBytes: 40 << 20, Footprint: 2 * gib, Splittable: true, SplitSyncCost: 0.03},
			{Name: "encode", Compute: 6e12, MemBytes: 900 << 20, IOBytes: 12 << 20, Footprint: 3 * gib, Splittable: true, SplitSyncCost: 0.03},
		},
	}
}

// ScanTask is a memory/I-O-bound data scan: streams a dataset from
// NVMe through RAM with trivial compute.
func ScanTask() workload.Task {
	return workload.Task{
		Name:    "dataset-scan",
		Repeats: 1,
		Stages: []workload.Stage{
			{Name: "read", Compute: 1e10, MemBytes: 8 * gib, IOBytes: 8 * gib, Footprint: 1 * gib, Splittable: true, SplitSyncCost: 0.01},
			{Name: "filter", Compute: 8e11, MemBytes: 8 * gib, Footprint: 1 * gib, Splittable: true, SplitSyncCost: 0.01},
		},
	}
}

// CompileTask is a large build: CPU-only stages (a compiler cannot
// run on a GPU), splittable across core clusters.
func CompileTask() workload.Task {
	cpuOnly := []device.UnitKind{device.KindCPUPerf, device.KindCPUEff, device.KindCPU}
	return workload.Task{
		Name:    "build",
		Repeats: 1,
		Stages: []workload.Stage{
			{Name: "compile-units", Compute: 9e14, MemBytes: 20 * gib, IOBytes: 2 * gib, Footprint: 12 * gib, Splittable: true, SplitSyncCost: 0.05, AllowedKinds: cpuOnly},
			{Name: "link", Compute: 6e13, MemBytes: 6 * gib, IOBytes: 1 * gib, Footprint: 8 * gib, AllowedKinds: cpuOnly},
		},
	}
}
