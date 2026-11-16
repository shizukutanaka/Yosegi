// Package vendor is the cross-vendor kernel gate ported from Tsugi
// (the GPU-vendor portability verification layer): given a tile-kernel
// configuration, it answers — before any code runs, on any vendor —
//
//   - Feasibility: will this configuration *launch at all* on the
//     target GPU, or does it exceed a per-block hardware hard limit
//     (shared memory / LDS, threads, registers)? A configuration tuned
//     for NVIDIA Hopper's 227KB/block of shared memory physically
//     does not fit AMD CDNA's 64KiB/workgroup LDS: that is not "slow",
//     it does not start. Feasibility is a categorical gate upstream of
//     every performance question.
//   - Occupancy: if it launches, what fraction of the compute unit's
//     warp slots can it fill, and which resource limits it? The same
//     configuration occupies NVIDIA and AMD units differently (warp
//     32 vs wavefront 64, different register files and LDS sizes), so
//     a plan that is efficient on one vendor can collapse on the
//     other.
//
// Yosegi's device model is deliberately vendor-neutral and the
// planner's fit test is capacity-only; this package supplies the
// vendor facts and the launch gate the planner's model was missing.
//
// All hardware constants are representative-architecture values from
// primary sources (NVIDIA Hopper Tuning Guide; AMD ROCm GPU arch
// specs; GPUOpen / RDNA3 ISA), carried over from Tsugi's
// feasibility/occupancy tables — see Tsugi's docs/SOURCES.md. They
// describe H100 (Hopper), MI300X (CDNA3) and RX 7900 XTX (RDNA3);
// other architectures need their numbers substituted.
package vendor

import (
	"fmt"
	"sort"
)

// ID identifies a GPU vendor architecture family.
type ID string

const (
	NVIDIA  ID = "nvidia"   // H100 / Hopper (CC 9.0)
	AMDCDNA ID = "amd_cdna" // MI300X / CDNA3 (gfx942)
	AMDRDNA ID = "amd_rdna" // RX 7900 XTX / RDNA3 (gfx1100)
)

// All lists every known vendor in a stable order.
var All = []ID{NVIDIA, AMDCDNA, AMDRDNA}

// KernelConfig is one tile-kernel launch configuration: the tile
// shape, pipeline depth and warp count a kernel is compiled with.
// It is Tsugi's TileConfig.
type KernelConfig struct {
	BlockM    int `json:"block_m"`
	BlockN    int `json:"block_n"`
	BlockK    int `json:"block_k"`
	NumStages int `json:"num_stages"`
	NumWarps  int `json:"num_warps"`
}

// SharedMemBytes estimates the shared-memory footprint of the
// configuration: the two operand tiles (BM×BK and BK×BN) held for
// every pipeline stage. This is the resource expression both the
// feasibility gate and the occupancy estimate use.
func (c KernelConfig) SharedMemBytes(dtypeBytes int) int {
	return (c.BlockM*c.BlockK + c.BlockK*c.BlockN) * dtypeBytes * c.NumStages
}

// Profile is one vendor architecture's hardware facts: the per-block
// hard limits a launch must fit (the feasibility gate) and the
// per-compute-unit capacities that decide how many blocks can be
// resident at once (the occupancy estimate). The two sets are
// different things — capacity is not launchability.
type Profile struct {
	ID   ID
	Name string

	WarpSize int // threads per warp (NVIDIA) / wavefront (AMD)

	// Per-block hard limits: exceeding one fails the launch.
	MaxSharedMemPerBlock int // bytes of shared memory / LDS per block (workgroup)
	MaxThreadsPerBlock   int
	MaxRegsPerThread     int

	// Per-compute-unit (SM / CU) capacities for occupancy.
	RegsPerUnit      int // 32-bit register slots per unit
	SharedMemPerUnit int // bytes of shared memory / LDS per unit
	MaxWarpsPerUnit  int // resident warp/wavefront slots per unit

	// Matrix-core facts (from Tsugi's portability tables).
	BF16Matrix   bool     // bf16 is handled by the matrix cores directly
	SupportedMMA [][3]int // representative directly-supported MMA shapes (M, N, K)
}

var profiles = map[ID]Profile{
	// NVIDIA H100 (Hopper, CC 9.0): Hopper Tuning Guide — 227KB
	// opt-in dynamic smem/block, 1024 threads/block, 255 regs/thread;
	// 64K 32-bit regs/SM, 228KB smem/SM (carveout), 64 warps/SM.
	NVIDIA: {
		ID: NVIDIA, Name: "nvidia (H100/Hopper)", WarpSize: 32,
		MaxSharedMemPerBlock: 227 * 1024, MaxThreadsPerBlock: 1024, MaxRegsPerThread: 255,
		RegsPerUnit: 65536, SharedMemPerUnit: 228 * 1024, MaxWarpsPerUnit: 64,
		BF16Matrix:   true,
		SupportedMMA: [][3]int{{16, 16, 16}, {8, 8, 4}, {16, 8, 16}}, // WMMA/HMMA
	},
	// AMD MI300X (CDNA3, gfx942): ROCm GPU arch specs — LDS
	// 64KiB/workgroup, 1024 work-items/workgroup, VGPR 256/lane;
	// VGPR 512KiB/CU = 131072 32-bit slots, LDS 64KiB/CU, 32
	// wavefronts/CU.
	AMDCDNA: {
		ID: AMDCDNA, Name: "amd_cdna (MI300X/CDNA3)", WarpSize: 64,
		MaxSharedMemPerBlock: 64 * 1024, MaxThreadsPerBlock: 1024, MaxRegsPerThread: 256,
		RegsPerUnit: 131072, SharedMemPerUnit: 64 * 1024, MaxWarpsPerUnit: 32,
		BF16Matrix:   true,
		SupportedMMA: [][3]int{{16, 16, 16}, {32, 32, 8}, {16, 16, 4}}, // MFMA
	},
	// AMD RX 7900 XTX (RDNA3, gfx1100): GPUOpen / RDNA3 ISA — LDS
	// 64KiB usable/workgroup, 1024 work-items, VGPR 256/lane
	// (wave32); 98304 VGPR slots/CU, LDS 64KiB usable/CU, 32
	// waves/CU. RDNA3 WMMA has no bf16 matrix path.
	AMDRDNA: {
		ID: AMDRDNA, Name: "amd_rdna (RX 7900 XTX/RDNA3)", WarpSize: 32,
		MaxSharedMemPerBlock: 64 * 1024, MaxThreadsPerBlock: 1024, MaxRegsPerThread: 256,
		RegsPerUnit: 98304, SharedMemPerUnit: 64 * 1024, MaxWarpsPerUnit: 32,
		BF16Matrix:   false,
		SupportedMMA: [][3]int{{16, 16, 16}}, // RDNA3 WMMA
	},
}

// Lookup returns the profile for a vendor ID.
func Lookup(id ID) (Profile, error) {
	p, ok := profiles[id]
	if !ok {
		return Profile{}, fmt.Errorf("unknown vendor: %s", id)
	}
	return p, nil
}

// SupportsMMA reports whether the vendor's matrix cores directly
// support the MMA shape (m, n, k). Shapes outside the table are
// decomposed or demoted — a portability risk, not an error.
func SupportsMMA(id ID, m, n, k int) (bool, error) {
	p, err := Lookup(id)
	if err != nil {
		return false, err
	}
	for _, s := range p.SupportedMMA {
		if s == [3]int{m, n, k} {
			return true, nil
		}
	}
	return false, nil
}

// Options tunes a check. Zero fields take the defaults Tsugi uses:
// 64 registers per thread (a typical GEMM range is 64–128) and
// 2-byte elements (fp16).
type Options struct {
	RegsPerThread int
	DTypeBytes    int
}

func (o Options) regs() int {
	if o.RegsPerThread > 0 {
		return o.RegsPerThread
	}
	return 64
}

func (o Options) dtypeBytes() int {
	if o.DTypeBytes > 0 {
		return o.DTypeBytes
	}
	return 2
}

// ResourceCheck is one per-block resource against its hard limit.
type ResourceCheck struct {
	Resource string // "shared_mem" | "threads" | "registers"
	Required int
	Limit    int
}

// Fits reports whether the requirement is within the limit.
func (c ResourceCheck) Fits() bool { return c.Required <= c.Limit }

// Feasibility is the launch gate's verdict for one vendor.
type Feasibility struct {
	Vendor ID
	Checks []ResourceCheck
}

// Launchable reports whether every resource fits its hard limit.
func (f Feasibility) Launchable() bool {
	for _, c := range f.Checks {
		if !c.Fits() {
			return false
		}
	}
	return true
}

// Blockers returns the resources that exceed their limit — the
// reasons the kernel will not launch on this vendor.
func (f Feasibility) Blockers() []ResourceCheck {
	var out []ResourceCheck
	for _, c := range f.Checks {
		if !c.Fits() {
			out = append(out, c)
		}
	}
	return out
}

// Check runs the feasibility gate: does cfg fit vendor v's per-block
// hard limits? This is upstream of occupancy — a configuration that
// cannot launch has no meaningful occupancy.
func Check(cfg KernelConfig, v ID, opts Options) (Feasibility, error) {
	p, err := Lookup(v)
	if err != nil {
		return Feasibility{}, err
	}
	threads := cfg.NumWarps * p.WarpSize
	rep := Feasibility{Vendor: v}
	rep.Checks = append(rep.Checks,
		ResourceCheck{"shared_mem", cfg.SharedMemBytes(opts.dtypeBytes()), p.MaxSharedMemPerBlock},
		ResourceCheck{"threads", threads, p.MaxThreadsPerBlock},
		ResourceCheck{"registers", opts.regs(), p.MaxRegsPerThread},
	)
	return rep, nil
}

// Occupancy is the estimated occupancy of one configuration on one
// vendor: the fraction of the compute unit's warp slots that resident
// blocks can fill, and the resource that limits it.
type Occupancy struct {
	Vendor          ID
	Occupancy       float64 // 0.0–1.0
	LimitedBy       string  // "registers" | "shared_mem" | "warps"
	ThreadsPerBlock int
	SharedMemBytes  int
}

// Estimate computes the occupancy of cfg on vendor v from the three
// residency constraints (warp slots, register file, shared memory).
func Estimate(cfg KernelConfig, v ID, opts Options) (Occupancy, error) {
	p, err := Lookup(v)
	if err != nil {
		return Occupancy{}, err
	}
	threads := cfg.NumWarps * p.WarpSize
	smem := cfg.SharedMemBytes(opts.dtypeBytes())

	byWarps := p.MaxWarpsPerUnit / max(1, cfg.NumWarps)
	byRegs := p.RegsPerUnit / max(1, threads*opts.regs())
	bySmem := p.SharedMemPerUnit / max(1, smem)

	blocks := min(byWarps, byRegs, bySmem)
	// The limiter is the smallest constraint; ties resolve by the
	// resource name, matching Tsugi's tuple-min behaviour.
	limiter := "warps"
	best := byWarps
	for _, cand := range []struct {
		n    int
		name string
	}{{byRegs, "registers"}, {bySmem, "shared_mem"}} {
		if cand.n < best || (cand.n == best && cand.name < limiter) {
			best, limiter = cand.n, cand.name
		}
	}

	occ := float64(blocks*cfg.NumWarps) / float64(p.MaxWarpsPerUnit)
	if occ > 1.0 {
		occ = 1.0
	}
	return Occupancy{
		Vendor: v, Occupancy: occ, LimitedBy: limiter,
		ThreadsPerBlock: threads, SharedMemBytes: smem,
	}, nil
}

// CrossVendorFeasibility runs the launch gate for every known
// vendor, keyed by vendor ID.
func CrossVendorFeasibility(cfg KernelConfig, opts Options) (map[ID]Feasibility, error) {
	out := make(map[ID]Feasibility, len(All))
	for _, v := range All {
		f, err := Check(cfg, v, opts)
		if err != nil {
			return nil, err
		}
		out[v] = f
	}
	return out, nil
}

// CrossVendorOccupancy estimates occupancy for every known vendor.
func CrossVendorOccupancy(cfg KernelConfig, opts Options) (map[ID]Occupancy, error) {
	out := make(map[ID]Occupancy, len(All))
	for _, v := range All {
		e, err := Estimate(cfg, v, opts)
		if err != nil {
			return nil, err
		}
		out[v] = e
	}
	return out, nil
}

// FirstVendorOnly lists the resources that launch on vendor a but
// not on vendor b (or vice versa): the places where a configuration
// that "works on my GPU" silently breaks the single-source promise
// on the other vendor.
func FirstVendorOnly(cfg KernelConfig, a, b ID, opts Options) ([]string, error) {
	ra, err := Check(cfg, a, opts)
	if err != nil {
		return nil, err
	}
	rb, err := Check(cfg, b, opts)
	if err != nil {
		return nil, err
	}
	var out []string
	for i := range ra.Checks {
		ca, cb := ra.Checks[i], rb.Checks[i]
		if ca.Fits() == cb.Fits() {
			continue
		}
		okV, ngV, ng := a, b, cb
		if !ca.Fits() {
			okV, ngV, ng = b, a, ca
		}
		out = append(out, fmt.Sprintf(
			"%s: launches on %s but not on %s (required=%d > %s limit %d)",
			ca.Resource, okV, ngV, ng.Required, ngV, ng.Limit))
	}
	return out, nil
}

// OccupancyGap is the absolute occupancy difference between two
// vendors for the same configuration: the size of the "performance
// collapses on one side only" risk.
func OccupancyGap(cfg KernelConfig, a, b ID, opts Options) (float64, error) {
	ea, err := Estimate(cfg, a, opts)
	if err != nil {
		return 0, err
	}
	eb, err := Estimate(cfg, b, opts)
	if err != nil {
		return 0, err
	}
	gap := ea.Occupancy - eb.Occupancy
	if gap < 0 {
		gap = -gap
	}
	return gap, nil
}

// BlockDimWarnings flags block dimensions that are not multiples of
// the vendor's warp/wavefront size — an implicit warp-size assumption
// is one of the main ways a kernel's behaviour (and occupancy)
// diverges across vendors. It checks the tile's M and N dimensions.
func BlockDimWarnings(cfg KernelConfig, v ID) ([]string, error) {
	p, err := Lookup(v)
	if err != nil {
		return nil, err
	}
	var out []string
	for _, d := range []int{cfg.BlockM, cfg.BlockN} {
		if d%p.WarpSize != 0 {
			out = append(out, fmt.Sprintf(
				"block dim %d is not a multiple of %s warp/wavefront=%d: occupancy loss and cross-vendor behaviour divergence",
				d, v, p.WarpSize))
		}
	}
	return out, nil
}

// VendorIDs returns the known vendor IDs in stable order (a copy of
// All, for callers that iterate reports deterministically).
func VendorIDs() []ID {
	out := append([]ID(nil), All...)
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}
