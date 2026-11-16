package vendor

import (
	"math"
	"strings"
	"testing"
)

// The feasibility and occupancy cases are ported from Tsugi's
// tests/correctness/test_feasibility.py and test_occupancy.py; the
// expected values follow from the primary-source hardware constants.

func TestSmallConfigLaunchableEverywhere(t *testing.T) {
	cfg := KernelConfig{BlockM: 32, BlockN: 32, BlockK: 32, NumStages: 2, NumWarps: 4}
	for _, v := range All {
		rep, err := Check(cfg, v, Options{})
		if err != nil {
			t.Fatalf("Check(%s): %v", v, err)
		}
		if !rep.Launchable() {
			t.Errorf("%s should launch the small config, blockers: %v", v, rep.Blockers())
		}
	}
}

func TestLargeSmemBreaksAMDNotNVIDIA(t *testing.T) {
	// (128*64 + 64*128)*2*4 = 131072 bytes = 128KB of shared memory:
	// NVIDIA Hopper allows 227KB/block (launches), AMD CDNA's LDS is
	// 64KiB/workgroup (does not launch).
	cfg := KernelConfig{BlockM: 128, BlockN: 128, BlockK: 64, NumStages: 4, NumWarps: 8}
	if got := cfg.SharedMemBytes(2); got != 131072 {
		t.Fatalf("SharedMemBytes = %d, want 131072", got)
	}
	nv, err := Check(cfg, NVIDIA, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !nv.Launchable() {
		t.Errorf("nvidia (227KB) should launch, blockers: %v", nv.Blockers())
	}
	amd, err := Check(cfg, AMDCDNA, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if amd.Launchable() {
		t.Fatal("amd_cdna (64KiB LDS) must NOT launch this config")
	}
	found := false
	for _, b := range amd.Blockers() {
		if b.Resource == "shared_mem" {
			found = true
		}
	}
	if !found {
		t.Errorf("amd_cdna blockers should include shared_mem, got %v", amd.Blockers())
	}
}

func TestFirstVendorOnlyFlagsSingleSourceBreak(t *testing.T) {
	cfg := KernelConfig{BlockM: 128, BlockN: 128, BlockK: 64, NumStages: 4, NumWarps: 8}
	only, err := FirstVendorOnly(cfg, NVIDIA, AMDCDNA, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if len(only) == 0 {
		t.Fatal("should flag a vendor-asymmetric launch failure")
	}
	joined := strings.Join(only, "\n")
	if !strings.Contains(joined, "shared_mem") {
		t.Errorf("asymmetry should name shared_mem, got: %s", joined)
	}
}

func TestUnknownVendorRaises(t *testing.T) {
	cfg := KernelConfig{BlockM: 32, BlockN: 32, BlockK: 32, NumStages: 2, NumWarps: 4}
	if _, err := Lookup("intel"); err == nil {
		t.Error("Lookup(intel) should error")
	}
	if _, err := Check(cfg, "intel", Options{}); err == nil {
		t.Error("Check(intel) should error")
	}
	if _, err := Estimate(cfg, "intel", Options{}); err == nil {
		t.Error("Estimate(intel) should error")
	}
}

func TestOccupancyInRange(t *testing.T) {
	cfg := KernelConfig{BlockM: 64, BlockN: 64, BlockK: 32, NumStages: 3, NumWarps: 4}
	for _, v := range All {
		e, err := Estimate(cfg, v, Options{})
		if err != nil {
			t.Fatalf("Estimate(%s): %v", v, err)
		}
		if e.Occupancy < 0.0 || e.Occupancy > 1.0 {
			t.Errorf("%s: occupancy %v out of [0,1]", v, e.Occupancy)
		}
	}
}

func TestOccupancyValuesPinned(t *testing.T) {
	// Hand-derived from the hardware constants for
	// cfg(64,64,32,stages=3,warps=4), smem=24576B, regs/thread=64:
	//   nvidia:  blocks = min(warps 16, regs 8, smem 9) = 8 → 8*4/64 = 0.50 (registers)
	//   amd_cdna: blocks = min(warps 8, regs 8, smem 2) = 2 → 2*4/32 = 0.25 (shared_mem)
	//   amd_rdna: blocks = min(warps 8, regs 12, smem 2) = 2 → 2*4/32 = 0.25 (shared_mem)
	cfg := KernelConfig{BlockM: 64, BlockN: 64, BlockK: 32, NumStages: 3, NumWarps: 4}
	want := map[ID]struct {
		occ     float64
		limited string
	}{
		NVIDIA:  {0.50, "registers"},
		AMDCDNA: {0.25, "shared_mem"},
		AMDRDNA: {0.25, "shared_mem"},
	}
	for v, w := range want {
		e, err := Estimate(cfg, v, Options{})
		if err != nil {
			t.Fatalf("Estimate(%s): %v", v, err)
		}
		if math.Abs(e.Occupancy-w.occ) > 1e-9 {
			t.Errorf("%s occupancy = %v, want %v", v, e.Occupancy, w.occ)
		}
		if e.LimitedBy != w.limited {
			t.Errorf("%s limited_by = %q, want %q", v, e.LimitedBy, w.limited)
		}
	}
	gap, err := OccupancyGap(cfg, NVIDIA, AMDCDNA, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(gap-0.25) > 1e-9 {
		t.Errorf("occupancy gap nvidia/amd_cdna = %v, want 0.25", gap)
	}
}

func TestLargeTileDropsOccupancy(t *testing.T) {
	small, err := Estimate(KernelConfig{BlockM: 32, BlockN: 32, BlockK: 32, NumStages: 2, NumWarps: 4}, AMDCDNA, Options{})
	if err != nil {
		t.Fatal(err)
	}
	large, err := Estimate(KernelConfig{BlockM: 128, BlockN: 128, BlockK: 64, NumStages: 4, NumWarps: 8}, AMDCDNA, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if large.Occupancy > small.Occupancy {
		t.Errorf("large tile occupancy %v should not exceed small %v", large.Occupancy, small.Occupancy)
	}
}

func TestCrossVendorDifferenceExists(t *testing.T) {
	cfg := KernelConfig{BlockM: 64, BlockN: 64, BlockK: 32, NumStages: 3, NumWarps: 4}
	occ, err := CrossVendorOccupancy(cfg, Options{})
	if err != nil {
		t.Fatal(err)
	}
	seen := map[float64]bool{}
	for _, e := range occ {
		seen[e.Occupancy] = true
	}
	if len(seen) <= 1 {
		t.Errorf("expected a cross-vendor occupancy difference, got %v", occ)
	}
}

func TestBlockDimWarnings(t *testing.T) {
	// BlockN=32 is a multiple of NVIDIA's warp (32) but not of CDNA's
	// wavefront (64).
	cfg := KernelConfig{BlockM: 64, BlockN: 32, BlockK: 32, NumStages: 2, NumWarps: 4}
	nv, err := BlockDimWarnings(cfg, NVIDIA)
	if err != nil {
		t.Fatal(err)
	}
	if len(nv) != 0 {
		t.Errorf("nvidia should have no block-dim warnings, got %v", nv)
	}
	amd, err := BlockDimWarnings(cfg, AMDCDNA)
	if err != nil {
		t.Fatal(err)
	}
	if len(amd) != 1 {
		t.Errorf("amd_cdna should warn once (block dim 32 vs wavefront 64), got %v", amd)
	}
}

func TestMatrixCoreFacts(t *testing.T) {
	for _, v := range All {
		ok, err := SupportsMMA(v, 16, 16, 16)
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			t.Errorf("%s should directly support MMA 16x16x16", v)
		}
	}
	// 32x32x8 MFMA is CDNA-only in the table.
	ok, err := SupportsMMA(AMDCDNA, 32, 32, 8)
	if err != nil || !ok {
		t.Errorf("amd_cdna should support MMA 32x32x8 (ok=%v err=%v)", ok, err)
	}
	ok, err = SupportsMMA(NVIDIA, 32, 32, 8)
	if err != nil || ok {
		t.Errorf("nvidia should not list MMA 32x32x8 (ok=%v err=%v)", ok, err)
	}
	// RDNA3 WMMA has no bf16 matrix path; the other two do.
	rdna, err := Lookup(AMDRDNA)
	if err != nil {
		t.Fatal(err)
	}
	if rdna.BF16Matrix {
		t.Error("amd_rdna BF16Matrix should be false")
	}
	for _, v := range []ID{NVIDIA, AMDCDNA} {
		p, err := Lookup(v)
		if err != nil {
			t.Fatal(err)
		}
		if !p.BF16Matrix {
			t.Errorf("%s BF16Matrix should be true", v)
		}
	}
}
