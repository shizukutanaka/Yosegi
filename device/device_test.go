package device

import "testing"

func testMachine() Machine {
	return Machine{
		Name:     "t",
		Memories: []Memory{{ID: "vram", CapacityBytes: 12 << 30, BandwidthBytesSec: 288e9}},
		Units:    []Unit{{ID: "gpu", Kind: KindGPU, ComputeRate: 30e12, MemoryID: "vram"}},
	}
}

func TestSignatureStableAndSensitive(t *testing.T) {
	a := testMachine()
	b := testMachine()
	if a.Signature() != b.Signature() {
		t.Fatal("identical machines must share a signature")
	}
	b.Units = append(b.Units, Unit{ID: "gpu2", Kind: KindGPU, ComputeRate: 5e12, MemoryID: "vram"})
	if a.Signature() == b.Signature() {
		t.Fatal("adding a unit must change the signature (tuned plans must not carry over)")
	}
	c := testMachine()
	c.Units[0].ComputeRate = 31e12
	if a.Signature() == c.Signature() {
		t.Fatal("changing a measured rate must change the signature")
	}
}

func TestApplyMicrobench(t *testing.T) {
	m := testMachine()
	out := m.ApplyMicrobench([]Microbench{
		{UnitID: "gpu", ComputeRate: 28e12, MemoryBandwidth: 270e9},
		{UnitID: "missing", ComputeRate: 1},
	})
	u, _ := out.Unit("gpu")
	if u.ComputeRate != 28e12 {
		t.Fatalf("compute rate not applied: %v", u.ComputeRate)
	}
	mem, _ := out.Memory("vram")
	if mem.BandwidthBytesSec != 270e9 {
		t.Fatalf("bandwidth not applied: %v", mem.BandwidthBytesSec)
	}
	// The receiver is not mutated (a measured profile is a new machine).
	u0, _ := m.Unit("gpu")
	if u0.ComputeRate == 28e12 {
		t.Fatal("ApplyMicrobench must not mutate the receiver")
	}
}

func TestEffectiveRateThermalBlend(t *testing.T) {
	u := Unit{ComputeRate: 100, SustainedRate: 50, ThermalTauSec: 10}
	if got := u.EffectiveRate(0); got != 100 {
		t.Fatalf("zero horizon must be peak: %v", got)
	}
	// Up to tau the unit holds peak; beyond it the rate blends
	// toward sustained as peak - (peak-sust)*(1 - tau/horizon).
	if got := u.EffectiveRate(10); got != 100 {
		t.Fatalf("at tau the unit still holds peak: %v", got)
	}
	if got := u.EffectiveRate(20); got != 75 {
		t.Fatalf("at 2*tau the blend is the midpoint: %v", got)
	}
	if got := u.EffectiveRate(1e9); got < 49.9 || got > 50.1 {
		t.Fatalf("very long horizon must approach sustained: %v", got)
	}
	u2 := Unit{ComputeRate: 100}
	if got := u2.EffectiveRate(1000); got != 100 {
		t.Fatalf("no sustained rate means no derating: %v", got)
	}
}

func TestValidate(t *testing.T) {
	m := testMachine()
	m.Units = append(m.Units, Unit{ID: "gpu", Kind: KindGPU, ComputeRate: 1, MemoryID: "vram"})
	if errs := m.Validate(); len(errs) == 0 {
		t.Fatal("duplicate unit ID must fail validation")
	}
	m2 := testMachine()
	m2.Units[0].MemoryID = "nope"
	if errs := m2.Validate(); len(errs) == 0 {
		t.Fatal("unknown memory reference must fail validation")
	}
	m3 := testMachine()
	if errs := m3.Validate(); len(errs) != 0 {
		t.Fatalf("valid machine rejected: %v", errs)
	}
}
