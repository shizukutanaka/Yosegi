package profiles

import "testing"

// The Alloy rig numbers are anchored to the 2026-11-12 investigation
// report (measured STREAM-like bandwidths and nameplate VRAM).
func TestAlloyRigMatchesResearch(t *testing.T) {
	m := AlloyRig()
	check := func(id string, bw, capBytes float64) {
		u, ok := m.Unit(id)
		if !ok {
			t.Fatalf("unit %s missing", id)
		}
		mem, _ := m.Memory(u.MemoryID)
		if mem.BandwidthBytesSec != bw {
			t.Fatalf("%s bandwidth: got %v want %v", id, mem.BandwidthBytesSec, bw)
		}
		if float64(mem.CapacityBytes) != capBytes {
			t.Fatalf("%s capacity: got %v want %v", id, mem.CapacityBytes, capBytes)
		}
	}
	check("a2000", 288e9, 12*gib)
	check("rx6400", 128e9, 4*gib)
	check("a310", 124e9, 4*gib)
}
