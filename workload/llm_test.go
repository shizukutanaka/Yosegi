package workload

import (
	"math"
	"testing"
)

func TestKVBytesPerElement(t *testing.T) {
	// Anchored to the Alloy research measurements: q8_0 KV costs
	// 1.0625 bytes per element, q4_0 0.5625, f16 2.
	if got := KVBytesPerElement["q8_0"]; got != 1.0625 {
		t.Fatalf("q8_0 KV: got %v", got)
	}
	if got := KVBytesPerElement["q4_0"]; got != 0.5625 {
		t.Fatalf("q4_0 KV: got %v", got)
	}
	if got := KVBytesPerElement["f16"]; got != 2.0 {
		t.Fatalf("f16 KV: got %v", got)
	}
}

func TestSpecSpeedupBreakEven(t *testing.T) {
	// A worthless draft (alpha=0) must never speed anything up; a
	// perfect draft gains steps-per-cycle over cycle cost; the sign
	// of the gain flips with acceptance — the break-even the cost
	// model relies on when it admits speculative candidates.
	if s := SpecSpeedup(0, 0.25, 3); s >= 1 {
		t.Fatalf("alpha=0 must not speed up: %v", s)
	}
	want := 4.0 / 1.75 // 4 steps per cycle, cost 1 + 3*0.25
	if s := SpecSpeedup(1, 0.25, 3); math.Abs(s-want) > 1e-9 {
		t.Fatalf("alpha=1, depth 3: got %v want %v", s, want)
	}
	lo := SpecSpeedup(0.3, 0.25, 3)
	hi := SpecSpeedup(0.7, 0.25, 3)
	if !(lo < 1 && hi > 1) {
		t.Fatalf("expected sign flip between alpha 0.3 and 0.7, got %v and %v", lo, hi)
	}
	// An expensive draft (ratio near 1) gains much less at the
	// same acceptance — the Alloy finding that a slow draft on a
	// fast target is a net loss.
	if s := SpecSpeedup(0.7, 0.9, 3); s >= hi {
		t.Fatalf("costlier draft must gain less: %v vs %v", s, hi)
	}
}

func TestDecodeTaskShape(t *testing.T) {
	m := ModelSpec{Name: "m", Params: 14e9, Layers: 40, HiddenSize: 5120, KVHeads: 8, HeadDim: 128, Quant: "Q4_K_M", KVType: "q8_0", ContextLen: 4096}
	task := m.DecodeTask(128)
	if len(task.Stages) != 40 {
		t.Fatalf("want 40 layer stages, got %d", len(task.Stages))
	}
	want := m.WeightsBytes() + m.KVBytesPerToken()*4096
	// Per-layer int64 division truncates a few bytes; allow 2/layer.
	if d := task.TotalFootprint() - want; d > 0 || d < -2*int64(m.Layers) {
		t.Fatalf("footprint: got %d want ~%d", task.TotalFootprint(), want)
	}
	if task.Repeats != 128 {
		t.Fatalf("repeats: got %d", task.Repeats)
	}
	st := task.Stages[0]
	if st.StreamedBytes() <= 0 || st.Compute <= 0 {
		t.Fatal("decode stage must carry both memory traffic and compute")
	}
	// The quant ladder must be attached, strictly shrinking.
	if len(task.Reductions) == 0 {
		t.Fatal("decode task must carry quant reductions for no-fit advice")
	}
	for _, r := range task.Reductions {
		if r.FootprintScale >= 1 || r.FootprintScale <= 0 {
			t.Fatalf("reduction %s has bad scale %v", r.Name, r.FootprintScale)
		}
	}
}

func TestPrefillCacheable(t *testing.T) {
	m := ModelSpec{Name: "m", Params: 14e9, Layers: 40, HiddenSize: 5120, KVHeads: 8, HeadDim: 128, Quant: "Q4_K_M", KVType: "q8_0", ContextLen: 4096}
	task := m.PrefillTask(2048, 512, 0.9)
	if task.Stages[0].Cacheable <= 0 {
		t.Fatal("prefill stages must be cacheable (prefix cache is the cheapest win)")
	}
	if task.CacheHitRate != 0.9 {
		t.Fatalf("cache hit rate not carried: %v", task.CacheHitRate)
	}
}
