package planner

import (
	"strings"
	"testing"

	"github.com/shizukutanaka/Yosegi/cost"
	"github.com/shizukutanaka/Yosegi/device"
	"github.com/shizukutanaka/Yosegi/profiles"
	"github.com/shizukutanaka/Yosegi/workload"
)

func unitsUsed(p cost.Plan) map[string]bool {
	used := map[string]bool{}
	for _, pl := range p.Placements {
		for _, u := range pl.Units {
			used[u] = true
		}
		if pl.Tier != "" {
			used["tier:"+pl.Tier] = true
		}
	}
	return used
}

// The Alloy investigation's central finding, pinned as a test: a
// model that fits on the fastest card must be planned onto that card
// alone — splitting is a capacity tool, not a speed tool.
func TestAlloyRig14BGoesToA2000Alone(t *testing.T) {
	m := profiles.AlloyRig()
	task := profiles.LLM14B().DecodeTask(128)
	out := Solve(&m, task, Options{TopK: 3})
	if out.NoFit != nil || len(out.Top) == 0 {
		t.Fatalf("14B must fit: %+v", out.NoFit)
	}
	top := out.Top[0]
	used := unitsUsed(top.Plan)
	if len(used) != 1 || !used["a2000"] {
		t.Fatalf("14B top plan must use a2000 alone, used %v", used)
	}
	for _, pl := range top.Plan.Placements {
		if len(pl.Units) != 1 || pl.Tier != "" {
			t.Fatalf("14B top plan must be plain single-unit placements: %+v", pl)
		}
	}
}

// A 32B model cannot fit the A2000's 12GB, so the planner must split
// it across cards (and still return a feasible plan).
func TestAlloyRig32BSplits(t *testing.T) {
	m := profiles.AlloyRig()
	task := profiles.LLM32B().DecodeTask(128)
	out := Solve(&m, task, Options{TopK: 3})
	if out.NoFit != nil || len(out.Top) == 0 {
		t.Fatalf("32B must fit split: %+v", out.NoFit)
	}
	top := out.Top[0]
	if !top.Result.Feasible {
		t.Fatalf("top plan must be feasible: %v", top.Result.Reasons)
	}
	used := unitsUsed(top.Plan)
	gpus := 0
	for _, id := range []string{"a2000", "rx6400", "a310"} {
		if used[id] {
			gpus++
		}
	}
	if gpus < 2 {
		t.Fatalf("32B must use at least 2 GPUs, used %v", used)
	}
}

// The MoE model's experts must make it fit the rig (offload or
// split), at a feasible top plan.
func TestAlloyRigMoEFits(t *testing.T) {
	m := profiles.AlloyRig()
	task := profiles.MoE35B().DecodeTask(128)
	out := Solve(&m, task, Options{TopK: 3})
	if out.NoFit != nil || len(out.Top) == 0 {
		t.Fatalf("MoE 35B must fit somehow: %+v", out.NoFit)
	}
	if !out.Top[0].Result.Feasible {
		t.Fatalf("top MoE plan infeasible: %v", out.Top[0].Result.Reasons)
	}
}

// No-fit honesty: a model far beyond total capacity returns reasons
// and — where a declared reduction would fit — a tested suggestion.
func TestNoFitHonesty(t *testing.T) {
	m := profiles.AlloyRig()
	huge := workload.ModelSpec{Name: "huge", Params: 400e9, Layers: 80, HiddenSize: 8192, KVHeads: 8, HeadDim: 128, Quant: "Q8_0", KVType: "q8_0", ContextLen: 4096}
	out := Solve(&m, huge.DecodeTask(16), Options{TopK: 3})
	if out.NoFit == nil {
		t.Fatal("400B Q8_0 cannot fit this rig; NoFit required")
	}
	if len(out.NoFit.Reasons) == 0 {
		t.Fatal("NoFit must carry reasons")
	}
	if len(out.NoFit.Suggestions) == 0 {
		t.Fatal("NoFit must carry suggestions")
	}
	// A 60B at Q8_0 does not fit, but its quant ladder reaches sizes
	// that do — the suggestion must name a reduction that was tested.
	mid := workload.ModelSpec{Name: "mid", Params: 60e9, Layers: 80, HiddenSize: 8192, KVHeads: 8, HeadDim: 128, Quant: "Q8_0", KVType: "q8_0", ContextLen: 4096}
	out2 := Solve(&m, mid.DecodeTask(16), Options{TopK: 3})
	if out2.NoFit == nil {
		t.Fatal("60B Q8_0 must not fit unreduced")
	}
	found := false
	for _, s := range out2.NoFit.Suggestions {
		if strings.Contains(s, "quant:") && strings.Contains(s, "fits") {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected a tested quant suggestion, got %v", out2.NoFit.Suggestions)
	}
}

// Split shares follow measured throughput, not VRAM size: on the
// Alloy rig the A2000 (288GB/s) must receive the largest share of a
// bandwidth-bound stage, more than the 4GB cards despite the split
// being over the same stage.
func TestSplitSharesFollowThroughput(t *testing.T) {
	m := profiles.AlloyRig()
	st := workload.Stage{Name: "s", Compute: 1e9, MemBytes: 8 << 30, Footprint: 8 << 30, Splittable: true}
	pl, ok := SplitPlacement(&m, st, 8<<30, m.Units)
	if !ok || len(pl.Units) < 2 {
		t.Fatalf("split must succeed across units: %+v ok=%v", pl, ok)
	}
	shares := map[string]float64{}
	sum := 0.0
	for i, id := range pl.Units {
		shares[id] = pl.Shares[i]
		sum += pl.Shares[i]
	}
	if sum < 0.999 || sum > 1.001 {
		t.Fatalf("shares must sum to 1, got %v", sum)
	}
	if !(shares["a2000"] > shares["rx6400"] && shares["a2000"] > shares["a310"]) {
		t.Fatalf("fastest unit must get the largest share: %v", shares)
	}
}

// Whole-PC profiles: a build's CPU-only stages must stay on CPU
// units (kind affinity), and the I/O-bound scan must report the I/O
// bottleneck.
func TestWholePCAffinityAndBottleneck(t *testing.T) {
	m := profiles.WholePC()
	out := Solve(&m, profiles.CompileTask(), Options{TopK: 3})
	if len(out.Top) == 0 {
		t.Fatalf("compile task must fit: %+v", out.NoFit)
	}
	for _, pl := range out.Top[0].Plan.Placements {
		for _, uid := range pl.Units {
			u, _ := m.Unit(uid)
			if u.Kind == device.KindGPU {
				t.Fatalf("compile stage placed on GPU %s", uid)
			}
		}
	}
	scan := Solve(&m, profiles.ScanTask(), Options{TopK: 1})
	if len(scan.Top) == 0 {
		t.Fatalf("scan must fit: %+v", scan.NoFit)
	}
	if scan.Top[0].Result.Bottleneck != "io" {
		t.Fatalf("scan bottleneck must be io, got %q", scan.Top[0].Result.Bottleneck)
	}
}
