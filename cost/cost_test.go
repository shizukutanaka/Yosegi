package cost

import (
	"strings"
	"testing"

	"github.com/shizukutanaka/Yosegi/device"
	"github.com/shizukutanaka/Yosegi/workload"
)

func twoGPU() *device.Machine {
	return &device.Machine{
		Name: "t",
		Memories: []device.Memory{
			{ID: "va", CapacityBytes: 16 << 30, BandwidthBytesSec: 400e9},
			{ID: "vb", CapacityBytes: 16 << 30, BandwidthBytesSec: 100e9},
		},
		Units: []device.Unit{
			{ID: "fast", Kind: device.KindGPU, ComputeRate: 40e12, MemoryID: "va"},
			{ID: "slow", Kind: device.KindGPU, ComputeRate: 10e12, MemoryID: "vb"},
		},
	}
}

func oneStage(st workload.Stage) workload.Task {
	st.Name = "s"
	return workload.Task{Name: "t", Stages: []workload.Stage{st}, Repeats: 1}
}

func whole(unit string) Plan {
	return Plan{Placements: []Placement{{Units: []string{unit}, Shares: []float64{1}}}}
}

func TestMonotonicityFasterUnitWins(t *testing.T) {
	m := twoGPU()
	computeBound := oneStage(workload.Stage{Compute: 4e13, MemBytes: 1e8, Footprint: 1 << 30})
	memBound := oneStage(workload.Stage{Compute: 1e9, MemBytes: 4e10, Footprint: 1 << 30})
	for name, task := range map[string]workload.Task{"compute": computeBound, "memory": memBound} {
		f := Evaluate(m, task, whole("fast"))
		s := Evaluate(m, task, whole("slow"))
		if !f.Feasible || !s.Feasible {
			t.Fatalf("%s: both placements must be feasible", name)
		}
		if !(f.TotalTime < s.TotalTime) {
			t.Fatalf("%s: fast unit must be faster (%v vs %v)", name, f.TotalTime, s.TotalTime)
		}
	}
}

func TestSplitShareMonotonicity(t *testing.T) {
	m := twoGPU()
	task := oneStage(workload.Stage{Compute: 4e13, MemBytes: 1e8, Footprint: 1 << 30, Splittable: true})
	split := func(fastShare float64) Plan {
		return Plan{Placements: []Placement{{
			Units:  []string{"fast", "slow"},
			Shares: []float64{fastShare, 1 - fastShare},
		}}}
	}
	balanced := Evaluate(m, task, split(0.5))
	skewed := Evaluate(m, task, split(0.8)) // proportional-ish: fast is 4x
	if !(skewed.TotalTime < balanced.TotalTime) {
		t.Fatalf("giving the faster unit more share must help: %v vs %v", skewed.TotalTime, balanced.TotalTime)
	}
	// And the proportional split must beat the worst single placement.
	worst := Evaluate(m, task, whole("slow"))
	if !(skewed.TotalTime < worst.TotalTime) {
		t.Fatalf("split must beat the slow unit alone: %v vs %v", skewed.TotalTime, worst.TotalTime)
	}
}

func TestFeasibilityBoundary(t *testing.T) {
	m := twoGPU()
	fits := oneStage(workload.Stage{Compute: 1e9, Footprint: 16 << 30})
	if r := Evaluate(m, fits, whole("fast")); !r.Feasible {
		t.Fatalf("footprint == capacity must fit: %v", r.Reasons)
	}
	over := oneStage(workload.Stage{Compute: 1e9, Footprint: (16 << 30) + 1})
	r := Evaluate(m, over, whole("fast"))
	if r.Feasible {
		t.Fatal("footprint == capacity+1 must not fit")
	}
	joined := strings.Join(r.Reasons, " ")
	if !strings.Contains(joined, "over capacity") {
		t.Fatalf("reason must name the overflow, got: %s", joined)
	}
}

func TestSpecBreakEvenInCostModel(t *testing.T) {
	m := twoGPU()
	base := workload.Task{Name: "spec", Repeats: 40, Stages: []workload.Stage{
		{Name: "s", Compute: 4e13, MemBytes: 1e8, Footprint: 1 << 30},
	}}
	plain := Evaluate(m, base, whole("fast"))
	if !plain.Feasible {
		t.Fatal("plain plan must be feasible")
	}
	withSpec := func(alpha float64) Result {
		tk := base
		tk.Spec = &workload.SpecCapability{DraftWorkRatio: 0.1, Acceptance: alpha}
		p := whole("fast")
		p.Spec = &SpecChoice{Depth: 3, DraftUnit: "fast"}
		return Evaluate(m, tk, p)
	}
	if r := withSpec(0.85); !(r.TotalTime < plain.TotalTime*0.7) {
		t.Fatalf("high acceptance must clearly win: spec %v vs plain %v", r.TotalTime, plain.TotalTime)
	}
	if r := withSpec(0.10); !(r.TotalTime > plain.TotalTime) {
		t.Fatalf("low acceptance must lose: spec %v vs plain %v", r.TotalTime, plain.TotalTime)
	}
}

func TestPrefixCacheEffect(t *testing.T) {
	m := twoGPU()
	mk := func(hit float64) workload.Task {
		return workload.Task{Name: "c", Repeats: 10, CacheHitRate: hit, Stages: []workload.Stage{
			{Name: "s", Compute: 4e13, MemBytes: 1e8, Footprint: 1 << 30, Cacheable: 1},
		}}
	}
	cold := Evaluate(m, mk(0), whole("fast"))
	warm := Evaluate(m, mk(0.9), whole("fast"))
	if !(warm.TotalTime < cold.TotalTime*0.5) {
		t.Fatalf("90%% cache hits on fully cacheable work must roughly halve+ time: %v vs %v", warm.TotalTime, cold.TotalTime)
	}
}

func TestThermalDerating(t *testing.T) {
	mk := func(sustained float64) *device.Machine {
		return &device.Machine{
			Name:     "th",
			Memories: []device.Memory{{ID: "v", CapacityBytes: 16 << 30, BandwidthBytesSec: 400e9}},
			Units:    []device.Unit{{ID: "g", Kind: device.KindGPU, ComputeRate: 40e12, SustainedRate: sustained, ThermalTauSec: 1, MemoryID: "v"}},
		}
	}
	task := workload.Task{Name: "long", Repeats: 100, Stages: []workload.Stage{
		{Name: "s", Compute: 4e13, MemBytes: 1e8, Footprint: 1 << 30},
	}}
	peak := Evaluate(mk(0), task, whole("g"))
	derated := Evaluate(mk(20e12), task, whole("g"))
	if !(derated.TotalTime > peak.TotalTime*1.2) {
		t.Fatalf("a long job on a throttling unit must be predicted slower: %v vs %v", derated.TotalTime, peak.TotalTime)
	}
}

func TestInterferenceSharedBandwidth(t *testing.T) {
	m := twoGPU()
	mkTask := func() workload.Task {
		return workload.Task{Name: "stream", Repeats: 10, Stages: []workload.Stage{
			{Name: "s", Compute: 1e9, MemBytes: 1e10, Footprint: 1 << 30},
		}}
	}
	same := EvaluateConcurrent(m, []Instance{
		{Task: mkTask(), Plan: whole("fast")},
		{Task: mkTask(), Plan: whole("fast")},
	})
	diff := EvaluateConcurrent(m, []Instance{
		{Task: mkTask(), Plan: whole("fast")},
		{Task: mkTask(), Plan: whole("slow")},
	})
	// Two bandwidth hogs on one domain must finish later than when
	// spread — even though "slow" alone is 4x slower, sharing 400GB/s
	// hurts less than... (assert direction only via the fast task's
	// own adjusted time).
	if !(same.Adjusted[0].TotalTime > diff.Adjusted[0].TotalTime) {
		t.Fatalf("sharing a domain must slow the fast task: same %v vs diff %v",
			same.Adjusted[0].TotalTime, diff.Adjusted[0].TotalTime)
	}
}
