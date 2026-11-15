package tuner

import (
	"path/filepath"
	"testing"

	"yosegi/cost"
	"yosegi/device"
	"yosegi/workload"
)

func planFor(unit string, stages int) cost.Plan {
	p := cost.Plan{}
	for i := 0; i < stages; i++ {
		p.Placements = append(p.Placements, cost.Placement{Units: []string{unit}, Shares: []float64{1}})
	}
	return p
}

func measured(p cost.Plan, tps float64) Candidate {
	return Candidate{Plan: p, Measurement: &Measurement{PlanID: p.ID(), Throughput: tps, State: "done"}}
}

func TestPickWinner(t *testing.T) {
	a := measured(planFor("a", 1), 10)
	b := measured(planFor("b", 1), 12)
	if got := PickWinner([]Candidate{a, b}); got != 1 {
		t.Fatalf("highest measured throughput must win, got %d", got)
	}
	// Failed measurements never win.
	f := Candidate{Plan: planFor("c", 1), Measurement: &Measurement{PlanID: "x", State: "failed"}}
	if got := PickWinner([]Candidate{f, a}); got != 1 {
		t.Fatalf("failed candidate must be skipped, got %d", got)
	}
	// Nothing measured: no winner (caller falls back to estimates).
	if got := PickWinner([]Candidate{{Plan: planFor("a", 1)}}); got != -1 {
		t.Fatalf("unmeasured candidates must yield -1, got %d", got)
	}
	// Tie within 1%: the simpler plan wins.
	simple := measured(planFor("a", 1), 10)
	split := cost.Plan{Placements: []cost.Placement{{Units: []string{"a", "b"}, Shares: []float64{0.5, 0.5}}}}
	complex := measured(split, 10.05)
	if got := PickWinner([]Candidate{complex, simple}); got != 1 {
		t.Fatalf("near-tie must prefer the simpler plan, got %d", got)
	}
}

func machineAndTask() (*device.Machine, workload.Task) {
	m := &device.Machine{
		Name:     "t",
		Memories: []device.Memory{{ID: "v", CapacityBytes: 16 << 30, BandwidthBytesSec: 400e9}},
		Units: []device.Unit{
			{ID: "a", Kind: device.KindGPU, ComputeRate: 40e12, MemoryID: "v"},
			{ID: "b", Kind: device.KindGPU, ComputeRate: 10e12, MemoryID: "v"},
		},
	}
	task := workload.Task{Name: "job", Repeats: 1, Stages: []workload.Stage{
		{Name: "s", Compute: 1e12, Footprint: 1 << 30},
	}}
	return m, task
}

func TestStoreRoundTripAndSignatureFallback(t *testing.T) {
	m, task := machineAndTask()
	path := filepath.Join(t.TempDir(), "tuned.json")
	st, err := LoadStore(path)
	if err != nil {
		t.Fatal(err)
	}
	cands := []Candidate{measured(planFor("a", 1), 10), measured(planFor("b", 1), 25)}
	if w := st.Commit(m, task, cands); w != 1 {
		t.Fatalf("commit winner: got %d", w)
	}
	if err := st.Save(); err != nil {
		t.Fatal(err)
	}
	st2, err := LoadStore(path)
	if err != nil {
		t.Fatal(err)
	}
	rec, ok := st2.Lookup(m, task)
	if !ok {
		t.Fatal("lookup after reload must hit")
	}
	if rec.WinnerID != planFor("b", 1).ID() {
		t.Fatalf("winner must persist, got %s", rec.WinnerID)
	}
	// Machine changed (a unit added): the record must NOT apply —
	// the planner falls back to estimates and re-tunes.
	m2 := *m
	m2.Units = append(append([]device.Unit{}, m.Units...),
		device.Unit{ID: "c", Kind: device.KindGPU, ComputeRate: 1e12, MemoryID: "v"})
	if _, ok := st2.Lookup(&m2, task); ok {
		t.Fatal("signature mismatch must miss (fallback to estimates)")
	}
	// Task changed: also a miss.
	task2 := task
	task2.Repeats = 99
	if _, ok := st2.Lookup(m, task2); ok {
		t.Fatal("task signature mismatch must miss")
	}
}
