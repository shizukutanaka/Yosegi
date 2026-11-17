// Package sim is Yosegi's verification harness: it generates small
// synthetic machines and tasks, then compares the planner's choice
// against exhaustive search (brute force) over a defined plan space,
// measuring the optimality gap instead of asserting optimality.
//
// The brute-force space (documented in docs/ALGORITHM.md):
//   - each stage assigned wholly to one unit (home tier), or
//   - each splittable stage split across ALL units with the planner's
//     own capped proportional shares.
//
// Speculation and tier-offload variants are excluded from the brute
// space: they are orthogonal multipliers whose break-evens are
// covered by unit tests. When the planner wins via those variants the
// gap is reported as negative (planner better than the brute space).
package sim

import (
	"math"
	"math/rand"
	"sort"

	"github.com/shizukutanaka/Yosegi/cost"
	"github.com/shizukutanaka/Yosegi/device"
	"github.com/shizukutanaka/Yosegi/planner"
	"github.com/shizukutanaka/Yosegi/workload"
)

// BruteForce returns the best feasible plan in the brute-force space
// under the cost model, or false when nothing in the space is
// feasible.
func BruteForce(m *device.Machine, t workload.Task) (cost.Plan, cost.Result, bool) {
	n := len(t.Stages)
	k := len(m.Units)
	if n == 0 || k == 0 {
		return cost.Plan{}, cost.Result{}, false
	}
	// Per-stage options: unit indices 0..k-1, plus option k = split
	// across all units (only when the stage is splittable).
	splitPl, splitOK := map[int]cost.Placement{}, map[int]bool{}
	for i, st := range t.Stages {
		if st.Splittable {
			pl, ok := planner.SplitPlacement(m, st, t.TotalFootprint(), m.Units)
			splitPl[i], splitOK[i] = pl, ok
		}
	}
	var best cost.Plan
	var bestRes cost.Result
	found := false
	choice := make([]int, n)
	var rec func(stage int)
	rec = func(stage int) {
		if stage == n {
			p := cost.Plan{Placements: make([]cost.Placement, n)}
			for i, c := range choice {
				if c == k {
					p.Placements[i] = splitPl[i]
				} else {
					p.Placements[i] = cost.Placement{Units: []string{m.Units[c].ID}, Shares: []float64{1}}
				}
			}
			r := cost.Evaluate(m, t, p)
			if r.Feasible && (!found || r.TotalTime < bestRes.TotalTime) {
				best, bestRes, found = p, r, true
			}
			return
		}
		for c := 0; c < k; c++ {
			if !t.Stages[stage].Allows(m.Units[c]) {
				continue
			}
			choice[stage] = c
			rec(stage + 1)
		}
		if splitOK[stage] {
			choice[stage] = k
			rec(stage + 1)
		}
	}
	rec(0)
	return best, bestRes, found
}

// Comparison is one case's planner-vs-brute outcome.
type Comparison struct {
	PlannerTime float64
	BruteTime   float64
	// Gap is (planner-brute)/brute: 0 = planner matched the brute
	// optimum, positive = planner worse, negative = planner beat the
	// brute space (possible via offload/spec variants).
	Gap         float64
	BothNoFit   bool
	PlannerOnly bool // planner found a plan, brute space had none
	BruteOnly   bool // brute found a plan, planner did not (a planner bug by definition)
}

// Compare runs the planner (without spec/offload variants, to compare
// within the brute space) and brute force on one case.
func Compare(m *device.Machine, t workload.Task) Comparison {
	out := planner.Solve(m, t, planner.Options{TopK: 1})
	_, br, bOK := BruteForce(m, t)
	var c Comparison
	switch {
	case len(out.Top) == 0 && !bOK:
		c.BothNoFit = true
	case len(out.Top) == 0 && bOK:
		c.BruteOnly = true
		c.BruteTime = br.TotalTime
	case len(out.Top) > 0 && !bOK:
		c.PlannerOnly = true
		c.PlannerTime = out.Top[0].Result.TotalTime
	default:
		c.PlannerTime = out.Top[0].Result.TotalTime
		c.BruteTime = br.TotalTime
		c.Gap = (c.PlannerTime - c.BruteTime) / c.BruteTime
	}
	return c
}

// Stats aggregates comparisons.
type Stats struct {
	Cases         int
	Compared      int // both found a plan
	AgreementRate float64
	MeanGap       float64
	P95Gap        float64
	MaxGap        float64
	PlannerBetter int // gap < -1e-9 (won via richer candidate shapes)
	BothNoFit     int
	PlannerOnly   int
	BruteOnly     int
}

// Run executes n generated cases from a seed and aggregates.
func Run(seed int64, n int) Stats {
	rng := rand.New(rand.NewSource(seed))
	var gaps []float64
	var st Stats
	st.Cases = n
	for i := 0; i < n; i++ {
		m, t := GenCase(rng)
		c := Compare(&m, t)
		switch {
		case c.BothNoFit:
			st.BothNoFit++
		case c.PlannerOnly:
			st.PlannerOnly++
		case c.BruteOnly:
			// Counted separately, never folded into gap statistics:
			// a feasible whole-stage assignment the planner's
			// generators all missed is a generator bug, reported
			// as its own number.
			st.BruteOnly++
		default:
			st.Compared++
			gaps = append(gaps, c.Gap)
			if c.Gap < -1e-9 {
				st.PlannerBetter++
			}
		}
	}
	if st.Compared > 0 {
		agree := 0
		sum := 0.0
		sorted := append([]float64(nil), gaps...)
		sort.Float64s(sorted)
		for _, g := range gaps {
			if math.Abs(g) <= 1e-9 {
				agree++
			}
			if g > 0 {
				sum += g
			}
			if g > st.MaxGap {
				st.MaxGap = g
			}
		}
		st.AgreementRate = float64(agree) / float64(st.Compared)
		st.MeanGap = sum / float64(st.Compared)
		idx := int(0.95 * float64(len(sorted)-1))
		st.P95Gap = sorted[idx]
	}
	return st
}

// GenCase generates one small random machine+task pair. Sizes are
// kept small enough for exhaustive search, and capacities are drawn
// so that some cases are tight (feasibility matters).
func GenCase(rng *rand.Rand) (device.Machine, workload.Task) {
	nUnits := 2 + rng.Intn(2) // 2..3 units
	memIDs := []string{"mem-a", "mem-b"}
	m := device.Machine{Name: "sim"}
	// Two domains; units share the second domain sometimes (CPU-like).
	m.Memories = []device.Memory{
		{ID: "mem-a", Name: "A", CapacityBytes: int64((2 + rng.Intn(14))) << 30, BandwidthBytesSec: (50 + float64(rng.Intn(250))) * 1e9},
		{ID: "mem-b", Name: "B", CapacityBytes: int64((4 + rng.Intn(28))) << 30, BandwidthBytesSec: (20 + float64(rng.Intn(100))) * 1e9},
	}
	m.Links = []device.Link{{FromMemory: "mem-a", ToMemory: "mem-b", BandwidthBytesSec: (4 + float64(rng.Intn(20))) * 1e9, LatencySec: 1e-6}}
	for i := 0; i < nUnits; i++ {
		mem := memIDs[0]
		if i > 0 && rng.Intn(2) == 0 {
			mem = memIDs[1]
		}
		m.Units = append(m.Units, device.Unit{
			ID:           string(rune('u')) + string(rune('0'+i)),
			Name:         "unit",
			Kind:         device.KindGPU,
			ComputeRate:  (0.2 + rng.Float64()*4) * 1e12,
			MemoryID:     mem,
			PowerActiveW: 50 + rng.Float64()*200,
		})
	}
	m.Storages = []device.Storage{{ID: "ssd", Name: "SSD", ReadBytesSec: (0.5 + rng.Float64()*6) * 1e9}}

	nStages := 3 + rng.Intn(4) // 3..6 stages
	t := workload.Task{Name: "sim-task", Repeats: 1 + rng.Intn(20)}
	for i := 0; i < nStages; i++ {
		st := workload.Stage{
			Name:       "s",
			Compute:    rng.Float64() * 2e12,
			MemBytes:   int64(rng.Float64() * 4e9),
			Footprint:  int64(rng.Float64() * 6e9),
			Activation: int64(rng.Float64() * 1e6),
			Splittable: rng.Intn(2) == 0,
		}
		if st.Splittable {
			st.SplitSyncCost = 0.02 + rng.Float64()*0.1
		}
		if rng.Intn(4) == 0 {
			st.IOBytes = int64(rng.Float64() * 2e9)
		}
		t.Stages = append(t.Stages, st)
	}
	return m, t
}
