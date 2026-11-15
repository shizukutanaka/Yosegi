package planner

import (
	"math"

	"yosegi/cost"
	"yosegi/device"
	"yosegi/workload"
)

// This file holds the feasibility repair and the local-search polish.
// The generators produce good shapes cheaply, but capacity is a
// global constraint: the time-optimal assignment may overflow a
// memory domain by one stage, and the fix is often a swap (move the
// I/O-bound stage out, move the compute-bound stage in) that no
// greedy pass finds. Repair + hill climbing on the TRUE objective
// (cost.Evaluate's total time) closes most of that gap.

// domainUse mirrors cost.Evaluate's residency accounting: dense
// shares live on each unit's home domain, expert shares on the tier.
func domainUse(m *device.Machine, t workload.Task, p cost.Plan) map[string]int64 {
	used := map[string]int64{}
	for i, pl := range p.Placements {
		st := t.Stages[i]
		dense := st.Footprint - st.ExpertBytes
		if dense < 0 {
			dense = 0
		}
		for j, uid := range pl.Units {
			u, _ := m.Unit(uid)
			used[u.MemoryID] += int64(float64(dense) * shareOf(pl, j))
			tier := pl.Tier
			if tier == "" {
				tier = u.MemoryID
			}
			used[tier] += int64(float64(st.ExpertBytes) * shareOf(pl, j))
		}
	}
	return used
}

func overflowDomain(m *device.Machine, used map[string]int64) (string, bool) {
	for dom, bytes := range used {
		mem, _ := m.Memory(dom)
		if mem.CapacityBytes > 0 && bytes > mem.CapacityBytes {
			return dom, true
		}
	}
	return "", false
}

// repair moves whole stages out of overflowing domains — always the
// stage whose move costs the least stage time — until the plan fits
// or no move helps. Split placements are collapsed to their dominant
// unit when moved (repair favors simple, feasible shapes).
func repair(m *device.Machine, t workload.Task, p cost.Plan) (cost.Plan, bool) {
	// Hopeless case: the task's total footprint exceeds the whole
	// machine's memory. No reassignment can fix that (every stage
	// must live somewhere), so don't burn time trying.
	var totalCap int64
	for _, mem := range m.Memories {
		totalCap += mem.CapacityBytes
	}
	if t.TotalFootprint() > totalCap {
		return p, false
	}
	cur := clonePlan(p)
	for iter := 0; iter < len(t.Stages)+2; iter++ {
		if cost.Evaluate(m, t, cur).Feasible {
			return cur, true
		}
		dom, over := overflowDomain(m, domainUse(m, t, cur))
		if !over {
			return cur, false // infeasible for a non-capacity reason
		}
		// Candidate stages: those with resident bytes in dom.
		used := domainUse(m, t, cur)
		bestStage, bestTarget, bestCost := -1, "", math.Inf(1)
		for i, pl := range cur.Placements {
			st := t.Stages[i]
			inDom := false
			for _, uid := range pl.Units {
				u, _ := m.Unit(uid)
				if u.MemoryID == dom || pl.Tier == dom {
					inDom = true
				}
			}
			if !inDom {
				continue
			}
			curTime := stagePlacementTime(m, st, pl)
			for _, u := range m.Units {
				if u.MemoryID == dom || !st.Allows(u) {
					continue
				}
				// Target domain must have room for the whole stage.
				mem, _ := m.Memory(u.MemoryID)
				if mem.CapacityBytes > 0 && used[u.MemoryID]+st.Footprint > mem.CapacityBytes {
					continue
				}
				c := stageTimeOn(m, st, u) - curTime
				if c < bestCost {
					bestCost, bestStage, bestTarget = c, i, u.ID
				}
			}
		}
		if bestStage < 0 {
			return cur, false
		}
		cur.Placements[bestStage] = cost.Placement{Units: []string{bestTarget}, Shares: []float64{1}}
	}
	return cur, cost.Evaluate(m, t, cur).Feasible
}

// stagePlacementTime is the generator-side time of a stage under an
// arbitrary placement.
func stagePlacementTime(m *device.Machine, st workload.Stage, pl cost.Placement) float64 {
	if len(pl.Units) == 1 {
		u, _ := m.Unit(pl.Units[0])
		return stageTimeOn(m, st, u)
	}
	return stageSplitTime(m, st, pl)
}

// hillClimb polishes a feasible plan: repeatedly apply the single
// best improving neighbor — moving one whole stage to another unit,
// or swapping the units of two whole stages — under the true
// objective. Split placements are left untouched (their shares are
// already the model's optimum for that shape).
func hillClimb(m *device.Machine, t workload.Task, p cost.Plan) cost.Plan {
	cur := clonePlan(p)
	curRes := cost.Evaluate(m, t, cur)
	if !curRes.Feasible {
		return p
	}
	whole := func(i int) bool { return len(cur.Placements[i].Units) == 1 }
	maxRounds := 8
	if len(t.Stages) > 24 {
		maxRounds = 3 // neighbor space grows quadratically; stay bounded
	}
	for round := 0; round < maxRounds; round++ {
		var bestPlan cost.Plan
		var bestRes cost.Result
		improved := false
		consider := func(np cost.Plan) {
			r := cost.Evaluate(m, t, np)
			if r.Feasible && r.TotalTime < curRes.TotalTime*0.999 {
				if !improved || r.TotalTime < bestRes.TotalTime {
					bestPlan, bestRes, improved = np, r, true
				}
			}
		}
		// Moves.
		for i := range t.Stages {
			if !whole(i) {
				continue
			}
			from := cur.Placements[i].Units[0]
			for _, u := range m.Units {
				if u.ID == from || !t.Stages[i].Allows(u) {
					continue
				}
				np := clonePlan(cur)
				np.Placements[i] = cost.Placement{Units: []string{u.ID}, Shares: []float64{1}}
				consider(np)
			}
		}
		// Swaps (skipped for many-stage tasks: the pair space is
		// quadratic and moves/shapes carry most of the gain there).
		for i := range t.Stages {
			if len(t.Stages) > 24 {
				break
			}
			if !whole(i) {
				continue
			}
			for j := i + 1; j < len(t.Stages); j++ {
				if !whole(j) {
					continue
				}
				ui, uj := cur.Placements[i].Units[0], cur.Placements[j].Units[0]
				if ui == uj {
					continue
				}
				ua, _ := m.Unit(ui)
				ub, _ := m.Unit(uj)
				if !t.Stages[i].Allows(ub) || !t.Stages[j].Allows(ua) {
					continue
				}
				np := clonePlan(cur)
				np.Placements[i] = cost.Placement{Units: []string{uj}, Shares: []float64{1}}
				np.Placements[j] = cost.Placement{Units: []string{ui}, Shares: []float64{1}}
				consider(np)
			}
		}
		// Shape changes: collapse a split stage onto each of its
		// units, or split a whole stage across all units. Crossing
		// between shapes is what lets the search escape valleys that
		// pure moves/swaps cannot (a collapse frees exactly the
		// domain another stage needs).
		for i := range t.Stages {
			st := t.Stages[i]
			if whole(i) {
				if st.Splittable {
					if pl, ok := SplitPlacement(m, st, t.TotalFootprint(), m.Units); ok && len(pl.Units) > 1 {
						np := clonePlan(cur)
						np.Placements[i] = pl
						consider(np)
					}
				}
				continue
			}
			for _, uid := range cur.Placements[i].Units {
				u, _ := m.Unit(uid)
				if !st.Allows(u) {
					continue
				}
				np := clonePlan(cur)
				np.Placements[i] = cost.Placement{Units: []string{uid}, Shares: []float64{1}}
				consider(np)
			}
		}
		if !improved {
			break
		}
		cur, curRes = bestPlan, bestRes
	}
	return cur
}
