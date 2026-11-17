// Package cost is the measured-cost model at the heart of Yosegi: a
// roofline evaluation of a placement plan against a machine profile.
//
// For each stage, time is the maximum of its resource demands —
// compute time on the assigned units, memory-streaming time from the
// resident tier, storage I/O time — plus inter-stage transfer time
// when consecutive stages live in different memory domains, plus a
// synchronization surcharge for split stages. Feasibility is checked
// against memory capacities and the machine power cap; thermal soak
// derates long runs from peak to sustained rates.
//
// The model is deliberately first-order. Its job is to RANK plans and
// to be wrong in consistent ways; the tuner (package tuner) replaces
// its predictions with measurements on real hardware, and the
// simulator (package sim) measures how far its rankings sit from
// exhaustive search within the same model.
package cost

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/shizukutanaka/Yosegi/device"
	"github.com/shizukutanaka/Yosegi/workload"
)

// Placement assigns one stage to one or more units.
type Placement struct {
	// Units and Shares: one unit with share 1 is a whole-stage
	// assignment; several units split the stage's work in parallel
	// (shares sum to 1).
	Units  []string  `json:"units"`
	Shares []float64 `json:"shares"`
	// Tier is the memory domain holding the stage's dense data.
	// "" means each unit's home domain (sharded by share).
	Tier string `json:"tier,omitempty"`
	// ExpertTier is the domain holding the stage's expert bytes
	// ("" = same as Tier). Setting it to RAM while the stage executes
	// on a GPU is the generalized form of MoE CPU offload.
	ExpertTier string `json:"expert_tier,omitempty"`
}

// SpecChoice enables predict-and-verify execution at a depth, with
// the draft pass running on DraftUnit.
type SpecChoice struct {
	Depth     int    `json:"depth"`
	DraftUnit string `json:"draft_unit"`
}

// Plan is a full placement for a task: one placement per stage.
type Plan struct {
	Placements []Placement `json:"placements"`
	Spec       *SpecChoice `json:"spec,omitempty"`
}

// ID is a canonical string form of the plan, used for tuner records
// and deduplication.
func (p Plan) ID() string {
	var b strings.Builder
	for i, pl := range p.Placements {
		if i > 0 {
			b.WriteByte(';')
		}
		for j, u := range pl.Units {
			if j > 0 {
				b.WriteByte('+')
			}
			fmt.Fprintf(&b, "%s:%.3f", u, shareAt(pl, j))
		}
		if pl.Tier != "" {
			b.WriteString("@")
			b.WriteString(pl.Tier)
		}
		if pl.ExpertTier != "" {
			b.WriteString("#")
			b.WriteString(pl.ExpertTier)
		}
	}
	if p.Spec != nil {
		fmt.Fprintf(&b, "|spec:%s:%d", p.Spec.DraftUnit, p.Spec.Depth)
	}
	return b.String()
}

// Complexity ranks plans for tie-breaking: simpler wins. Splits,
// off-home tiers and speculation all cost rank (the Alloy tuner's
// rule: on a tie, prefer the plan with fewer moving parts).
func (p Plan) Complexity() int {
	c := 0
	for _, pl := range p.Placements {
		c += len(pl.Units) - 1
		if pl.Tier != "" {
			c++
		}
		if pl.ExpertTier != "" {
			c++
		}
	}
	if p.Spec != nil {
		c += 2
	}
	return c
}

func shareAt(pl Placement, j int) float64 {
	if j < len(pl.Shares) {
		return pl.Shares[j]
	}
	if len(pl.Units) == 1 {
		return 1
	}
	return 0
}

// StageBreakdown is the per-stage accounting of one pass.
type StageBreakdown struct {
	ComputeTime  float64
	MemTime      float64
	IOTime       float64
	TransferTime float64
	Total        float64
	Bottleneck   string // compute | memory | io | transfer
}

// Result is the outcome of evaluating one plan.
type Result struct {
	Feasible   bool
	Reasons    []string // why not, when infeasible
	Notes      []string // deratings and assumptions applied
	TotalTime  float64  // seconds for all repeats
	PerRepeat  float64  // seconds per execution of the stage sequence
	Throughput float64  // repeats per second
	EnergyJ    float64
	PeakPowerW float64
	Stages     []StageBreakdown
	Bottleneck string // dominant bottleneck across the plan
}

// homeDomain resolves where a unit's share of a stage's data sits.
func homeDomain(m *device.Machine, pl Placement, unitID string) string {
	if pl.Tier != "" {
		return pl.Tier
	}
	u, _ := m.Unit(unitID)
	return u.MemoryID
}

func expertDomain(m *device.Machine, pl Placement, unitID string) string {
	if pl.ExpertTier != "" {
		return pl.ExpertTier
	}
	return homeDomain(m, pl, unitID)
}

// planDomain is where a placement's output is considered to live
// for inter-stage transfers: the tier if set, else the home domain
// of the largest share. The planner's DP uses the same definition,
// so generator metrics and evaluated costs agree.
func planDomain(m *device.Machine, pl Placement) string {
	if pl.Tier != "" {
		return pl.Tier
	}
	best, bestShare := "", -1.0
	for j, uid := range pl.Units {
		if s := shareAt(pl, j); s > bestShare {
			bestShare = s
			u, _ := m.Unit(uid)
			best = u.MemoryID
		}
	}
	return best
}

// Evaluate predicts a plan's cost on a machine. Pure: no I/O.
func Evaluate(m *device.Machine, t workload.Task, p Plan) Result {
	// Two passes: the first at peak rates estimates the horizon; the
	// second derates units whose thermal time constant is exceeded.
	r := evaluate(m, t, p, 0)
	if r.TotalTime > 0 {
		r2 := evaluate(m, t, p, r.TotalTime)
		// Keep the derated estimate (it is the honest one for planning
		// sustained work); feasibility is identical either way.
		r2.Reasons = r.Reasons
		return r2
	}
	return r
}

func evaluate(m *device.Machine, t workload.Task, p Plan, horizon float64) Result {
	var res Result
	if len(p.Placements) != len(t.Stages) {
		res.Reasons = append(res.Reasons, fmt.Sprintf("plan has %d placements for %d stages", len(p.Placements), len(t.Stages)))
		return res
	}
	// --- feasibility: resident footprints per memory domain ---
	used := map[string]int64{}
	charge := func(domain string, bytes int64) {
		if bytes > 0 {
			used[domain] += bytes
		}
	}
	for i, pl := range p.Placements {
		st := t.Stages[i]
		for _, uid := range pl.Units {
			if u, ok := m.Unit(uid); ok && !st.Allows(u) {
				res.Reasons = append(res.Reasons, fmt.Sprintf(
					"stage %s cannot run on unit %s (kind %s)", st.Name, uid, u.Kind))
			}
		}
		dense := st.Footprint - st.ExpertBytes
		if dense < 0 {
			dense = 0
		}
		for j, uid := range pl.Units {
			sh := shareAt(pl, j)
			charge(homeDomain(m, pl, uid), int64(float64(dense)*sh))
			charge(expertDomain(m, pl, uid), int64(float64(st.ExpertBytes)*sh))
		}
	}
	if p.Spec != nil && t.Spec != nil {
		if u, ok := m.Unit(p.Spec.DraftUnit); ok {
			charge(u.MemoryID, t.Spec.DraftFootprint)
		} else {
			res.Reasons = append(res.Reasons, "spec draft unit "+p.Spec.DraftUnit+" not in machine")
		}
	}
	for domain, bytes := range used {
		mem, ok := m.Memory(domain)
		if !ok {
			res.Reasons = append(res.Reasons, "unknown memory domain "+domain)
			continue
		}
		if mem.CapacityBytes > 0 && bytes > mem.CapacityBytes {
			res.Reasons = append(res.Reasons, fmt.Sprintf(
				"memory domain %s over capacity: needs %.2f GiB, has %.2f GiB",
				domain, gib(bytes), gib(mem.CapacityBytes)))
		}
	}

	storageBW := m.FastestStorageRead()
	if storageBW <= 0 {
		storageBW = 5e8 // unmodelled storage: conservative SATA-class fallback
	}

	// --- time, first pass and repeat pass (cache-discounted) ---
	first := evalPass(m, t, p, horizon, false, storageBW, &res)
	repeat := first
	if t.Repeats > 1 {
		repeat = evalPass(m, t, p, horizon, true, storageBW, nil)
	}
	perRepeat := repeat.total
	total := first.total + float64(t.Repeats-1)*repeat.total

	// --- speculation: replaces the repeat-pass time with the
	// predict-and-verify cycle time ---
	if p.Spec != nil && t.Spec != nil && t.Repeats > 1 {
		if draft := draftPassTime(m, t, p, horizon); draft > 0 {
			steps := workload.ExpectedStepsPerCycle(t.Spec.Acceptance, p.Spec.Depth)
			cycle := repeat.total + float64(p.Spec.Depth)*draft
			specRepeat := cycle / steps
			total = first.total + float64(t.Repeats-1)*specRepeat
			perRepeat = specRepeat
			if specRepeat > repeat.total {
				res.Notes = append(res.Notes, fmt.Sprintf(
					"speculation at depth %d is predicted to LOSE (acceptance %.2f): %.3f s vs %.3f s per repeat",
					p.Spec.Depth, t.Spec.Acceptance, specRepeat, repeat.total))
			}
		}
	}

	res.TotalTime = total
	res.PerRepeat = perRepeat
	if total > 0 {
		res.Throughput = float64(t.Repeats) / total
	}
	res.Stages = first.stages
	res.EnergyJ = first.energy*1 + repeat.energy*float64(maxInt(0, t.Repeats-1))
	if p.Spec != nil && t.Spec != nil && t.Repeats > 1 {
		// Energy under speculation is not modelled in detail; keep the
		// non-spec energy as a lower bound and say so.
		res.Notes = append(res.Notes, "energy under speculation is a lower bound (draft passes not charged)")
	}
	// Bottleneck: the dominant component across the first pass.
	tot := map[string]float64{}
	for _, sb := range first.stages {
		tot["compute"] += sb.ComputeTime
		tot["memory"] += sb.MemTime
		tot["io"] += sb.IOTime
		tot["transfer"] += sb.TransferTime
	}
	best, bestV := "", -1.0
	for k, v := range tot {
		if v > bestV {
			best, bestV = k, v
		}
	}
	res.Bottleneck = best
	res.PeakPowerW = first.peakPower

	// --- power cap: throttle rather than forbid (a real chassis
	// throttles); the derating is reported, never silent ---
	if m.PowerCapW > 0 && res.PeakPowerW > m.PowerCapW && res.TotalTime > 0 {
		factor := res.PeakPowerW / m.PowerCapW
		res.TotalTime *= factor
		res.PerRepeat *= factor
		res.Throughput = float64(t.Repeats) / res.TotalTime
		res.Notes = append(res.Notes, fmt.Sprintf(
			"peak power %.0f W exceeds machine cap %.0f W: times derated x%.2f (throttling model)",
			res.PeakPowerW, m.PowerCapW, factor))
	}

	res.Feasible = len(res.Reasons) == 0
	return res
}

type passResult struct {
	total     float64
	stages    []StageBreakdown
	energy    float64
	peakPower float64
}

// evalPass times one execution of the stage sequence. When cached is
// true, cacheable work is discounted by the task's hit rate.
func evalPass(m *device.Machine, t workload.Task, p Plan, horizon float64, cached bool, storageBW float64, res *Result) passResult {
	var out passResult
	out.stages = make([]StageBreakdown, len(t.Stages))
	prevDomain := ""
	for i, st := range t.Stages {
		pl := p.Placements[i]
		discount := 1.0
		if cached {
			discount = 1 - st.Cacheable*t.CacheHitRate
		}
		compute := st.Compute * discount
		streamed := float64(st.StreamedBytes()) * discount
		denseBytes := streamed * denseFrac(st)
		expertStreamed := streamed - denseBytes

		// Group participating units by the domain they stream dense
		// data from: a domain's bandwidth is shared by concurrent
		// streamers, so the group's time is set by its total bytes.
		groupBytes := map[string]float64{}
		for j, uid := range pl.Units {
			groupBytes[homeDomain(m, pl, uid)] += shareAt(pl, j) * denseBytes
		}
		groupTime := map[string]float64{}
		for dom, b := range groupBytes {
			mem, _ := m.Memory(dom)
			groupTime[dom] = b / mem.BandwidthBytesSec
		}
		// Expert bytes stream from the expert tier over the link into
		// the executing unit's domain (or locally if same domain).
		expertGroupBytes := map[string]float64{}
		for j, uid := range pl.Units {
			expertGroupBytes[expertDomain(m, pl, uid)] += shareAt(pl, j) * expertStreamed
		}
		expertTime := 0.0
		for dom, b := range expertGroupBytes {
			if b <= 0 {
				continue
			}
			// Charged at the slower of the tier's own bandwidth and
			// the link into the first unit's home domain.
			bw := math.Inf(1)
			if mem, ok := m.Memory(dom); ok {
				bw = mem.BandwidthBytesSec
			}
			if len(pl.Units) > 0 {
				home := homeDomain(m, pl, pl.Units[0])
				if dom != home {
					lbw, _ := m.Transfer(dom, home)
					if lbw < bw {
						bw = lbw
					}
				}
			}
			if tt := b / bw; tt > expertTime {
				expertTime = tt
			}
		}

		var sb StageBreakdown
		stagePower := 0.0
		for j, uid := range pl.Units {
			u, ok := m.Unit(uid)
			if !ok {
				if res != nil {
					res.Reasons = append(res.Reasons, "unknown unit "+uid)
				}
				continue
			}
			sh := shareAt(pl, j)
			ct := sh * compute / u.EffectiveRate(horizon)
			mt := groupTime[homeDomain(m, pl, uid)]
			if expertTime > mt {
				mt = expertTime
			}
			unitTime := math.Max(ct, mt)
			if ct >= mt && ct > sb.ComputeTime {
				sb.ComputeTime = ct
			}
			if mt > sb.MemTime {
				sb.MemTime = mt
			}
			if unitTime > sb.Total {
				sb.Total = unitTime
			}
			stagePower += u.PowerActiveW
		}
		if len(pl.Units) > 1 && st.SplitSyncCost > 0 {
			sb.Total *= 1 + st.SplitSyncCost*float64(len(pl.Units)-1)
		}
		if st.IOBytes > 0 {
			sb.IOTime = float64(st.IOBytes) / storageBW
			if sb.IOTime > sb.Total {
				sb.Total = sb.IOTime
			}
		}
		// Transfer from the previous stage's home domain.
		curDomain := planDomain(m, pl)
		if i > 0 && prevDomain != "" && curDomain != "" && prevDomain != curDomain {
			prev := t.Stages[i-1]
			if prev.Activation > 0 {
				bw, lat := m.Transfer(prevDomain, curDomain)
				sb.TransferTime = float64(prev.Activation)/bw + lat
				sb.Total += sb.TransferTime
			}
		}
		prevDomain = curDomain

		switch {
		case sb.TransferTime >= sb.ComputeTime && sb.TransferTime >= sb.MemTime && sb.TransferTime >= sb.IOTime && sb.TransferTime > 0:
			sb.Bottleneck = "transfer"
		case sb.IOTime >= sb.ComputeTime && sb.IOTime >= sb.MemTime && sb.IOTime > 0:
			sb.Bottleneck = "io"
		case sb.ComputeTime >= sb.MemTime:
			sb.Bottleneck = "compute"
		default:
			sb.Bottleneck = "memory"
		}
		out.stages[i] = sb
		out.total += sb.Total
		out.energy += stagePower * sb.Total
		if stagePower > out.peakPower {
			out.peakPower = stagePower
		}
	}
	return out
}

// denseFrac is the dense share of a stage's streamed bytes.
func denseFrac(st workload.Stage) float64 {
	if st.MemBytes <= 0 {
		return 1
	}
	dense := st.MemBytes - st.ExpertBytes
	if dense < 0 {
		dense = 0
	}
	return float64(dense) / float64(st.MemBytes)
}

// draftPassTime times one draft pass: the stage sequence scaled by the
// draft work ratio, executed wholly on the draft unit.
func draftPassTime(m *device.Machine, t workload.Task, p Plan, horizon float64) float64 {
	u, ok := m.Unit(p.Spec.DraftUnit)
	if !ok {
		return 0
	}
	mem, _ := m.Memory(u.MemoryID)
	total := 0.0
	for _, st := range t.Stages {
		ct := st.Compute * t.Spec.DraftWorkRatio / u.EffectiveRate(horizon)
		mt := float64(st.StreamedBytes()) * t.Spec.DraftWorkRatio / mem.BandwidthBytesSec
		total += math.Max(ct, mt)
	}
	return total
}

// Instance pairs a task with the plan chosen for it, for concurrent
// (co-scheduled) evaluation.
type Instance struct {
	Task workload.Task
	Plan Plan
}

// ConcurrentResult is the interference-adjusted evaluation of several
// tasks running at the same time.
type ConcurrentResult struct {
	Adjusted []Result
	Makespan float64
	Notes    []string
}

// EvaluateConcurrent models task interference, first-order: tasks
// assumed fully overlapped share each memory domain's bandwidth in
// proportion to their streaming demand, and share the machine power
// cap (uniform derating when the cap binds). The makespan is the
// slowest adjusted task.
func EvaluateConcurrent(m *device.Machine, instances []Instance) ConcurrentResult {
	var out ConcurrentResult
	base := make([]Result, len(instances))
	for i, in := range instances {
		base[i] = Evaluate(m, in.Task, in.Plan)
		out.Adjusted = append(out.Adjusted, base[i])
	}
	// Per-domain streaming demand (bytes/sec) of each task.
	demand := make([]map[string]float64, len(instances))
	totalDemand := map[string]float64{}
	for i, in := range instances {
		demand[i] = map[string]float64{}
		if base[i].TotalTime <= 0 {
			continue
		}
		for j, st := range in.Task.Stages {
			pl := in.Plan.Placements[j]
			if len(pl.Units) == 0 {
				continue
			}
			dom := planDomain(m, pl)
			d := float64(st.StreamedBytes()) * float64(in.Task.Repeats) / base[i].TotalTime
			demand[i][dom] += d
			totalDemand[dom] += d
		}
	}
	for i := range instances {
		if !base[i].Feasible || base[i].TotalTime <= 0 {
			continue
		}
		slow := 1.0
		for dom, d := range demand[i] {
			mem, ok := m.Memory(dom)
			if !ok || totalDemand[dom] <= mem.BandwidthBytesSec {
				continue
			}
			// This task's memory-bound share of time stretches by the
			// oversubscription of its domain.
			memShare := memFraction(base[i])
			factor := 1 + memShare*(totalDemand[dom]/mem.BandwidthBytesSec-1)*(d/totalDemand[dom]*float64(len(instances)))
			if factor > slow {
				slow = factor
			}
		}
		if slow > 1 {
			out.Adjusted[i].TotalTime = base[i].TotalTime * slow
			out.Adjusted[i].Throughput = float64(instances[i].Task.Repeats) / out.Adjusted[i].TotalTime
			out.Notes = append(out.Notes, fmt.Sprintf("task %s slowed x%.2f by shared memory bandwidth", instances[i].Task.Name, slow))
		}
	}
	// Shared power cap across co-running tasks.
	if m.PowerCapW > 0 {
		sum := 0.0
		for _, r := range base {
			sum += r.PeakPowerW
		}
		if sum > m.PowerCapW {
			f := sum / m.PowerCapW
			for i := range out.Adjusted {
				out.Adjusted[i].TotalTime *= f
				if out.Adjusted[i].TotalTime > 0 {
					out.Adjusted[i].Throughput = float64(instances[i].Task.Repeats) / out.Adjusted[i].TotalTime
				}
			}
			out.Notes = append(out.Notes, fmt.Sprintf("combined power %.0f W exceeds cap %.0f W: all tasks derated x%.2f", sum, m.PowerCapW, f))
		}
	}
	for _, r := range out.Adjusted {
		if r.TotalTime > out.Makespan {
			out.Makespan = r.TotalTime
		}
	}
	return out
}

// memFraction estimates the memory-bound fraction of a result's time.
func memFraction(r Result) float64 {
	var memT, tot float64
	for _, sb := range r.Stages {
		memT += sb.MemTime
		tot += sb.Total
	}
	if tot <= 0 {
		return 0
	}
	return memT / tot
}

func gib(b int64) float64 { return float64(b) / (1 << 30) }

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// SortedReasons returns feasibility reasons sorted, for stable output.
func SortedReasons(r Result) []string {
	out := append([]string(nil), r.Reasons...)
	sort.Strings(out)
	return out
}
