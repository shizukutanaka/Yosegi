// Package workload models work to be placed on a machine (package
// device) in resource terms only: how much computation, how many bytes
// streamed from memory, how many bytes of storage I/O, how much data
// must stay resident, and how much data flows to the next stage.
//
// A task is an ordered sequence of stages. Stages run one after another
// (a pipeline); within a stage, work may be split across units when the
// stage is splittable. This single shape covers the important families:
//
//   - LLM inference: one stage per layer (decode streams every layer's
//     weights per token; prefill is one compute-bound stage set).
//   - Transcoding / rendering: a few large splittable stages.
//   - Data scans / ETL: memory- or I/O-bound stages.
//   - Builds: many small CPU stages (modelled as grouped stages).
//
// The cost model (package cost) is a roofline over these numbers:
// a stage takes as long as its slowest resource demands.
package workload

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"

	"github.com/shizukutanaka/Yosegi/device"
)

// Stage is one assignable piece of a task.
type Stage struct {
	Name string `json:"name"`

	// Compute is the arithmetic work, in the same abstract units as
	// device.Unit.ComputeRate (FLOPs are the natural choice).
	Compute float64 `json:"compute"`
	// MemBytes is the data streamed from the stage's resident memory
	// domain per execution (weights, working set scans).
	MemBytes int64 `json:"mem_bytes"`
	// IOBytes is storage traffic per execution (reads; writes are
	// charged at the same rate unless a profile says otherwise).
	IOBytes int64 `json:"io_bytes,omitempty"`
	// Footprint is the data that must be resident in the stage's
	// memory tier while the task runs (weights, caches, buffers).
	Footprint int64 `json:"footprint"`
	// ExpertBytes is the part of Footprint (and of MemBytes, scaled by
	// ActiveExpertRatio) that is sparsely used — MoE expert weights are
	// the archetype. It is the part eligible for tier offload: resident
	// in a slower memory domain and either streamed or executed there.
	ExpertBytes       int64   `json:"expert_bytes,omitempty"`
	ActiveExpertRatio float64 `json:"active_expert_ratio,omitempty"` // 0..1 of ExpertBytes streamed per execution

	// Activation is the data handed to the next stage; it is charged
	// as a transfer when consecutive stages live in different memory
	// domains.
	Activation int64 `json:"activation,omitempty"`

	// Splittable allows the stage's work to be shared across units in
	// parallel (data-parallel). SplitSyncCost is the fractional time
	// added per extra participating unit (synchronization, merge).
	Splittable    bool    `json:"splittable,omitempty"`
	SplitSyncCost float64 `json:"split_sync_cost,omitempty"`

	// Cacheable is the fraction of Compute+MemBytes avoidable on a
	// repeat execution when the task's prefix/state is unchanged —
	// prefix caching for LLM prompts is the archetype. Charged via
	// Task.CacheHitRate.
	Cacheable float64 `json:"cacheable,omitempty"`

	// AllowedKinds restricts which unit kinds may execute this stage
	// (a compiler stage cannot run on a GPU; a CUDA-kernel stage
	// cannot run on a CPU). Empty means any kind.
	AllowedKinds []device.UnitKind `json:"allowed_kinds,omitempty"`
}

// Allows reports whether the stage may execute on the unit.
func (s Stage) Allows(u device.Unit) bool {
	if len(s.AllowedKinds) == 0 {
		return true
	}
	for _, k := range s.AllowedKinds {
		if u.Kind == k {
			return true
		}
	}
	return false
}

// StreamedBytes returns the bytes actually streamed per execution:
// dense bytes plus the active fraction of expert bytes.
func (s Stage) StreamedBytes() int64 {
	dense := s.MemBytes - s.ExpertBytes
	if dense < 0 {
		dense = 0
	}
	return dense + int64(float64(s.ExpertBytes)*s.ActiveExpertRatio)
}

// Reduction is a declared way to shrink a task's footprint at some
// quality cost (a lower-precision build of a model, a shorter context,
// a smaller working set). The planner never applies one silently; when
// nothing fits it reports the first reduction that would fit.
type Reduction struct {
	Name           string  `json:"name"`
	FootprintScale float64 `json:"footprint_scale"` // multiplies every stage Footprint
	ComputeScale   float64 `json:"compute_scale,omitempty"`
	Note           string  `json:"note,omitempty"`
}

// SpecCapability declares that a task supports predict-and-verify
// execution (speculative decoding is the LLM instance): a cheap draft
// pass proposes work that the full pass verifies in parallel.
type SpecCapability struct {
	// DraftWorkRatio is the draft pass cost as a fraction of one full
	// pass over the same stage sequence (draft model size / target
	// model size is the usual estimate).
	DraftWorkRatio float64 `json:"draft_work_ratio"`
	// DraftFootprint is the resident bytes of the draft (charged to
	// the draft's placement unit's memory, or the home tier).
	DraftFootprint int64 `json:"draft_footprint"`
	// Acceptance is the measured per-step acceptance probability
	// (0..1). It MUST come from measurement on the real workload;
	// the planner treats speculation as guilty until measured
	// innocent (see docs/ALGORITHM.md principle 3).
	Acceptance float64 `json:"acceptance"`
	// MaxDepth caps the speculation depth the planner may try.
	MaxDepth int `json:"max_depth"`
}

// ExpectedTokensPerCycle is the expected number of accepted steps per
// verify cycle at depth n with per-step acceptance alpha:
// E = (1 - alpha^(n+1)) / (1 - alpha).
func ExpectedStepsPerCycle(alpha float64, n int) float64 {
	if n <= 0 {
		return 1
	}
	if alpha >= 1 {
		return float64(n + 1)
	}
	if alpha <= 0 {
		return 1
	}
	num := 1 - pow(alpha, n+1)
	return num / (1 - alpha)
}

func pow(x float64, n int) float64 {
	r := 1.0
	for i := 0; i < n; i++ {
		r *= x
	}
	return r
}

// SpecSpeedup is the expected speedup of speculation at depth n versus
// plain execution: expected steps per cycle divided by the cycle cost
// (one full pass + n draft passes). Values < 1 mean speculation LOSES;
// the break-even acceptance is where this crosses 1.
func SpecSpeedup(alpha, draftRatio float64, n int) float64 {
	if n <= 0 || draftRatio < 0 {
		return 1
	}
	cycleCost := 1 + float64(n)*draftRatio
	return ExpectedStepsPerCycle(alpha, n) / cycleCost
}

// Task is a unit of work to plan.
type Task struct {
	Name   string  `json:"name"`
	Stages []Stage `json:"stages"`
	// Repeats is how many times the stage sequence executes (tokens
	// generated, frames processed, files scanned). >= 1.
	Repeats int `json:"repeats"`
	// CacheHitRate is the fraction of repeats (after the first) whose
	// cacheable work is actually avoided (measured, workload-specific).
	CacheHitRate float64 `json:"cache_hit_rate,omitempty"`

	Spec       *SpecCapability `json:"spec,omitempty"`
	Reductions []Reduction     `json:"reductions,omitempty"`
}

// TotalFootprint is the resident bytes if all stages share one tier.
func (t Task) TotalFootprint() int64 {
	var sum int64
	for _, s := range t.Stages {
		sum += s.Footprint
	}
	return sum
}

// Signature identifies a task's structure for tuner records: stage
// shapes and repeat count. Same rule as machines — a winner measured
// for one signature is never applied to another.
func (t Task) Signature() string {
	h := sha256.New()
	enc := json.NewEncoder(h)
	_ = enc.Encode(struct {
		Name    string
		Stages  []Stage
		Repeats int
	}{t.Name, t.Stages, t.Repeats})
	return fmt.Sprintf("%x", h.Sum(nil))[:16]
}

// ApplyReduction returns a copy of the task with the reduction applied
// (footprints and optionally compute scaled). Used by the planner to
// test "would it fit if..." without mutating the caller's task.
//
// The result's own Reductions keep only the entries that shrink
// FURTHER than the applied one, rescaled to be relative to the new
// task. Without this, probing reductions recursively would re-apply
// the same ladder forever (each probe's no-fit path probes again).
func (t Task) ApplyReduction(r Reduction) Task {
	out := t
	out.Stages = make([]Stage, len(t.Stages))
	for i, s := range t.Stages {
		s.Footprint = int64(float64(s.Footprint) * r.FootprintScale)
		s.ExpertBytes = int64(float64(s.ExpertBytes) * r.FootprintScale)
		if r.ComputeScale > 0 {
			s.Compute *= r.ComputeScale
		}
		out.Stages[i] = s
	}
	out.Reductions = nil
	for _, other := range t.Reductions {
		if other.Name == r.Name || other.FootprintScale >= r.FootprintScale {
			continue
		}
		nr := other
		nr.FootprintScale = other.FootprintScale / r.FootprintScale
		if other.ComputeScale > 0 && r.ComputeScale > 0 {
			nr.ComputeScale = other.ComputeScale / r.ComputeScale
		}
		out.Reductions = append(out.Reductions, nr)
	}
	return out
}
