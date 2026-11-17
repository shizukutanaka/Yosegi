// Package tuner is the online half of Yosegi: the framework that
// measures the planner's top candidates on the real machine and keeps
// the measured winner.
//
// Rules (carried over from Alloy's auto-tuner, which learned them
// from the heterogeneous-GPU evidence base):
//
//   - Estimates rank candidates; only MEASUREMENTS crown a winner.
//   - The winner is persisted with the signatures of the machine and
//     the task it was measured on. If either signature changes —
//     different hardware, different drivers changing measured rates,
//     a different model/task shape — the record does not apply and
//     planning falls back to estimates until re-measured.
//   - Ties (within 1%) prefer the simpler plan: fewer split units,
//     no speculation, no offload.
//   - A failed candidate never aborts a tuning run; it is recorded
//     as failed and loses.
package tuner

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"time"

	"github.com/shizukutanaka/Yosegi/cost"
	"github.com/shizukutanaka/Yosegi/device"
	"github.com/shizukutanaka/Yosegi/workload"
)

// Measurement is one candidate's measured outcome.
type Measurement struct {
	PlanID     string   `json:"plan_id"`
	Seconds    float64  `json:"seconds,omitempty"`    // measured wall time for the run
	Throughput float64  `json:"throughput,omitempty"` // measured repeats/sec
	Acceptance *float64 `json:"acceptance,omitempty"` // measured spec acceptance, when reported
	State      string   `json:"state"`                // done | failed | skipped
	Note       string   `json:"note,omitempty"`
}

// Candidate pairs a plan with its measurement (if any).
type Candidate struct {
	Plan        cost.Plan
	Measurement *Measurement
}

// PickWinner returns the index of the best MEASURED candidate:
// highest throughput (or lowest seconds when throughput is absent).
// Ties within 1% prefer the lower-complexity plan. -1 when nothing
// was measured successfully.
func PickWinner(cands []Candidate) int {
	best := -1
	score := func(c Candidate) float64 {
		if c.Measurement == nil {
			return 0
		}
		if c.Measurement.Throughput > 0 {
			return c.Measurement.Throughput
		}
		if c.Measurement.Seconds > 0 {
			return 1 / c.Measurement.Seconds
		}
		return 0
	}
	for i, c := range cands {
		if c.Measurement == nil || c.Measurement.State != "done" {
			continue
		}
		s := score(c)
		if s <= 0 {
			continue
		}
		if best < 0 {
			best = i
			continue
		}
		bs := score(cands[best])
		switch {
		case s > bs*1.01:
			best = i
		case bs > s*1.01:
			// current best clearly faster: keep
		case c.Plan.Complexity() < cands[best].Plan.Complexity():
			best = i
		}
	}
	return best
}

// Record is one persisted tuning result.
type Record struct {
	MachineSignature string        `json:"machine_signature"`
	TaskSignature    string        `json:"task_signature"`
	TaskName         string        `json:"task_name"`
	WinnerID         string        `json:"winner_id"`
	Winner           cost.Plan     `json:"winner"`
	Measurements     []Measurement `json:"measurements"`
	TunedAt          string        `json:"tuned_at"` // RFC3339
}

// Store is a JSON file of tuning records keyed by task signature.
type Store struct {
	Path    string
	Records map[string]Record `json:"records"`
}

// LoadStore reads a store file; a missing file yields an empty store.
func LoadStore(path string) (*Store, error) {
	s := &Store{Path: path, Records: map[string]Record{}}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return s, nil
		}
		return nil, err
	}
	if len(data) == 0 {
		return s, nil
	}
	if err := json.Unmarshal(data, s); err != nil {
		return nil, fmt.Errorf("parse tuner store: %w", err)
	}
	if s.Records == nil {
		s.Records = map[string]Record{}
	}
	s.Path = path
	return s, nil
}

// Save persists the store atomically (write-then-rename).
func (s *Store) Save() error {
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	tmp := s.Path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, s.Path)
}

// Commit records a finished tuning run: it picks the winner from the
// measurements and stores the record under the task's signature.
// Returns the winning candidate index, or -1 (nothing stored) when
// no candidate measured successfully.
func (s *Store) Commit(m *device.Machine, t workload.Task, cands []Candidate) int {
	wi := PickWinner(cands)
	if wi < 0 {
		return -1
	}
	rec := Record{
		MachineSignature: m.Signature(),
		TaskSignature:    t.Signature(),
		TaskName:         t.Name,
		WinnerID:         cands[wi].Plan.ID(),
		Winner:           cands[wi].Plan,
		TunedAt:          time.Now().Format(time.RFC3339),
	}
	for _, c := range cands {
		if c.Measurement != nil {
			rec.Measurements = append(rec.Measurements, *c.Measurement)
		}
	}
	s.Records[t.Signature()] = rec
	return wi
}

// Lookup returns the tuned winner for this exact machine+task pair.
// The second return value is false when there is no record, or when
// the record's machine signature does not match the current machine —
// the caller must then fall back to the planner's estimates. This is
// the "configuration changed → back to estimates" rule.
func (s *Store) Lookup(m *device.Machine, t workload.Task) (Record, bool) {
	rec, ok := s.Records[t.Signature()]
	if !ok {
		return Record{}, false
	}
	if rec.MachineSignature != m.Signature() {
		return Record{}, false
	}
	return rec, true
}

// Runner is the measurement harness contract: the host application
// (e.g. Alloy) implements Measure by actually executing a plan and
// timing it. The tuner owns selection and persistence; it never
// executes anything itself.
type Runner interface {
	Measure(m *device.Machine, t workload.Task, p cost.Plan) Measurement
}

// Tune runs a full tuning pass: measure every candidate (failures are
// recorded, never fatal), commit the winner. Returns the committed
// record and the winner index (-1 when nothing measured).
func Tune(s *Store, r Runner, m *device.Machine, t workload.Task, plans []cost.Plan) (Record, int) {
	cands := make([]Candidate, len(plans))
	for i, p := range plans {
		meas := r.Measure(m, t, p)
		meas.PlanID = p.ID()
		mm := meas
		cands[i] = Candidate{Plan: p, Measurement: &mm}
	}
	wi := s.Commit(m, t, cands)
	if wi < 0 {
		return Record{}, -1
	}
	rec, _ := s.Lookup(m, t)
	return rec, wi
}

// SortedMeasurements returns measurements sorted by plan ID, for
// stable display.
func SortedMeasurements(ms []Measurement) []Measurement {
	out := append([]Measurement(nil), ms...)
	sort.Slice(out, func(i, j int) bool { return out[i].PlanID < out[j].PlanID })
	return out
}
