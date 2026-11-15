package sim

import "testing"

// Optimality, measured — not asserted. The planner is a heuristic
// (candidate generation + local search), so this test pins the
// EMPIRICAL gap against brute force over the shared search space,
// with thresholds set from measured runs (seed 42: agreement 83.7%,
// mean gap 0.84%, p95 0.61%, max 69.4%; seed 7 agrees closely).
// If a change makes the planner worse beyond these bounds, the test
// fails; if it makes it better, tighten the bounds.
//
// The residual worst cases are adversarial tight-capacity mixes
// whose optimum needs multi-stage coordinated moves the local
// search cannot reach; they are documented in docs/ALGORITHM.md.
func TestPlannerOptimalityGap(t *testing.T) {
	s := Run(42, 200)
	t.Logf("compared=%d agreement=%.3f mean=%.4f p95=%.4f max=%.4f plannerBetter=%d bothNoFit=%d plannerOnly=%d bruteOnly=%d",
		s.Compared, s.AgreementRate, s.MeanGap, s.P95Gap, s.MaxGap,
		s.PlannerBetter, s.BothNoFit, s.PlannerOnly, s.BruteOnly)
	if s.Compared < 100 {
		t.Fatalf("too few comparable cases: %d", s.Compared)
	}
	if s.AgreementRate < 0.80 {
		t.Errorf("agreement rate regressed: %.3f < 0.80", s.AgreementRate)
	}
	if s.MeanGap > 0.02 {
		t.Errorf("mean gap regressed: %.4f > 0.02", s.MeanGap)
	}
	if s.P95Gap > 0.02 {
		t.Errorf("p95 gap regressed: %.4f > 0.02", s.P95Gap)
	}
	if s.MaxGap > 0.75 {
		t.Errorf("max gap regressed: %.4f > 0.75", s.MaxGap)
	}
	if s.BruteOnly > 2 {
		t.Errorf("planner missed feasible plans in %d cases (> 2)", s.BruteOnly)
	}
}
