// Command yosegi is a small demonstration CLI for the Yosegi
// planning algorithms: it plans the built-in example tasks on the
// built-in machine profiles and runs the optimality simulator.
//
//	yosegi plan   — plan example tasks on the Alloy rig and a whole-PC profile
//	yosegi sim    — planner vs brute force over synthetic cases
package main

import (
	"fmt"
	"os"

	"yosegi/cost"
	"yosegi/device"
	"yosegi/planner"
	"yosegi/profiles"
	"yosegi/sim"
	"yosegi/workload"
)

func main() {
	cmd := "plan"
	if len(os.Args) > 1 {
		cmd = os.Args[1]
	}
	switch cmd {
	case "plan":
		runPlan()
	case "sim":
		st := sim.Run(42, 200)
		fmt.Printf("simulator: %d cases, %d compared\n", st.Cases, st.Compared)
		fmt.Printf("  agreement with brute-force optimum: %.1f%%\n", st.AgreementRate*100)
		fmt.Printf("  mean gap %.4f, p95 gap %.4f, max gap %.4f\n", st.MeanGap, st.P95Gap, st.MaxGap)
		fmt.Printf("  planner better (richer shapes): %d, both no-fit: %d, planner-only: %d, brute-only: %d\n",
			st.PlannerBetter, st.BothNoFit, st.PlannerOnly, st.BruteOnly)
	default:
		fmt.Fprintf(os.Stderr, "usage: yosegi [plan|sim]\n")
		os.Exit(2)
	}
}

func show(m *device.Machine, label string, t workload.Task, opts planner.Options) {
	out := planner.Solve(m, t, opts)
	fmt.Printf("== %s on %s\n", t.Name, label)
	if out.NoFit != nil {
		fmt.Printf("   NO FIT:\n")
		for _, r := range out.NoFit.Reasons {
			fmt.Printf("     - %s\n", r)
		}
		for _, s := range out.NoFit.Suggestions {
			fmt.Printf("     suggestion: %s\n", s)
		}
		return
	}
	for i, sp := range out.Top {
		fmt.Printf("   #%d %.4f s total (%.4f s/repeat, %.2f repeats/s) bottleneck=%s complexity=%d\n",
			i+1, sp.Result.TotalTime, sp.Result.PerRepeat, sp.Result.Throughput, sp.Result.Bottleneck, sp.Plan.Complexity())
		fmt.Printf("      plan: %s\n", sp.Plan.ID())
		for _, n := range sp.Result.Notes {
			fmt.Printf("      note: %s\n", n)
		}
		if i == 0 {
			_ = cost.SortedReasons(sp.Result)
		}
	}
}

func runPlan() {
	rig := profiles.AlloyRig()
	show(&rig, "alloy-rig", profiles.LLM14B().DecodeTask(128), planner.Options{WithSpec: true, WithOffload: true})
	show(&rig, "alloy-rig", profiles.LLM32B().DecodeTask(128), planner.Options{WithSpec: true, WithOffload: true})
	moe := profiles.MoE35B().DecodeTask(128)
	show(&rig, "alloy-rig", moe, planner.Options{WithOffload: true})

	pc := profiles.WholePC()
	show(&pc, "whole-pc", profiles.CompileTask(), planner.Options{})
	show(&pc, "whole-pc", profiles.TranscodeTask(), planner.Options{})
	show(&pc, "whole-pc", profiles.ScanTask(), planner.Options{})
}
