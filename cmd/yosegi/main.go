// Command yosegi is a small demonstration CLI for the Yosegi
// planning algorithms: it plans the built-in example tasks on the
// built-in machine profiles and runs the optimality simulator.
//
//	yosegi plan        — plan example tasks on the Alloy rig and a whole-PC profile
//	yosegi plan -machine my-pc.json -task my-task.json [-bench bench.json]
//	                   — plan YOUR task on YOUR machine, from local JSON files
//	yosegi template machine|task — print an editable JSON starting point
//	yosegi sim         — planner vs brute force over synthetic cases
//	yosegi vendorcheck — cross-vendor launch/occupancy gate for example kernel configs
package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/shizukutanaka/Yosegi/cost"
	"github.com/shizukutanaka/Yosegi/device"
	"github.com/shizukutanaka/Yosegi/planner"
	"github.com/shizukutanaka/Yosegi/profiles"
	"github.com/shizukutanaka/Yosegi/sim"
	"github.com/shizukutanaka/Yosegi/vendor"
	"github.com/shizukutanaka/Yosegi/workload"
)

func main() {
	cmd := "plan"
	if len(os.Args) > 1 {
		cmd = os.Args[1]
	}
	switch cmd {
	case "plan":
		if len(os.Args) > 2 {
			if err := runPlanCmd(os.Stdout, os.Args[2:]); err != nil {
				fmt.Fprintln(os.Stderr, "yosegi plan:", err)
				os.Exit(1)
			}
			return
		}
		runPlan()
	case "template":
		kind := ""
		if len(os.Args) > 2 {
			kind = os.Args[2]
		}
		if err := writeTemplate(os.Stdout, kind); err != nil {
			fmt.Fprintln(os.Stderr, "yosegi template:", err)
			os.Exit(2)
		}
	case "sim":
		st := sim.Run(42, 200)
		fmt.Printf("simulator: %d cases, %d compared\n", st.Cases, st.Compared)
		fmt.Printf("  agreement with brute-force optimum: %.1f%%\n", st.AgreementRate*100)
		fmt.Printf("  mean gap %.4f, p95 gap %.4f, max gap %.4f\n", st.MeanGap, st.P95Gap, st.MaxGap)
		fmt.Printf("  planner better (richer shapes): %d, both no-fit: %d, planner-only: %d, brute-only: %d\n",
			st.PlannerBetter, st.BothNoFit, st.PlannerOnly, st.BruteOnly)
	case "vendorcheck":
		runVendorCheck()
	default:
		fmt.Fprintf(os.Stderr, "usage: yosegi [plan [-machine FILE -task FILE [-bench FILE]]|template machine|task|sim|vendorcheck]\n")
		os.Exit(2)
	}
}

func show(m *device.Machine, label string, t workload.Task, opts planner.Options) {
	showTo(os.Stdout, m, label, t, opts)
}

func showTo(w io.Writer, m *device.Machine, label string, t workload.Task, opts planner.Options) {
	out := planner.Solve(m, t, opts)
	fmt.Fprintf(w, "== %s on %s\n", t.Name, label)
	if out.NoFit != nil {
		fmt.Fprintf(w, "   NO FIT:\n")
		for _, r := range out.NoFit.Reasons {
			fmt.Fprintf(w, "     - %s\n", r)
		}
		for _, s := range out.NoFit.Suggestions {
			fmt.Fprintf(w, "     suggestion: %s\n", s)
		}
		return
	}
	for i, sp := range out.Top {
		fmt.Fprintf(w, "   #%d %.4f s total (%.4f s/repeat, %.2f repeats/s) bottleneck=%s complexity=%d\n",
			i+1, sp.Result.TotalTime, sp.Result.PerRepeat, sp.Result.Throughput, sp.Result.Bottleneck, sp.Plan.Complexity())
		fmt.Fprintf(w, "      plan: %s\n", sp.Plan.ID())
		for _, n := range sp.Result.Notes {
			fmt.Fprintf(w, "      note: %s\n", n)
		}
		if i == 0 {
			_ = cost.SortedReasons(sp.Result)
		}
	}
}

// runVendorCheck runs the cross-vendor gate (ported from Tsugi) over
// two example tile configurations: a small one that launches
// everywhere, and a Hopper-tuned one whose 128KB shared-memory
// footprint physically does not fit AMD's 64KiB LDS — the canonical
// "works on NVIDIA, does not even start on AMD" case.
func runVendorCheck() {
	cfgs := []struct {
		label string
		cfg   vendor.KernelConfig
	}{
		{"small (32x32x32, 2 stages, 4 warps)", vendor.KernelConfig{BlockM: 32, BlockN: 32, BlockK: 32, NumStages: 2, NumWarps: 4}},
		{"hopper-tuned (128x128x64, 4 stages, 8 warps)", vendor.KernelConfig{BlockM: 128, BlockN: 128, BlockK: 64, NumStages: 4, NumWarps: 8}},
	}
	for _, c := range cfgs {
		fmt.Printf("== kernel %s\n", c.label)
		for _, v := range vendor.All {
			rep, err := vendor.Check(c.cfg, v, vendor.Options{})
			if err != nil {
				fmt.Println("   error:", err)
				continue
			}
			status := "LAUNCHABLE"
			if !rep.Launchable() {
				status = "NOT-LAUNCHABLE"
			}
			fmt.Printf("   %-9s %s", v, status)
			if rep.Launchable() {
				occ, err := vendor.Estimate(c.cfg, v, vendor.Options{})
				if err == nil {
					fmt.Printf("  occupancy=%.0f%% (limited by %s)", occ.Occupancy*100, occ.LimitedBy)
				}
			}
			fmt.Println()
			for _, b := range rep.Blockers() {
				fmt.Printf("      blocker: %s required=%d limit=%d\n", b.Resource, b.Required, b.Limit)
			}
		}
		only, err := vendor.FirstVendorOnly(c.cfg, vendor.NVIDIA, vendor.AMDCDNA, vendor.Options{})
		if err == nil {
			for _, s := range only {
				fmt.Printf("   asymmetry: %s\n", s)
			}
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

// runPlanCmd plans a user-described task on a user-described machine.
// Everything is read from local files the user owns; nothing leaves
// the machine. Optional -bench folds measured microbenchmark results
// into the machine before planning (measurement beats estimates).
func runPlanCmd(w io.Writer, args []string) error {
	fs := flag.NewFlagSet("plan", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	machinePath := fs.String("machine", "", "machine JSON (start from: yosegi template machine)")
	taskPath := fs.String("task", "", "task JSON (start from: yosegi template task)")
	benchPath := fs.String("bench", "", "optional microbench results JSON (array of records)")
	top := fs.Int("top", 3, "plans to show")
	spec := fs.Bool("spec", false, "consider speculative variants")
	offload := fs.Bool("offload", false, "consider tier-offload variants")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *machinePath == "" || *taskPath == "" {
		return errors.New("both -machine and -task are required (see: yosegi template machine|task)")
	}
	var m device.Machine
	if err := readJSON(*machinePath, &m); err != nil {
		return err
	}
	if *benchPath != "" {
		var bench []device.Microbench
		if err := readJSON(*benchPath, &bench); err != nil {
			return err
		}
		for _, r := range bench {
			if _, ok := m.Unit(r.UnitID); !ok {
				return fmt.Errorf("%s: unknown unit_id %q", *benchPath, r.UnitID)
			}
		}
		m = m.ApplyMicrobench(bench)
	}
	if problems := m.Validate(); len(problems) > 0 {
		return fmt.Errorf("%s: invalid machine: %s", *machinePath, strings.Join(problems, "; "))
	}
	var t workload.Task
	if err := readJSON(*taskPath, &t); err != nil {
		return err
	}
	if len(t.Stages) == 0 || t.Repeats < 1 {
		return fmt.Errorf("%s: task needs at least one stage and repeats >= 1", *taskPath)
	}
	label := m.Name
	if label == "" {
		label = *machinePath
	}
	showTo(w, &m, label, t, planner.Options{TopK: *top, WithSpec: *spec, WithOffload: *offload})
	return nil
}

// readJSON decodes a file strictly: a misspelled field is an error,
// not a silently ignored measurement.
func readJSON(path string, v any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return nil
}

func writeTemplate(w io.Writer, kind string) error {
	var v any
	switch kind {
	case "machine":
		v = profiles.WholePC()
	case "task":
		v = profiles.TranscodeTask()
	default:
		return errors.New("usage: yosegi template machine|task")
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}
