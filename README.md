# Yosegi

**Measured planning for a whole PC.** Yosegi is a small Go library
(standard library only) that decides *where work should run* on a
heterogeneous machine — performance/efficiency CPU cores, several
dissimilar GPUs, the RAM/VRAM hierarchy, and storage — and *in what
proportions*, using measured device numbers instead of guesses.

It grew out of the [Alloy](https://github.com/shizukutanaka) investigation
into running LLMs across three dissimilar GPUs (RTX A2000 + RX 6400 +
Arc A310), but the model is deliberately general: a workload is a
sequence of stages, each with compute, memory-traffic, I/O and
residency demands, and LLM inference is one workload family among
others (transcoding, builds, dataset scans are included as profiles).

## Design principles (measured, not negotiable)

These come from the Alloy research and are pinned by tests:

1. **Splitting is a capacity tool, not a speed tool.** If work fits
   on the fastest device, the fastest device alone wins. Splits are
   considered when capacity demands them, or when they measurably win.
2. **Decide by measurement, not estimation.** Device inputs are
   measured rates (microbench import format included). Estimated
   plans only rank candidates; the online tuner measures the top
   candidates and persists the winner under the machine+task
   signature. Change the hardware and the signature changes —
   planning falls back to estimates and re-tunes.
3. **Speculative execution is gated by measurement.** Draft/spec
   speedups are modeled with an acceptance probability; at low
   acceptance the model predicts a *loss* and the planner says so.
4. **Prefix/state caching is the cheapest large win.** Cacheable
   work and measured hit rates are first-class inputs to plan cost.
5. **Split ratios follow measured throughput**, never raw memory
   size, and are capped by what each memory domain can hold.

Around those: thermal reality (peak vs sustained rates over the
job's horizon; a power cap throttles rather than forbids), and
interference (co-running tasks share memory bandwidth and the power
cap) are part of the cost model, not afterthoughts.

## Packages

| Package | What it does |
|---|---|
| `device` | Machine model: units (CPU-P/E cores, GPUs), memory domains, storage, links, power cap; microbench import; configuration signature. |
| `workload` | Task model: stages with compute/memory/I/O/footprint, split & cache metadata, speculative-execution capability, reductions (e.g. quantization ladder); the LLM family (decode/prefill tasks from a model spec). |
| `cost` | The cost model: predicts time/energy per plan, feasibility with named reasons, bottleneck classification, thermal derating, concurrent-task interference. |
| `planner` | Candidate generation (single, DP assignment, greedy, throughput-proportional splits, MoE offload, speculative variants), capacity repair + local search, ranking with a simplicity tie-break, honest no-fit reports with *tested* reduction suggestions. |
| `tuner` | Online tuning: measured winners, persistence keyed by machine+task signatures, fallback to estimates on any signature change. |
| `vendor` | Cross-vendor kernel gate (ported from Tsugi): per-vendor hardware facts (warp/wavefront, shared-memory/LDS hard limits, register files, matrix-core shapes), launch feasibility, and occupancy estimation for a tile-kernel configuration — the "will it even start on the other vendor" check upstream of planning. |
| `sim` | Verification harness: random small cases, brute-force optimum over a defined space, gap statistics. |
| `profiles` | Reference machines/tasks, including the Alloy rig (A2000/RX 6400/A310 with the investigation's measured bandwidths). |
| `cmd/yosegi` | Demo CLI: `plan` prints planner decisions for the bundled profiles; `sim` prints optimality statistics. |

## The `vendor` package (from Tsugi)

The `vendor` package is Tsugi's portability verification layer,
integrated here in 2026-11 because its question — *given a kernel
configuration, what happens on the other vendor's GPU?* — is the
upstream gate Yosegi's vendor-neutral device model was missing: the
planner could say a task fits a GPU's memory, but not whether a
kernel tuned for one vendor would even launch on another (Hopper's
227KB/block shared memory vs CDNA's 64KiB LDS is the canonical
break), nor how occupancy differs when it does.

Ported: the vendor hardware tables (primary-source constants), the
launch-feasibility gate, the occupancy estimator, and the
cross-vendor asymmetry/gap helpers, with Tsugi's own test cases.
Not ported, deliberately: Tsugi's numerical-equivalence layers
(they compare two vendors' executed results — a runtime QA concern,
not placement planning) and its tile DSL / compiler (Yosegi plans
work; it does not compile kernels). Tsugi's repository remains the
reference for those.

## Optimality — measured, with the limits stated

The planner is a heuristic (candidate generation + local search),
not an exact solver. Over 200 random small cases (seed 42) against
brute force in the shared search space:

- exact agreement with the brute-force optimum: **83.7%** of
  comparable cases (seed 7: 81.9%)
- mean time gap when not exact: **0.8%**; p95 gap: **0.6%**
- worst observed gap: **~70%**, in adversarial tight-capacity mixes
  whose optimum needs coordinated multi-stage moves the local
  search cannot reach (documented in `docs/ALGORITHM.md`)

These numbers are pinned by `sim/sim_test.go`; a regression fails
the test suite.

## Usage

```sh
go run ./cmd/yosegi plan        # planner decisions on the bundled profiles
go run ./cmd/yosegi sim         # optimality statistics vs brute force
go run ./cmd/yosegi vendorcheck # cross-vendor launch/occupancy gate (from Tsugi)
go test ./...                    # full test suite
```

As a library:

```go
m := profiles.AlloyRig()
task := profiles.LLM14B().DecodeTask(128)
out := planner.Solve(&m, task, planner.Options{TopK: 3})
// out.Top[0] is the best predicted plan; out.NoFit explains, with
// tested suggestions, when nothing fits.
```

## Honest limitations

- **No real-GPU validation yet.** The cost model's constants come
  from the Alloy investigation's published/measured figures; the
  planner has not yet been run against live hardware in this repo.
  The tuner exists precisely because estimates are second-class:
  on real deployments, measure.
- The interference model is first-order (bandwidth sharing by
  demand proportion, uniform power derating).
- The local search cannot cross every infeasible valley; see the
  worst-case note above.
- Costs are predictions in seconds/joules under the model, not
  guarantees; money cost is intentionally *not* a constraint —
  Yosegi optimizes efficiency (time/energy), not price.

## License

MIT — see `LICENSE`.
