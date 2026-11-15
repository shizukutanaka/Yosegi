# Yosegi algorithm design

Yosegi plans *where each stage of a task runs* on a heterogeneous PC
and *in what proportions*, from measured device numbers. This
document describes the models, the planner, the tuner, and — with
measurements — how close the planner gets to optimal.

## 1. Machine model (`device`)

A machine is:

- **Units**: execution engines with a kind (`cpu-performance`,
  `cpu-efficiency`, `cpu`, `gpu`), a measured compute rate, an
  optional sustained (post-throttle) rate with a thermal time
  constant, an active/idle power draw, and a home memory domain.
  Effective rate over a horizon blends peak → sustained as
  `peak − (peak − sustained)·(1 − τ/horizon)` once the horizon
  exceeds τ; before τ the unit holds peak.
- **Memory domains**: capacity + shared bandwidth (VRAM per GPU,
  system RAM, …). Bandwidth is shared by everything resident —
  the machine's main interference channel.
- **Storage**: read/write rates; stage I/O is timed against the
  fastest store.
- **Links** between domains (PCIe etc.): bandwidth + latency for
  inter-stage transfers. A default link rate applies when a pair
  is not listed.
- A whole-machine **power cap**: a scheduling constraint, modeled
  as throttling (below), not as a hard feasibility wall.

Micro-benchmark results enter through `Microbench` records
(unit ID + measured compute rate / local-domain bandwidth / storage
rate) and `ApplyMicrobench`, which returns an updated machine
(the receiver is not mutated). A machine **signature** (sorted unit
IDs with kinds, capacities, rounded rates, power cap) identifies
the configuration for tuner records.

## 2. Task model (`workload`)

A task is a sequence of **stages** repeated `Repeats` times. A stage
declares:

- `Compute` (operations), `MemBytes` (bytes streamed per execution),
  `IOBytes` (storage traffic), `Footprint` (resident bytes);
- `ExpertBytes` / `ActiveExpertRatio` for MoE-style stages (only a
  measured fraction of expert bytes is touched per execution);
- `Activation` bytes handed to the next stage (transfer costing);
- `Splittable` + `SplitSyncCost` (fractional overhead per extra
  participating unit);
- `Cacheable` fraction avoidable on repeats, charged through the
  task's measured `CacheHitRate`;
- `AllowedKinds` unit-kind affinity (a compiler stage cannot run
  on a GPU);
- optional `SpecCapability` (draft cost ratio, draft footprint,
  **measured** acceptance α) and `Reductions` (declared ways to
  shrink the task, e.g. a quantization ladder).

The **LLM family** builds tasks from a model spec: decode is one
stage per layer, repeated per generated token, streaming that
layer's weights + KV share (bandwidth-bound — the Alloy
investigation's central measurement); prefill is the same stages
with compute scaled by prompt tokens and `Cacheable ≈ 0.95`
(prefix caching). KV bytes/token use the measured per-element
sizes (f16 2.0, q8_0 1.0625, q4_0 0.5625).

## 3. Cost model (`cost`)

For a plan (one placement per stage; a placement is a set of units
with shares, optionally reading from a different memory tier, with
MoE experts optionally tiered elsewhere, plus an optional
speculative-execution choice), `Evaluate` predicts:

- **Per-stage time** = max(compute time, memory-group time, I/O
  time) + inter-stage transfer time. Memory time is computed per
  *domain group*: units sharing a domain stream through its shared
  bandwidth, so co-resident shares contend. Split stages pay the
  sync multiplier. Bottleneck classification
  (compute/memory/io/transfer) falls out of the max.
- **Repeats**: first execution in full; later executions discount
  the cacheable fraction by the hit rate.
- **Feasibility**: resident bytes per domain (dense shares on home
  domains, expert shares on their tier, draft footprint on the
  draft unit) must fit capacities; kind affinity must hold.
  Violations produce named reasons, never silent failure.
- **Thermal**: two passes — a peak-rate pass estimates the
  horizon, a second pass re-times with horizon-effective rates.
- **Power cap**: if the plan's active power exceeds the cap,
  times scale by the over-draw ratio (throttling), flagged in the
  result's notes.
- **Speculative execution**: expected steps per verify cycle
  `E = (1 − α^(n+1))/(1 − α)`; a cycle costs one full pass plus
  `n` draft passes. If the predicted cycle time per accepted step
  is worse than plain execution, the result notes the loss —
  speculation is admitted only when the model, fed with a
  *measured* α, says it wins.
- **Energy** (lower bound): active energy of used units + idle
  energy of the rest over the horizon.

`EvaluateConcurrent` adjusts several co-running instances:
bandwidth demand per domain is summed; over-subscribed domains
slow their memory-bound fractions proportionally; the power cap
derates everyone uniformly. Makespan is the slowest instance.

## 4. Planner (`planner`)

`Solve` = generate → evaluate → polish → rank.

**Candidate generation**

1. Single-unit plans (one per unit).
2. **DP assignment**: dynamic programming over (stage, option),
   where an option is a whole assignment to one allowed unit or
   the split-across-all-units placement; transfers are charged
   between consecutive options' home domains (largest-share
   domain — the same definition the cost model uses, so generator
   metrics and evaluated costs agree). Up to 3 *penalty repair
   iterations* re-run the DP with ×4 stage-time penalties on units
   whose domains the previous solution overflowed.
3. **Greedy assignments**: fastest-fitting unit per stage, in
   stage order and in largest-footprint-first order.
4. **Subset splits**: for every unit subset of size 2..MaxSplit,
   each splittable stage is split with shares ∝ 1/(stage time)
   (i.e. ∝ measured throughput for that stage), water-filling
   capped so that a uniform split's residency respects every
   domain (`SplitPlacement`, also used by the simulator's brute
   force so both search the same shapes).
5. **Offload variants**: MoE experts (or whole stages) tiered to
   the largest domain, on the strongest base plans.
6. **Speculative variants**: depths {1,2,3,5,8} × draft units, on
   the strongest base plans — only when the task declares a
   measured acceptance.

**Polish (local search on the true objective)**

- *Repair*: infeasible candidates are fixed by moving whole stages
  out of overflowing domains at the least stage-time cost
  (immediately abandoned when the task's total footprint exceeds
  total machine memory — no assignment can fix that).
- *Hill climbing* from the best feasible plan, up to 3 repaired
  plans, and the best few distinct feasible shapes: neighbors are
  single-stage moves, pairwise swaps, and shape changes (collapse
  a split stage onto one unit; split a whole stage). Improvements
  must beat 0.1% under `cost.Evaluate`. For many-stage tasks the
  quadratic swap neighborhood is skipped and rounds are capped.

**Ranking**: feasible plans by predicted time; ties within 1%
prefer the *simpler* plan (fewer split units/tiers/speculation) —
complexity must pay for itself in measured time, not in noise.

**No-fit honesty**: when nothing is feasible, the planner returns
the blocking reasons (from actual evaluations) plus suggestions
that were *tested*: each declared reduction is re-planned
(feasibility probe, no polish; at most 2 fitting suggestions are
reported). Reductions strictly descend — applying one removes it
and all weaker entries from the ladder, rescaled — so probing
always terminates.

## 5. Online tuner (`tuner`)

Estimates rank; measurements decide. Given a `Runner` (the host
application executes a plan and reports seconds/throughput and,
for speculative plans, measured acceptance), `Tune`:

- takes the planner's top-K plans,
- measures them,
- picks the winner by measured throughput (ties within 1% prefer
  the simpler plan; failed/unmeasured candidates never win),
- persists a `Record` — machine signature, task signature, the
  winning plan, all measurements — as JSON, written atomically.

`Lookup` returns the record only when *both* signatures match the
current machine and task. Any hardware change, re-measurement, or
task change misses, and planning falls back to estimates until the
next tune. This mirrors Alloy's tuned-plan persistence, with the
signature computed from the full machine model.

## 6. Simulator and measured optimality (`sim`)

`BruteForce` enumerates, per stage: every allowed single unit,
plus the split-across-all-units placement (`SplitPlacement`, the
same shape the planner generates). Speculative execution and
offload tiers are outside this space (the planner can therefore
occasionally *beat* brute force with richer shapes; those cases
are counted separately, never as agreement).

Measured on random small cases (2–3 units, 3–6 stages, mixed
compute/memory/IO stages, capacities that bind in ~40% of cases):

| Seed | Cases | Compared | Agreement | Mean gap | p95 gap | Max gap | Brute-only |
|---|---|---|---|---|---|---|---|
| 42 | 200 | 141 | 83.7% | 0.84% | 0.61% | 69.4% | 1 |
| 7 | 150 | 105 | 81.9% | 1.01% | 0.78% | 70.1% | 1 |

So: **the planner is a heuristic, and this is its measured
distance from optimal** — exact in ~5 of 6 comparable cases,
within ~1% on average when not exact. The residual worst cases
are adversarial tight-capacity mixes whose optimum requires
coordinated multi-stage moves (move A out *and* B in *and*
collapse C) that single-neighbor local search cannot reach
without crossing infeasible or worse intermediate plans. One
case per seed remains where brute force finds a feasible plan
the planner misses entirely. Both facts are pinned by
`sim/sim_test.go` with regression thresholds (agreement ≥ 80%,
mean gap ≤ 2%, p95 ≤ 2%, max ≤ 75%, brute-only ≤ 2): improvements
should tighten them; regressions fail CI.

## 7. What the Alloy evidence changed vs. naive planning

- Layer splitting across the three cards measured ~flat
  (70B, 2→6 devices: 16.6 → 17.1 tok/s): hence splits are
  capacity-driven, and single-device plans win ties.
- A classical draft model made a fast target *slower* (0.27×)
  while MTP measured ~1.98×: hence speculation is modeled with
  acceptance and admitted only when predicted (and later
  measured) to win.
- Prefix caching measured 6197 ms → 42 ms on repeat prompts:
  hence cache effects are in the cost model, not an annotation.
- Split ratios by VRAM would have overloaded the small cards;
  ratios here follow measured per-stage times, capped by
  residency.

## 8. Known limitations

- No live-hardware validation in this repository yet; profile
  constants come from the Alloy investigation's measurements.
- Interference and power models are first-order approximations.
- Split shares are uniform across stages in subset-split
  candidates (per-stage residency coupling is handled by repair
  and local search, not by the split generator).
- The planner optimizes one task at a time; concurrent planning
  is evaluated (`EvaluateConcurrent`) but not searched.
