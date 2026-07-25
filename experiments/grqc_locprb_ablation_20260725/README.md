# GrQc LocPRB Ablation Plan

Date: 2026-07-25

Local safety baseline: `edaa26c` (`main`, retained locally)

Remote-safe source baseline: `8aea770` (same active source code, excludes the
server access document)

## Objective

Diagnose and reduce GrQc regret with LocPRB before changing the other datasets.
The historical original-repository PRB result,
`4386.9 +/- 325.3` at `T=10000` over ten seeds, is an audit anchor only. It is
not directly comparable with the current formal LocPRB result because the two
runs use different entry files, learning rates, propagation solvers, and random
streams.

All selectable comparisons in this experiment use deterministic event tapes
and paired seeds. Buggy variants may be retained for attribution, but a buggy
variant cannot become the final implementation merely because its regret is
lower.

## Common protocol

- Dataset: GrQc, 5,242 nodes.
- Candidate pool: 10 arms, one positive and nine negatives.
- Primary method: per-round scratch Numba LocPRB.
- Training: every 50 rounds before round 2,000 and every 100 rounds afterward.
- Screen: `T=1000`, validation seeds `100,101,102`.
- Confirmation: `T=10000`, five validation seeds selected only after screen
  results are summarized.
- Final test seeds are not used in this experiment.
- Every result stores the event digest, configuration, cumulative regret,
  direct/LocPRB/power shadow decisions, graph-update counters, and source
  statistics.

Before every run, the following contracts are checked when applicable:

1. Exactly one positive arm exists.
2. `candidate_edges[positive_arm] == positive_edge`.
3. Shared-serving variants have one common source endpoint.
4. Correct-update variants insert `candidate_edges[selected_arm]`.
5. Warm and online-positive undirected edge sets are disjoint for corrected
   split variants.
6. Online and evaluation event tapes have different seeds and no identical
   event records.

## Group 1: comparable anchors

| ID | Entry pool | LR1/LR2 | Solver | Purpose |
|---|---|---:|---|---|
| `g1_original_loc` | root `GrQc_ALLusers_entry.npy` | 0.1/0.01 | LocPRB, eps 1e-6 | Replace the original PRB propagation path with accurate scratch LocPRB while retaining the original data/LR style |
| `g1_current_loc` | `Insert/GrQc_ALLusers_entry.npy` | 0.01/0.004 | LocPRB, eps 1.91e-4 | Reproduce the current formal LocPRB protocol in the isolated runner |
| `g1_current_power` | same as current Loc | 0.01/0.004 | power iteration, 50 | Paired current-protocol solver reference |

The historical PRB `4386.9 +/- 325.3` is shown in reports but is not rerun or
used as a paired delta.

## Group 2: GrQc semantic ablations

All one-factor variants start from `s0_legacy`. `sf_all_fixed` accumulates the
valid fixes.

| ID | Change from legacy | Question |
|---|---|---|
| `s0_legacy` | none | Isolated legacy baseline |
| `s1_selected_edge_update` | update the graph from the selected candidate edge | How much regret is caused by the `arm+1` endpoint bug? |
| `s2_disjoint_undirected_split` | canonicalize undirected warm/online sets and remove overlap | How much does the 18% warm/online overlap matter? |
| `s3_shared_serving` | all ten candidate edges share the positive edge source | Does the paper serving-node protocol improve regret? |
| `s4_no_positive_replacement` | draw online positives without replacement | Do repeated already-known positives harm learning? |
| `s5_independent_eval` | independent fixed evaluation tape | Does online/evaluation overlap distort reported accuracy? |
| `sf_all_fixed` | selected-edge update + disjoint split + shared serving + no replacement + independent evaluation | Correct cumulative semantic version |

For the disjoint split, existing warm edges are first canonicalized as
`(min(u,v), max(u,v))`; online positives are the full undirected ground-truth
edge set minus those warm edges. This preserves the legacy warm graph as much
as possible while enforcing disjointness.

Negatives in shared-serving variants are uniformly sampled unique targets that
are not the serving node and are not neighbors in the full ground-truth graph.

## Group 3: reranking ablations

These runs use `sf_all_fixed` events and graph semantics.

| ID | Training decision | Purpose |
|---|---|---|
| `r0_eenet_direct` | raw EE-Net score | Network-only reference |
| `r1_locprb` | scratch LocPRB score | Primary method |
| `r2_power` | 50-step power-iteration score | Full-propagation reference |

Each run also computes all three shadow decisions before updating the model.
The report includes:

- EE-Net wrong / LocPRB correct;
- EE-Net correct / LocPRB wrong;
- LocPRB/power decision disagreement;
- first disagreement round;
- cumulative regret for each shadow policy on the same model trajectory.

The net reranking gain is:

```text
count(EE-Net wrong, LocPRB correct)
- count(EE-Net correct, LocPRB wrong)
```

## Group 4: source ablations

These runs use `sf_all_fixed`, scratch LocPRB, and the current learning rates.

| ID | Personalization/source |
|---|---|
| `p0_raw` | current raw exploitation + exploration score |
| `p1_l1` | raw score divided by its L1 norm; signs retained |
| `p2_clip_l1` | clip negatives to zero, then L1 normalize |
| `p3_softmax` | softmax over the ten candidate scores |
| `p4_exploit_only` | exploitation score only |
| `p5_exploit_explore` | explicit exploitation + current exploration rule; control duplicate of raw semantics |

`p0_raw` and `p5_exploit_explore` are intentional controls. Their event digests,
decisions, and regret must match for the same seed.

## Selection rules

1. Contract violations disqualify a result.
2. Rank screen variants by mean paired final regret, but also report regret at
   rounds 100/500/1000 and area under the cumulative-regret curve.
3. Use paired per-seed deltas and a 95% confidence interval.
4. For solver/source choices, inspect LocPRB-versus-power disagreement and net
   reranking gain; do not use a numerically inaccurate APPR variant solely
   because it happens to lower validation regret.
5. Select confirmation variants from validation seeds only.

## Remote execution layout

Four persistent `nohup` queues run concurrently:

```text
GPU 0 -> group1
GPU 1 -> group2
GPU 2 -> group3
GPU 3 -> group4
```

Within each queue, variants and seeds run sequentially. Each task writes an
atomic `status.json`; completed tasks are skipped on restart and failed tasks
retain their logs.

After the T=1000 screen, run the selected unique configurations at T=10000:

```bash
bash experiments/grqc_locprb_ablation_20260725/launch_confirmation.sh \
  DATA_DIR OUTPUT_DIR LOG_DIR 10000 200 201 202 203 204
```

The confirmation queues intentionally avoid exact duplicate configurations:
`sf_all_fixed`, `r1_locprb`, and `p0_raw` are equivalent; `g1_current_loc` and
`s0_legacy` are equivalent. Results from the representative are reused for
those labeled comparisons.
