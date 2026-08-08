# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**SawitGuard-GNN** — a reproduction package for a Datathon 2026 (Ristek CSUI) submission on early warning
for Basal Stem Rot (BSR / *Ganoderma boninense*) in oil palm. It is a **research/evidence package**, not an
application: there is no server, no CLI entry point, no packaging. Every script is a standalone experiment
that prints a report and/or freezes a CSV.

Start at `00_HASIL.md` — it holds the full pipeline description and every result. `00_RINGKASAN.csv` is the
raw number table. Prose documentation is in Indonesian; code docstrings are mixed (Layer 1 English,
`layer2_real/dataset.py` and the v2 runners Indonesian).

## Commands

No test runner, no linter, no build system. Everything is `python <script>.py`, run **from the script's own
directory** (all paths are anchored to `__file__`, so cwd rarely matters, but the README order assumes it).

```bash
pip install numpy pandas scipy scikit-learn torch lightgbm opencv-python ultralytics roboflow
# paper/ scripts additionally need python-docx
```

**Verification / guard suite** (the closest thing to a test — ~90 assertions, 4 leakage guards, ~10 s, CPU):

```bash
python layer2_real/test_dataset.py      # exit 0 = all guards pass; exit 1 = prints which failed
```

**Freeze the clean datasets** (optional — the CSVs are committed; these rebuild them and hard-assert every
headline number, so a shifted count aborts the run):

```bash
cd data_clean
python build_layer1.py        # -> layer1_crowns.csv (5,077 unique trees, 66 positives)
python build_layer2_real.py   # -> layer2_nodes/panel/edges.csv (1,200 nodes, 45 censuses, 3,354 edges)
```

**Layer 1 (UAV imagery):**

```bash
cd layer1_build
python exp_health.py                      # LightGBM health, leave-one-ortho-out, ~20 s CPU
python yolo_prep.py                       # COCO -> yolo_B/ + per-fold yamls
python yolo_train.py                      # needs GPU; env-configured (see below)
python compare_models.py [baseline_tag]   # paired per-fold diff across yolo_results_*.json
python premise_test.py                    # join-count spatial clustering test (read-only)
python stitch_probe.py [B]                # reconstruct plantation geometry from tile filenames
python grids.py B                         # dataset stats + sample grids into out/
```

**Layer 2 (Eg9PP field panel), all CPU:**

```bash
cd layer2_real
python run_real.py [seeds] [n_perm]  # main decomposition -> results_real.csv  (~22 min at defaults 20/500)
python run_real.py probe [seeds]     # SI(D) initialisation probe only; APPENDS to results_real.csv
python perm_null.py [n_perm]         # genotype-controlled permutation null   (~45 s)
python run_v2.py                     # locality ladder + feature ablation -> results_v2.csv
python sweep_v2.py                   # graph-radius and history-window sweep (prints only)
python run_v3.py 20                  # single-photo variant, within-census AP -> results_v3.csv (~12 min)
python run_v3_perm.py 200 2          # within-family null for v3 (~28 min); STRATA=progeny_parcel for the strict one
```

Environment knobs (defaults in parentheses): `EPOCHS` (60 for Layer 2, 25 for YOLO), `SEEDS` (5 in the v2
runners), `SEEDS_SEC` (`seeds//5`, used for the non-primary horizons), `H` (3), `OUT`, `FOLDS` (`0,1,2`),
`IMGSZ` (640), `MODEL` (`yolo11n.pt`), `CACHE`, `WORKERS`, `ROBOFLOW_API_KEY`.

**Windows/PowerShell note:** the README uses bash-style inline env vars (`FOLDS=0,1,2 EPOCHS=50 python
yolo_train.py`), which PowerShell does not parse. Use `$env:FOLDS='0,1,2'; $env:EPOCHS=50; python
yolo_train.py`, or run those lines through the Bash tool.

## Architecture

### Two layers, deliberately not joined

The pipeline is **cut in the middle, on purpose**, and that cut is a finding rather than a bug:

- **Layer 1** (`layer1_build/`, `layer1_data_audit/`) — UAV RGB imagery from Roboflow `ds_B`. Crown
  detection (YOLOv12n via `y12.py`; the older YOLO11n path in `yolo_train.py` is superseded),
  crown-health classification (LightGBM), crown geometry. Labels are *generic crown health, not BSR*,
  on a single survey date. Stage 1's **primary metric is crown-centre P/R/F1 on unique trees**, not
  mAP — the ground-truth boxes are fixed-size stamps, so mAP has a model-independent ceiling
  (`LABEL_QUALITY_AUDIT.md`). A third, independent source (`anom.py`, Peru `PalmAnom`/`PalmSan`) is
  a separate evidence line and is never merged with `ds_B` either.
- **Layer 2** (`layer2_real/`) — the Eg9PP field panel (Tisné et al. 2017): 1,200 palms, 45 censuses over 25
  years, **field-verified Ganoderma**, but **no imagery at all**.

No dataset has both imagery and per-tree spread, and the two sources have different estates, different eras,
no join key and no georeferencing. They are therefore never merged; what is measured instead is *interface
compatibility* — mean degree at r = 1.5 × planting distance, inner trees only, is **5.54 ± 0.12**
(Layer 1, from *detector predictions*) vs 5.74 (Layer 2); 5.62 ± 0.05 is the same figure computed on
ground-truth boxes. Do not write code that joins them.

An inference bridge is a *different* thing. The frozen Layer 2 checkpoint takes 24 features =
4 SELF (needs a time axis) + 14 GENO (genotype, invisible from imagery) + 6 STATE, over a 3-census
window, so 18 of 24 columns are unfillable from one flight — that checkpoint cannot consume Layer 1
output, and zero-filling the gap is forbidden.

**The `v3` variant closes this, and it is the most consequential result in the package.**
`layer2_real/dataset_v3.py` + `run_v3.py` drop SELF and GENO and set `W=1`, leaving only STATE as
diffusion payload — i.e. *neighbour condition through the graph*, which is exactly what one photo
yields. Two things came out of it:

- **Within-census AP is the only fair metric.** Pooled AP rewards guessing *which census this is* —
  useless for ranking palms inside one snapshot. The full model loses 47% of its value under the
  fair metric (0.1818 → 0.0973); v3 loses almost nothing (0.1259 → 0.1015). On the real task the
  photo model **matches** the full model (+0.0042 ± 0.0035, 36/40 — "not weaker", not "better").
- **77% of v3's skill comes specifically from the CORRECT contact map** (+0.0296 ± 0.0057, 40/40).
  Without the graph it is exactly at chance, because STATE is provably 0 on the risk set.

v3 deliberately breaks two locked rules (`WINDOW=3`, and prohibition #7 requiring genotype). That is
declared in `layer2_real/INTERFACE.md`, not done silently, and the cost is **measured** rather than
assumed: a within-family permutation null (`run_v3_perm.py`, 200 permutations per stratum) puts the
family contamination at **36%**, with the remaining **64% spatial** and 0/200 permutations reaching
the observed value under the strictest stratum. Prohibition #7 stays fully binding for `run_real.py`
and `run_v2.py` — only v3 has the null that quantifies its own contamination.

The end-to-end cost of that handoff is now **measured** (`run_v3_noisy.py`): trained on clean status,
tested on detector-rate noise (recall 0.446, fpr 0.0094, measured on ds_B by
`layer1_build/unhealthy_threshold.py`), within-census AP falls 0.0916 → 0.0800 — lift 1.45× → **1.27×**,
**59% of the signal survives**. Real cost, not a fatal one. What remains unmeasured is narrower: those
rates come from ds_B, a different estate than Eg9PP.

One hypothesis was tested and **rejected**: a separate confidence threshold for the Unhealthy class,
chosen cross-fold, is *worse* than reusing the localisation threshold 0.75 (F1 0.370 vs 0.406) — with
17–31 positives per orthomosaic the optimum is noise and does not transfer. Keep 0.75. Related
correction: "the detector only finds 0–1 symptomatic palms per tile" is the base rate, not a failure —
a 1024² tile holds ~65 palms at a 1.3% Unhealthy rate.

### `data_clean/` is the only entry point

`build_layer1.py` and `build_layer2_real.py` freeze the raw sources into six CSVs. Every downstream script
reads those CSVs; **nothing downstream may read raw COCO or `Eg9PP_Phenotypes.csv` again**. Both builders
assert their headline numbers (5,077 / 66 / 45 / 3,354 / 0 cross-parcel edges), so drift aborts loudly.
`DATASET_CARD.md` documents the resulting datasets plus a "batas yang dipaksakan" (forced limit) row per
dataset that bounds what the paper may claim.

Two non-obvious data facts baked into the builders:

- Layer 1 deduplicates **annotations → unique trees**. Roboflow tiles overlap ~30×, so one physical tree
  appears in a median of 32 tiles; 151,060 annotations are only 5,077 trees. Each tree gets one *canonical
  view* (the tile where its crown centre is furthest from any tile edge). Never treat 151,060 as a sample size.
- Layer 2's `X_POSITION`/`Y_POSITION` are **not to scale**. `xm = X * cos30°` makes the six nearest
  neighbours land at exactly distance 1.000 (equilateral triangular planting). Without that correction the
  contact graph is wrong.

### The Layer 2 contract

`layer2_real/INTERFACE.md` is a **locked contract** (three agents built against it in parallel) fixing
`WINDOW=3`, `HORIZONS=(1,2,3,4)` in census steps, `N_REL=1`, the six functions `dataset.py` must export, the
tensor shapes `F_seq [B,W,d]` / `D_seq [B,W,N_REL,d]`, and six hard prohibitions. Changing it must be stated
explicitly, not done silently.

`_ds.py` is a dataset switch: it prefers the real `dataset.py` and falls back to `_stub_dataset.py`
(synthetic scaffolding from the parallel-development phase). `IS_REAL`/`TAG` is threaded into every banner
and every row of `results_real.csv` so a stub run can never be mistaken for a finding. `dataset_v2.py`
*extends* `dataset.py` without modifying it, so the frozen baseline stays bit-reproducible.

### The core experiment: a three-way decomposition

Every Layer 2 result is a paired difference between graph *views* of the same model:

```
temporal   = nograph - MLP        value of modelling time + genotype
prevalence = random  - nograph    value of having ANY graph
STRUCTURE  = true    - random     value of the CORRECT contact map
```

`random` preserves each node's degree (double-edge swap **within parcel**, so the zero-cross-parcel-edge
property survives); `random_local6`/`random_local3` (v2) additionally cap new edge length, turning the
control into a locality ladder. The adjacency scale is always taken from the **true** map so all views share
one normalisation — that is what makes `random` a fair control.

Models (`models_real.py`): `MLPBaseline` (no graph, imported verbatim from repo-root `models.py` so the two
halves cannot drift), `STGNN` (single-relation diffusion + GRU), `STGNN_SID` (STGNN + a 3-parameter
mechanistic head). `run_v2.py` defines a local `STGNN_MR` subclass for the 2-hop variant, because
`models_real.STGNN` deliberately *rejects* `n_rel != 1` (softmax over one relation is a constant 1.0 and the
parameter would get exactly zero gradient).

### Significance harness

`paired()` in `run_real.py` (copied verbatim into `run_v2.py`) is the single decision rule everywhere,
including `layer1_build/compare_models.py`: report mean ± std across paired (fold, seed) units plus a
**sign-count**, and call anything whose |mean| < 1 std `INCONCLUSIVE`. Do not replace it with a t-test or
quietly widen a claim past its verdict.

Evaluation is **block-CV only**: leave-one-ortho-out (3 folds) for Layer 1, leave-one-parcel-out (2 folds)
for Layer 2. Random splits leak 100% on Layer 1; on Layer 2 the parcel split is legitimate specifically
because 0 of 3,354 edges cross a parcel boundary and all 14 families appear in both parcels. Metric is
AUC-PR (positive rates are 1.3–6.0%).

### Leakage discipline

`dataset.py`'s feature matrix is split into three blocks — SELF (4 time columns), GENO (14 one-hot progeny),
STATE (6 own-status columns). STATE is **provably constant 0 on the risk set**; it exists only as *diffusion
payload* so that `D = A @ F` carries symptomatic/dead neighbour counts. Columns are excluded on purpose and
the reasons are in the module docstring: `prev_global` would hand the MLP the prevalence component for free,
`deg`/`n_nb_*` would smuggle the graph into the no-graph baseline, `xm`/`ym` would let it memorise hotspots,
and the event columns are the outcome itself.

`test_dataset.py` proves four guards fire (per-progeny identity, STATE-zero, bit-exact truncated rebuild,
single-feature AUC smell test) and deliberately builds three leaky variants to show the guards catch them.
`run_real.py::leak_sentinel` flags any model implausibly far above the no-skill line at run time. Run
`test_dataset.py` after touching anything in `layer2_real/`.

## Working rules in this repo

The stated value is **leakage-first**: leaks, zeros and limits are reported *before* any result, and no
improvement is claimed while it sits inside the noise band. Several headline results are negative, and one
prior claim was retracted by the authors — that is intentional, so do not "clean up" negative findings.

The six prohibitions carried through the whole package (`README.md`, expanded in `INTERFACE.md`):

1. No random splits on Layer 1 — 3 orthomosaics only; random splits leak completely.
2. Never cite 151,060 as the sample size; the unit is **5,077 unique trees**.
3. Never call the Roboflow labels BSR — generic crown health, no field verification.
4. Never report accuracy — a constant "healthy" classifier is >98% accurate and useless.
5. Never treat a censored (`C`) palm as healthy — it leaves the risk set, it does not become a negative.
6. Never compare the SI(D) head against any SEIR variant — the latent E compartment is unobserved, and the
   head drops from 112 parameters to 3.

Genotype (`progeny`) is a mandatory covariate in **all** models including the MLP; without it the graph's
advantage is contaminated by related families being planted adjacently.

## Gotchas

- **Dangling references.** Docstrings mention repo-root `run_experiment.py`, `train.py` and `data/pwd.csv`
  (the synthetic SEIR simulator half). Those files are **not in this package** — the simulator was
  deliberately cut from scope. Root `models.py` is present but only `MLPBaseline` is actually imported;
  `STGNN`/`STGNN_SEIR` there belong to the absent synthetic side.
- **`paper/` is stale.** `section3.tex` still describes the simulator, claims Stage 5 runs over real
  plantation geometry (it never did), promises calibration/Brier/reliability outputs that no code produces,
  and states a 69:1 class imbalance where unique trees give ~76:1. `section3.docx` is staler still. No Eg9PP
  result has entered any manuscript. Do not quote `paper/` as fact; `paper/METHODOLOGY_PLAN.md` is the
  rewrite plan. `.docx` files are always regenerated from source by `tex_to_docx.py` /
  `make_*_docx.py` — edit the source, not the Word file.
- **`run_real.py probe` appends.** It calls `load_existing()` and rewrites `results_real.csv` with the old
  rows plus the probe rows; running it against a missing/partial CSV silently produces a partial file.
- **`ds_B/` images are hardlinks** to the working repo (~377 MB linked, ~21 MB real). They behave like normal
  files, and editing code here does not touch the original repo.
- **Runtimes are real:** `run_real.py` at defaults is ~22 min of CPU training; `yolo_train.py` is ~15–20
  min/fold on an RTX 5060. Layer 2 pins `DEVICE = cpu` on purpose (1,200 nodes × 45 censuses); only
  `run_v2.py` picks CUDA when available.
- The Layer 2 across-seed std is an **optimisation** std, not a data std — reseeding reinitialises the
  network and redraws the random graph, it does not resample the epidemic. Only the 2 parcels give
  data-level replication.
