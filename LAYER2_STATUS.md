# Layer 2 — Code Map & Status

> Answers **Section 9 (`TODO — fill in before coding`)** of [`context_layer2.md`](context_layer2.md).
> Compiled 2026-08-02 by reading the repo, not the documentation. Every claim points at a file and line.
>
> Method: read all of `layer2_real/`, run `test_dataset.py` (read-only), inspect
> `results_real.csv` / `results_v2.csv` / `run_real.log`.
>
> **Update 2026-08-02, later the same day:** §9.2 originally reported that no checkpoint existed and
> no code could produce one. That gap has since been closed — see [**Update**](#update--dod-4-closed)
> at the end. The §9.2 findings are kept as written, because they explain *why* the work was needed.

---

## One-paragraph summary

The scientific half of Layer 2 is **complete and verified**: frozen data, an asserted graph, three
models, a 4-horizon decomposition over 40 paired units, a genotype-controlled permutation null, and
~90 leakage-guard assertions that pass. What was **entirely absent** were the two artifacts
`context_layer2.md` §8 calls the most important deliverables of this phase: the **STGNN checkpoint**
and the **per-tree risk output**. Neither was half-finished — neither had been started. The
checkpoint has since been built (see Update); the risk output has not.

---

## 9.1 Current Layer-2 code shape

**No notebooks** were part of the original Layer 2 code. It is entirely plain Python scripts in
`layer2_real/`. The only notebook in the repo was `layer1_build/solution_clean.ipynb`, which belongs
to Layer 1. (A second notebook, `layer2_real/checkpoint_walkthrough.ipynb`, was added by the
checkpoint work — see Update.)

| File | Lines | Status | Role |
|---|---:|---|---|
| `dataset.py` | 492 | **Runs** | Full implementation of the `INTERFACE.md` contract |
| `models_real.py` | 211 | **Runs** | `MLPBaseline` / `STGNN` / `STGNN_SID` |
| `run_real.py` | 482 | **Runs** | Main decomposition driver → `results_real.csv` |
| `perm_null.py` | 229 | **Runs** | Permutation null + Mantel-Haenszel stratification |
| `test_dataset.py` | 514 | **Runs — verified, exit 0** | ~90 assertions + 4 leakage guards |
| `dataset_v2.py` | 233 | **Runs** | Locality ladder + inoculum-age features |
| `run_v2.py` | 280 | **Runs** (see 9.5c) | → `results_v2.csv` |
| `sweep_v2.py` | 98 | **Runs** | Radius and window sweep; prints only |
| `_ds.py` | 40 | **Leftover scaffolding** | Real ↔ stub dataset switch |
| `_stub_dataset.py` | 300 | **Leftover scaffolding** | Synthetic fallback from the parallel-agent phase |

`_ds.py` + `_stub_dataset.py` are remnants of the phase when three agents worked in parallel against
`INTERFACE.md`. `dataset.py` has landed, so the stub never activates — but the fallback path is still
live code, and `IS_REAL`/`TAG` is still threaded into every banner and every row of
`results_real.csv` so a stub run cannot be mistaken for a finding (`_ds.py:8`).

Stored evidence of completed runs: `results_real.csv` (160 rows), `results_v2.csv` (116 rows),
`run_real.log`.

### What was half-finished was not the code — it was two of the three §8 outputs

| Output, `context_layer2.md` §8 | Status at audit time |
|---|---|
| 1. Trained STGNN checkpoint, loadable standalone | ❌ **no code at all** |
| 2. Decomposition table per horizon | ✅ complete, stored |
| 3. Per-tree risk score → ranked CSV + quantile risk map | ❌ **no code at all** |

Searching for `risk_score` / `quantile` / `kuantil` / risk map across all of `layer2_real/` and
`figures/` returned **one** match: the text label `"Peta risiko tingkat blok"` in
`figures/make_pipeline_drawio.py:130` — a box on a flowchart, not a map generated from data.

---

## 9.2 Checkpoint status

**No checkpoint existed.** The only `.pt` in the repo was `layer1_build/yolo11n.pt`, the pretrained
YOLO weights belonging to Layer 1.

More importantly: **no code could save one.** Searching `torch.save`, `state_dict`, `torch.load`,
`pickle.dump`, `joblib.dump` across the whole repo (`.py` and `.ipynb`) returned **zero matches**.

See `run_real.py:231-254` — `train_eval()` builds a model, trains it for 60 epochs, evaluates it,
then:

```python
    return float(average_precision_score(te[2], p)), float(te[2].mean())
```

The model falls out of scope and is gone.

### Correction to how Section 9 framed the question

The document asks *"is there a saved trained STGNN file yet, or does training still need to
run/finish? If not trained, training is the first blocker."* Based on what is on disk, the framing
needed shifting:

**Training had already run, extensively.** `run_real.log` records `[910.3s of training]` for the
primary h=3 decomposition alone — 5 configs × 2 folds × 20 seeds = **200 models trained**. With the
secondary horizons (3 × 5 × 2 × 4 = 120) and the SI(D) initialisation probe (~96), `run_real.py`
alone trained **~416 models**. `run_v2.py` trained many more.

**All of them were discarded** once their AUC-PR was recorded, because persistence was never
implemented.

So the real blocker was not *"train first"* but: **add a save path, then retrain one final model.**
The compute cost is small — one `train_eval` is ≈3.3 s on CPU (`run_real.py:418`).

### The design question no document had answered

With leave-one-parcel-out there are **two** models (one per fold) × 20 seeds. Which becomes the demo
artifact?

| Option | Consequence |
|---|---|
| Trained on 44A (tested on 44B) | Valid for evaluation; has only seen 480 palms |
| Trained on 44B (tested on 44A) | Same, 720 palms |
| A new model over all 1,200 palms | Strongest for the demo, but **has no held-out set** — no number from it may be reported as performance |

This had to be decided before the save code was written, not after. The effect size differs **2.6×**
between blocks (44A +0.0084 vs 44B +0.0219, `00_HASIL.md` Section 4), so the fold choice is not a
technical detail.

---

## 9.3 Repo path and where the Eg9PP CSV sits

- **Layer 2:** `layer2_real/`
  (absolute: `C:\Users\Rifqi\Documents\Datathon\oil-palm-detection-datathon-2026\layer2_real`)
- **Raw Eg9PP CSV:** `data_clean/Eg9PP_Phenotypes.csv`

However, **no file in `layer2_real/` reads `Eg9PP_Phenotypes.csv`.** The flow is already two-stage
and frozen:

```
data_clean/Eg9PP_Phenotypes.csv
    └─ data_clean/build_layer2_real.py      cos30 correction · KD-tree r=1.5 · A/S/D/C panel
         └─ data_clean/layer2_nodes.csv
            data_clean/layer2_panel.csv     ← FROZEN, the only downstream entry point
            data_clean/layer2_edges.csv
              └─ layer2_real/dataset.py     reads only these three (dataset.py:73-75)
```

### The "first task" at the end of `context_layer2.md` is already done — and asserted in code

The document closes with an instruction: load `Eg9PP_Phenotypes.csv`, apply `x × cos30°`, build the
graph at 1.5× spacing, report edge count and mean degree; stop there.

That is no longer pending work — it has become a **gate that halts the build** if the numbers shift:

| Check | Location | Content |
|---|---|---|
| Triangular lattice | `build_layer2_real.py:74` | `assert np.allclose(nn6, 1.0, atol=0.01)` — six nearest neighbours at 1.000 after the cos30 correction |
| Zero cross-parcel edges | `build_layer2_real.py:82` | `assert cross == 0` — this is what makes leave-one-parcel-out valid |
| Census grid | `build_layer2_real.py:67` | `assert len(census) == 45` |

The actual result is printed in `run_real.log` line 5:

```
features (T,N,d) = (45, 1200, 24) | mean degree 5.59 | edges 3354 | adjacency scale 0.1789
```

**3,354 edges, degree 5.59** — exactly the §3 expectation. The sanity check *"if the degree is ≈ 8
the scale/lattice is wrong"* already passes.

---

## 9.4 Framework

**Pure PyTorch. No PyG, no DGL, no NetworkX** — all three searched for, zero matches.

The graph is a **dense `(1200, 1200)` float32 NumPy adjacency matrix**, and message passing is a
single einsum (`run_real.py:110-113`):

```python
def diffuse(Ft, A_scaled):
    """Ft [T,N,d], A [N,N] -> D [T,N,N_REL,d] with N_REL = 1."""
    D = torch.einsum("ij,tjd->tid", A_scaled, Ft)
    return D.unsqueeze(2)
```

At N = 1,200 this is the right call: the full matrix is only ~5.8 MB, and a graph library would add
a dependency for no gain. The temporal side is an ordinary `nn.GRUCell` (`models_real.py:138`).

| Component | Used |
|---|---|
| Model & autograd | `torch`, `torch.nn`, `torch.nn.functional` |
| Graph construction | `scipy.spatial.cKDTree` |
| Features & metrics | `sklearn` (`StandardScaler`, `average_precision_score`) |
| Data | `numpy`, `pandas` |
| Training | **full-batch**, no `DataLoader`, Adam lr 5e-3 wd 1e-4, 60 epochs |
| Device | `run_real.py:64` **pins CPU** deliberately; only `run_v2.py:47` selects CUDA when available |

The 60-epoch choice is not inherited — `run_real.py:66-70` records the curve that measured it
(ep20 .168 · ep40 .164 · ep60 .173 · ep100 .173 · ep150 .157 · ep400 .106): a plateau at 60–100,
pure overfitting beyond that.

---

## 9.5 Additional findings before proceeding

`context_layer2.md` was written as a handoff **before** the code existed. The code has since
overtaken it in several places.

### a. The parameter table in §5 is stale

| Model | Document §5 | **Actual** (`run_real.log` lines 7-11) |
|---|---:|---:|
| MLP | 4,225 | **3,713** |
| STGNN | 9,422 | **8,875** |
| STGNN+SI(D) | 9,425 | **8,878** |

`00_HASIL.md` records this correction, dated 2026-07-24: the old figures came from the **synthetic**
configuration (`in_dim=32`, `n_rel=3`), not the Eg9PP run (`in_dim=24`, `n_rel=1`). The SI(D) head
contributes **3 parameters** (`rates` 2 + `res_scale` 1), not the 112 of the synthetic SEIR head.

### b. The 40-pair 4-horizon numbers are NOT in `results_real.csv`

This matters when quoting results into the manuscript:

| Source | h=3 | h=1, 2, 4 |
|---|---|---|
| `results_real.csv` (`run_real.py`) | n=40 | **n=8** — reduced-seed shape check (`SEEDS_SEC`) |
| `results_v2.csv` (`run_v2.py`, `TANGGA` block) | n=40 | **n=40** |

The 4-horizon table in `00_HASIL.md` §2.2 comes from **`results_v2.csv`**. Cross-checked: h=1
temporal `+0.00442`, sign 34/40 — an exact match. Do not look for those numbers in
`results_real.csv`.

`run_real.py:471-473` does state this plainly in its banner (*"REDUCED seed count … Read these as a
shape check, not as the primary table"*), but it is easy to miss if the CSV is read directly without
the log.

### c. `results_v2.csv` cannot be reproduced with a single command

`run_v2.py` processes **one** horizon per run (`H`, default 3, `run_v2.py:49`) and opens `OUT` in
`"w"` mode (`run_v2.py:272`). The committed file holds 29 rows × 4 horizons = 116 rows, so it was
assembled from **four runs** with different `OUT` values and then merged. **There is no merge script
in the repo.**

⚠ Running a bare `python run_v2.py` will **overwrite** `results_v2.csv` down to h=3 only — deleting
87 rows of results. The same applies to `00_RINGKASAN.csv`: no script in the repo produces it.

### d. Naming differs between document and code

`context_layer2.md` §6 calls the view `nograf`. The code uses `zero` (the adjacency view name,
`dataset.py:330`) and `nograph` (the config label, `run_real.py:259`). Not a bug, but search for the
right term.

### e. Honesty constraint #1 was untested because no risk output existed

No code emits a calibrated probability. `run_real.py:253` does call `torch.sigmoid()`, but the result
goes straight into `average_precision_score`, which is **rank-based**. This constraint only bites
once §8 output 3 is built — that is where "quantiles, not percentages" has to be enforced.

### f. Honesty constraint #4 is not fully carried into the code

`INTERFACE.md:23-30` records the 2026-07-24 decision to rename **"root contact" → "proximity
(contact proxy)"**, with explicit reasoning (Rees et al. 2009 vs Pilotti et al. 2018). Docstrings and
variable names still use the old term in many places:

| Location | Content |
|---|---|
| `models_real.py:120` | `N_REL = 1    # root contact only — see Change 1` |
| `run_real.py:111` | `"""… -> D [T,N,N_REL,d] with N_REL = 1 (root contact only)."""` |
| `models_real.py:172` | the `[beta_root, lambda0]` parameter |
| `dataset.py:330` | `true  graf kontak akar dari layer2_edges.csv` |

Cosmetic, and it **does not change behaviour** (INTERFACE.md states the contract is unchanged) — but
the constraint itself reads *"Judges will check these."*

---

## Definition of Done scorecard (`context_layer2.md` §10)

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Graph ~3,354 edges, degree ~5.59 | ✅ | `build_layer2_real.py:74,82` (asserts) · `run_real.log:5` |
| 2 | Three models, leave-one-parcel-out, 20 seeds | ✅ | `results_real.csv` h=3 n=40 · `run_real.log` |
| 3 | Decomposition reproduces (structure POS, strengthening; SI(D) NEG reported honestly) | ✅ | `results_v2.csv` `TANGGA` block, 4 horizons at n=40 |
| 4 | Checkpoint saves + reloads + runs one forward pass standalone | ✅ **closed** | see Update below |
| 5 | Ranked CSV + quantile risk map | ❌ | **no code** |
| 6 | Honesty constraints from §1 respected | ⚠️ | #2 ✅ #3 ✅ · #1 now enforced in the checkpoint path (9.5e) · #4 still open (9.5f) |

---

## Update — DoD #4 closed

Three files were added. **No existing file was modified** (`git status` shows zero `M` entries), and
`test_dataset.py` still passes.

| File | Role |
|---|---|
| `layer2_real/train_final.py` | Trains one final STGNN over all 1,200 palms, saves the checkpoint |
| `layer2_real/verify_checkpoint.py` | Five checks proving a standalone, faithful reload |
| `layer2_real/stgnn_final.pt` | **The artifact** — 60.4 KB |
| `layer2_real/checkpoint_walkthrough.ipynb` | Executed notebook walking through all of the above |

**Decision taken on the §9.2 question:** the third option — a new model over all 1,200 palms. The
artifact is used to *rank palms within the same plantation*, not to generalise to a different one, and
a fold model would score 480–720 palms through a half-unfamiliar graph. The cost is that the model has
no held-out set, so the warning is stored **inside** the checkpoint as `scope_warning`, where it
cannot be separated from the weights.

**Architecture is reused, not reimplemented.** `train_final.py` imports `run_real.py` and
`models_real.py` (both import-safe — all execution sits under `__main__`) and uses `R.Bundle`,
`R.diffuse`, `R._gather_window`, `R._focal`, `M.build("STGNN")`. The only substantive difference from
the decomposition path is the absence of fold-mask filtering.

| | |
|---|---|
| Trained on | 40,828 examples · 1,915 positives (4.69%) · h=3 · 60 epochs · seed 0 · 17 s CPU |
| Query coverage | 1,194/1,200 palms — the other 6 were symptomatic by t ≤ 2.5 yr, i.e. before the first valid census, so they never enter the risk set (prohibition #5). They still train the model as graph neighbours, degree 3–6 |
| Inference input | `F_seq (672, 3, 24)` · `D_seq (672, 3, 1, 24)` |
| Output | `logits (672,)` float32; on the full lattice a `(1200,)` vector with 528 NaN |
| Faithful reload | probe logits reproduced to max \|diff\| **2.98e-08** |

The binding check is not `load_state_dict` — a model with corrupted weights would pass that too. It
is the probe comparison: logits computed after reload must equal the logits stored at save time. The
checkpoint also stores SHA-256 fingerprints of the feature and adjacency tensors, so a shift in the
frozen CSVs fails hard rather than silently producing wrong scores.

**Still open: DoD #5.** The ranked CSV and quantile risk map are not built. The notebook's §9 plot is
a diagnostic only. This checkpoint is the input that step needs.
