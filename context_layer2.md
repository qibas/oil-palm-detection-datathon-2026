# CONTEXT — Layer 2: Graph-Based Ganoderma Risk Forecasting

> Handoff for Claude Code. Scope of THIS document = **Layer 2 only**.
> Layer 1 (UAV detection + health) is being built by a teammate in parallel — out of scope here.
> Read fully before writing code.

---

## 0. Scope right now

Layer 2 takes a **plantation graph** (trees as nodes, proximity as edges, some trees marked as disease sources) and produces a **per-tree risk score** for still-healthy trees. It is trained and evaluated on the Eg9PP field panel.

We are NOT building the UAV bridge here (that joins Layer 1 to Layer 2 later). We ARE building/finishing: data prep → graph construction → the three models → decomposition + evaluation → saved checkpoint + risk output.

The saved Layer-2 checkpoint is the single most important artifact this phase, because the whole demo later loads it and runs inference. Getting a clean, loadable checkpoint is the priority.

---

## 1. Honesty constraints (apply to Layer 2 code, paper, video)

Design constraints, not disclaimers. Judges will check these.

1. **Output is RELATIVE risk** (rank / quantile), NOT calibrated probability. Any code that emits a "% chance of infection" is a bug. Risk maps use quantile colouring.
2. **Model trained on a breeding-trial plantation** (Eg9PP, 2 parcels), NOT a production estate. Effect size differs up to 2.6× between blocks. State this; don't claim production-ready accuracy.
3. **Decomposition must report negatives honestly.** The SI(D) mechanistic head HURTS performance at all horizons — this is reported, not hidden. Inconclusive gaps are labelled INCONCLUSIVE, not spun as wins.
4. **"Proximity graph" not "root-contact graph".** BSR spread mechanism is disputed in the literature (root contact vs basidiospore). Edges are proximity as a proxy; the design does not separate soil / planting material / microclimate / spore gradient.

---

## 2. Data — Eg9PP

- **Source:** Tisné et al. 2017, G3 7(6):1683-1692. Repo: github.com/DenisMarie/Eg9PP_Ganoderma (CC BY-SA — cite, do NOT re-host).
- **File used:** `Eg9PP_Phenotypes.csv` (ALL 1,200 palms, NOT the 604-subset `_Mapping` file — the subset leaves holes in the lattice).
- **Columns:** `ID, PROGENY, GA_PARENT, GB_PARENT, PARCEL, PLOT, X_POSITION, Y_POSITION, EVENT_T1S, Y_T1S, EVENT_TD, Y_TD`
  - `X_POSITION, Y_POSITION` = planting grid indices per tree (NOT metres).
  - `EVENT_T1S=1, Y_T1S=t` → first symptom at time t. `EVENT_T1S=0` → right-censored (never symptomatic by end of trial).
  - `EVENT_TD` / `Y_TD` = death due to Ganoderma, same encoding.
  - `PROGENY / GA_PARENT / GB_PARENT` = genetic family → used as covariate.
- **Scope:** 1,200 palms, 14 families, 2 parcels, 45 censuses over 25 years, field-verified Ganoderma, NO imagery.

### Coordinate handling
- Raw axes are not to-scale. After correcting `x × cos(30°)`, the 6 nearest neighbours land at distance exactly 1.000 → triangular lattice confirmed.
- Do this correction before building the graph.

---

## 3. Graph construction

- Edge if inter-tree distance ≤ **1.5 × planting spacing**.
- Expected: **3,354 edges, mean degree 5.59**. Sanity check — if degree ≈ 8 the scale/lattice is wrong (that would be a square lattice).
- Single relation only (proximity).

---

## 4. Forecasting task

- Tree asymptomatic at census `t` → enters the **risk set**.
- Target: does it become symptomatic or die within `h` censuses (**h = 1..4**).
- Excluded from risk set: already symptomatic, dead, or censored. **Censored ≠ healthy.**
- Each model reads a **3-census window**.
- **Genotype is a mandatory covariate in ALL models including baseline**, because related families are planted close together — without it, kinship leaks in as a fake transmission effect.
- Positive rate ranges 1.58–5.65% depending on horizon.

---

## 5. Models (three, identical task)

| Model | Graph | Params | Purpose |
|---|---|---|---|
| MLP | none | 4,225 | baseline — all non-relational info, no neighbour knowledge |
| STGNN | neighbour diffusion + GRU | 9,422 | does the graph add value |
| STGNN+SI(D) | + trained epidemiology head | 9,425 | does a mechanistic head add value (spoiler: it hurts) |

- MLP–STGNN gap isolates the graph contribution (all three graph views run on the SAME STGNN architecture, so model size is held constant).
- SI(D): latent E compartment is unobserved in the field, so the SEIR head is reduced to SI(D).
- Output used as a **ranking**, scored by **AUC-PR**. No probability calibration.

---

## 6. Control design + decomposition (this is the novelty — must exist)

Four graph views, architecture held fixed, swap only the graph:

- `true` — actual proximity graph
- `nograf` — edges removed
- `random` — degree preserved, structure destroyed (Maslov–Sneppen style)
- `random_local` — degree AND locality preserved; each new edge capped at ≤6 then ≤3 spacings

Decomposition (differences, they sum back to STGNN−MLP):

```
temporal   = nograf − MLP     value of modelling time + genotype
prevalence = random − nograf  value of having any graph at all
structure  = true − random    value of the correct map
```

Why `random_local` is needed: all true edges are length 1.000, but global random rewire links trees at median distance 13.2 spacings — so `true − random` risks measuring locality, not map correctness. The local ladder closes this. Smaller radius = harder to win.

---

## 7. Evaluation protocol

- **Split:** leave-one-parcel-out, 2 folds. No edge crosses parcels (graph stays intact); all 14 families in both parcels (no genotype confound).
- **Repeats:** 20 seeds × 2 folds = 40 pairs. Report mean ± std + sign count.
- **Decision rule (fixed before seeing results):** gap within 1 std = INCONCLUSIVE.
- **Extra test:** 500× permutation, shuffled WITHIN family (same reason genotype is a covariate).
- Optional: Mantel–Haenszel stratification per census for the neighbour relative-risk (naive 1.36× → 1.29× stratified; don't report the unstratified inflated number).

---

## 8. Outputs of Layer 2

1. **Trained STGNN checkpoint** — must load and run a forward pass standalone. THIS is what the demo consumes later.
2. **Decomposition table** — temporal / prevalence / structure / SI(D) per horizon (for the paper Results section).
3. **Per-tree risk score** → ranked CSV + quantile risk map on the Eg9PP lattice (proves the output artifact works on real data before the UAV bridge exists).

---

## 9. TODO — fill in before coding

- [ ] **Current Layer-2 code shape** — which files/notebooks exist, what already runs vs is half-done. (Codebase exists but unfinished — map it here.)
- [ ] **Checkpoint status** — is there a saved trained STGNN file yet, or does training still need to run/finish? If not trained, training is the first blocker.
- [ ] Repo path where Layer 2 lives, and where the Eg9PP CSV sits locally.
- [ ] Confirm framework (PyTorch? PyG / DGL for the graph model?).

---

## 10. Definition of done (Layer 2)

1. `Eg9PP_Phenotypes.csv` → graph with ~3,354 edges, mean degree ~5.59 (sanity check passes).
2. Three models train and evaluate under leave-one-parcel-out, 20 seeds.
3. Decomposition table reproduces (structure positive, strengthens with horizon; SI(D) negative — reported honestly).
4. A trained STGNN checkpoint saves AND reloads AND runs one forward pass in isolation.
5. Ranked CSV + quantile risk map generated on Eg9PP.
6. Every honesty constraint in Section 1 respected in code + outputs.

---

## First task for Claude Code (do ONLY this first)

Do not build everything at once. Start with the data + graph, verified against the known numbers:

> "Read this context.md. Load `Eg9PP_Phenotypes.csv`, apply the x×cos(30°) correction, build the proximity graph at 1.5× spacing, and report edge count and mean degree. Expected ~3,354 edges and mean degree ~5.59. Stop there and report — do not train anything yet."

If those two numbers don't match, the coordinate/scale handling is wrong and everything downstream would be built on a broken graph. Fix that before proceeding.
