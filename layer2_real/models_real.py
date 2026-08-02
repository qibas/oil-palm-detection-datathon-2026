"""SawitGuard-GNN — models for the REAL Eg9PP panel (`layer2_real/`).

Three models, head to head, exactly as on the synthetic side:

    MLPBaseline   per-tree, last census only, NO graph        (imported verbatim
                  from the repo-root `models.py` — same class, same weights count
                  logic, so the two halves of the paper cannot drift apart)
    STGNN         single-relation diffusion + temporal GRU     (n_rel = 1)
    STGNN_SID     STGNN + a trainable SI(D) head               (SEIR, degraded)

--------------------------------------------------------------------------
Change 1 — n_rel = 3  ->  n_rel = 1  (this is a code change, not a flag)
--------------------------------------------------------------------------
The synthetic simulator carries three relations: root contact, wind cone and
harvest path. Eg9PP carries ONE — root contact at r = 1.5 x planting distance.
There is no wind field and no management-path record in the source, so the other
two relations have no observable counterpart.

Root `models.py::STGNN` holds `self.attn = nn.Parameter(torch.zeros(n_rel))` and
applies `softmax(attn)` over the relation axis. With n_rel = 1 that softmax is
identically 1.0 for every value of the parameter: the gradient w.r.t. `attn` is
exactly zero, and the parameter is dead. Keeping it would (a) add an untrainable
parameter to the count and (b) imply to a reader that the model is choosing
between relations when there is nothing to choose. So relation attention is
REMOVED here, not merely configured away. The [B, W, N_REL, d] tensor shape from
INTERFACE.md is preserved; the relation axis is simply indexed, not weighted.

--------------------------------------------------------------------------
Change 2 — STGNN_SEIR -> STGNN_SID: the honest degradation
--------------------------------------------------------------------------
Eg9PP records two event times per palm: first symptom (T1S) and death (TD),
each with a censoring indicator. It does NOT record infection. The latent
exposed compartment E is therefore **entirely unobserved** — there is no column,
no proxy and no auxiliary assay from which E could be identified.

Root `STGNN_SEIR` is built around E:
    * `w_state`: hidden -> 3 logits, softmax'd into (pS, pE, pI)          105 par
    * `rates`  : 6 parameters (b_root, b_wind, b_mgmt, sigma, gamma, kappa)
    * two-stage probability
          p_fromE = pE * (1 - (1-sigma)^h)                 already exposed
          p_fromS = pS * (1 - exp(-press)) * (1-(1-sigma)^(h-1))   newly exposed
      i.e. sigma (the E->I incubation rate) is the hinge of the whole head, and
      pE is what it acts on.

What is DROPPED here and why:

  pE, and with it the 3-way softmax head `w_state`.
      Within the forecasting risk set every example is by construction status
      'A' (asymptomatic, still observed). Splitting 'A' into "already infected
      but silent" vs "truly susceptible" IS the E compartment. With only T1S and
      TD in the data that split is not identifiable from anything — a 2-way
      softmax here would be E wearing a different label, and fitting it would be
      fitting noise. It is removed rather than renamed.

  sigma (E -> I incubation rate).
      Meaningless once there is no E to leave. Removed.

  b_wind, b_mgmt.
      No wind and no harvest-path relation exists (see Change 1). Removed.

  gamma, kappa.
      Note for the record: in root `models.py::STGNN_SEIR` these two are
      unpacked from `self.rates` and then NEVER USED in `forward`. They were
      already dead parameters on the synthetic side. They are not carried over.

  R (recovered).
      Palms do not recover from Ganoderma; the absorbing state is D (death),
      which is why the compartment name is SI(D) and not SIR. D is absorbing and
      lies AFTER the forecast event (the label is "becomes S or D within h", so a
      palm that runs A->S->D inside the window is one event, not two). D
      therefore does not enter the hazard for an at-risk palm and needs no rate
      of its own. It is in the name because it is the reason there is no
      recovery term, and because a dead palm's stump stays an inoculum source —
      the neighbour signal counts S and D alike.

What SURVIVES: a single force of infection on an at-risk palm,
      force = softplus(beta_root * neighbour_pressure + lambda0)
      p(event within h censuses) = 1 - exp(-h * force)
plus the same additive GNN residual (`res_scale * gnn_logit`) as the root head.
`neighbour_pressure` is read from the DIFFUSED FEATURE tensor at the query
census, mean-reduced over the feature axis — identical in construction to root's
`Dt[:, 0].mean(-1)`. It never touches a ground-truth state.

COST OF THE DEGRADATION, stated plainly:
    root STGNN_SEIR head = w_state (105) + rates (6) + res_scale (1) = 112 par
    STGNN_SID       head = rates (2) + res_scale (1)                 =   3 par
The mechanistic head shrinks from 112 to 3 parameters. It keeps the discrete-
time survival unrolling over the horizon and the transmission-rate-on-neighbour-
pressure term; it loses the entire latent-state machinery. Consequently

    *** STGNN_SID - STGNN IS NOT COMPARABLE TO STGNN_SEIR - STGNN. ***

They are different heads answering different questions on different data. Any
table that puts those two deltas side by side is wrong. This is also stated in
INTERFACE.md rule 6 and in DATASET_CARD.md's "Batas yang dipaksakan" for the
Eg9PP panel.

MEASURED OUTCOME (h=3, 20 seeds x 2 folds): SID - STGNN = -0.029 +/- 0.019,
4/40 seeds positive, verdict NEG. The head HURTS, and the deficit grows with the
horizon (-0.012 / -0.023 / -0.029 / -0.043 for h = 1..4). That growth pattern is
also what a bad initialisation would produce, so it was tested and REFUTED — see
`run_real.py::sid_init_probe`. The deficit survives a neutral initialisation, so
it is a property of the head, not of its starting point.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# repo root, so MLPBaseline is literally the same class the synthetic half uses
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from models import MLPBaseline, count_params      # noqa: E402,F401  (re-exported)

HIDDEN = 34          # identical to root models.HIDDEN
N_REL = 1            # root contact only — see Change 1


class STGNN(nn.Module):
    """Single-relation neighbour diffusion + temporal GRU.

    F_seq : [B, W, d]        own past-only features over the window
    D_seq : [B, W, N_REL, d] neighbour-diffused features, N_REL = 1
    """

    def __init__(self, in_dim, hidden=HIDDEN, n_rel=N_REL):
        super().__init__()
        if n_rel != 1:
            raise ValueError("layer2_real is single-relation (root contact); n_rel must be 1")
        # NOTE: no `self.attn`. softmax over one relation is a constant 1.0 and the
        # parameter would receive exactly zero gradient. See Change 1 in the docstring.
        self.w_self = nn.Linear(in_dim, hidden)
        self.w_agg = nn.Linear(in_dim, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.readout = nn.Linear(hidden, 1)
        self.hidden = hidden

    def encode(self, F_seq, D_seq):
        B, W, _ = F_seq.shape
        h = torch.zeros(B, self.hidden, device=F_seq.device)
        for t in range(W):
            agg = D_seq[:, t, 0]                       # [B, d] — the one relation
            m = F.relu(self.w_self(F_seq[:, t]) + self.w_agg(agg))
            h = self.gru(m, h)
        return h

    def forward(self, F_seq, D_seq):
        return self.readout(self.encode(F_seq, D_seq)).squeeze(-1)


class STGNN_SID(STGNN):
    """STGNN + a 3-parameter trainable SI(D) head. See Change 2 in the docstring.

    NOT comparable to the synthetic STGNN_SEIR variant.
    """

    def __init__(self, in_dim, hidden=HIDDEN, n_rel=N_REL, horizon=3, base_hazard=None):
        super().__init__(in_dim, hidden, n_rel)
        # `rates = zeros` is root's convention. It is NOT neutral: softplus(0) = 0.693,
        # so the head starts at a baseline force of infection of 0.69 per census, i.e.
        # p(event) ~ 0.88 at h=3, against a true base rate of ~4.7%. The head therefore
        # opens with a large positive logit offset the optimiser has to unlearn.
        # `base_hazard` sets lambda0 so the head STARTS at that per-census hazard (the
        # mechanistic equivalent of initialising a logistic bias at the base rate). It
        # is a diagnostic knob: default None reproduces root exactly, so the headline
        # number is never quietly improved by an initialisation change.
        if base_hazard is None:
            self.rates = nn.Parameter(torch.zeros(2))   # [beta_root, lambda0]
        else:
            inv = lambda z: float(np.log(np.expm1(max(z, 1e-6))))   # noqa: E731
            self.rates = nn.Parameter(torch.tensor([inv(0.1), inv(base_hazard)]))
        self.res_scale = nn.Parameter(torch.tensor(1.0))
        self.horizon = horizon

    def forward(self, F_seq, D_seq):
        h = self.encode(F_seq, D_seq)
        gnn_logit = self.readout(h).squeeze(-1)

        beta_root, lam0 = F.softplus(self.rates).unbind(0)
        press = D_seq[:, -1, 0].mean(-1)                # neighbour pressure at query census
        force = F.softplus(beta_root * press + lam0)    # force of infection per census
        p = 1.0 - torch.exp(-float(self.horizon) * force)
        p = torch.clamp(p, 1e-4, 1 - 1e-4)
        sid_logit = torch.log(p / (1 - p))
        return sid_logit + self.res_scale * gnn_logit


def build(name, in_dim, horizon=3, base_hazard=None):
    if name == "MLPBaseline":
        return MLPBaseline(in_dim=in_dim)
    if name == "STGNN":
        return STGNN(in_dim)
    if name == "STGNN_SID":
        return STGNN_SID(in_dim, horizon=horizon, base_hazard=base_hazard)
    raise ValueError(name)


if __name__ == "__main__":
    d = 26
    for n in ("MLPBaseline", "STGNN", "STGNN_SID"):
        m = build(n, d)
        print(f"{n:12s} params={count_params(m)}")
    B, W = 8, 3
    Fs = torch.randn(B, W, d)
    Ds = torch.randn(B, W, N_REL, d)
    for n in ("MLPBaseline", "STGNN", "STGNN_SID"):
        print(n, tuple(build(n, d)(Fs, Ds).shape))
