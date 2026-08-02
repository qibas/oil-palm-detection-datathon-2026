"""SawitGuard-GNN — models.

Guards:
  * No model ever sees ground-truth S/E/I/R state as input — only spectral
    features and their neighbour aggregates (diffused features).
  * STGNN and STGNN_SEIR are identical except the SEIR head (~+106 params).
  * MLPBaseline uses no graph at all — the fixed field baseline.
"""
import torch, torch.nn as nn, torch.nn.functional as F

HIDDEN = 34


class MLPBaseline(nn.Module):
    """Per-tree, no graph. Input = node feature [pca, delta] at the query cycle."""
    def __init__(self, in_dim=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU(),
                                 nn.Linear(32, 1))

    def forward(self, F_seq, D_seq):
        return self.net(F_seq[:, -1, :]).squeeze(-1)   # use last cycle only


class STGNN(nn.Module):
    """Multi-relation diffusion + relation attention + temporal GRU."""
    def __init__(self, in_dim=32, hidden=HIDDEN, n_rel=3):
        super().__init__()
        self.attn = nn.Parameter(torch.zeros(n_rel))       # relation attention logits
        self.w_self = nn.Linear(in_dim, hidden)
        self.w_agg = nn.Linear(in_dim, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.readout = nn.Linear(hidden, 1)
        self.hidden = hidden

    def encode(self, F_seq, D_seq):
        B, W, _ = F_seq.shape
        a = torch.softmax(self.attn, 0)                    # [n_rel]
        h = torch.zeros(B, self.hidden, device=F_seq.device)
        for t in range(W):
            agg = (D_seq[:, t] * a[None, :, None]).sum(1)  # [B,in_dim] relation-weighted
            m = F.relu(self.w_self(F_seq[:, t]) + self.w_agg(agg))
            h = self.gru(m, h)
        return h

    def forward(self, F_seq, D_seq):
        return self.readout(self.encode(F_seq, D_seq)).squeeze(-1)


class STGNN_SEIR(STGNN):
    """STGNN + a trainable networked-SEIR head unrolled over the horizon.

    The head reads a soft (S,E,I) state from the GNN embedding and estimates the
    probability of turning symptomatic within h using learnable rates and a
    neighbour-pressure term derived from DIFFUSED SPECTRAL features (never true
    states). The GNN output is a data-driven residual on top.
    """
    def __init__(self, in_dim=32, hidden=HIDDEN, n_rel=3, horizon=3):
        super().__init__(in_dim, hidden, n_rel)
        self.w_state = nn.Linear(hidden, 3)                # -> (S,E,I) logits (+105)
        self.rates = nn.Parameter(torch.zeros(6))          # b_root,b_wind,b_mgmt,sigma,gamma,kappa
        self.res_scale = nn.Parameter(torch.tensor(1.0))
        self.horizon = horizon

    def forward(self, F_seq, D_seq):
        h = self.encode(F_seq, D_seq)
        gnn_logit = self.readout(h).squeeze(-1)
        pS, pE, pI = torch.softmax(self.w_state(h), -1).unbind(-1)
        b = F.softplus(self.rates)
        b_root, b_wind, b_mgmt, sigma, gamma, kappa = b
        sigma = torch.clamp(sigma, 0.01, 0.99)
        # neighbour pressure from diffused spectra at the query cycle (mean-reduced)
        Dt = D_seq[:, -1]                                   # [B,n_rel,in_dim]
        press = F.softplus(b_root * Dt[:, 0].mean(-1)
                           + b_wind * Dt[:, 1].mean(-1)
                           + b_mgmt * Dt[:, 2].mean(-1))
        h_ = self.horizon
        p_fromE = pE * (1 - (1 - sigma) ** h_)
        p_fromS = pS * (1 - torch.exp(-press)) * (1 - (1 - sigma) ** max(h_ - 1, 1))
        p = torch.clamp(p_fromE + p_fromS + 1e-4, 1e-4, 1 - 1e-4)
        seir_logit = torch.log(p / (1 - p))
        return seir_logit + self.res_scale * gnn_logit


def count_params(m):
    return sum(p.numel() for p in m.parameters())
