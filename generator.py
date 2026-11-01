"""Bayesian conditional WGAN-GP scenario generator for meteorological covariates.

The generator works on day-of-year deseasonalised, standardised anomalies (training-period
climatology), so the annual cycle and seasonal variance are restored exactly when anomalies
are mapped back to physical units. This prevents the variance collapse/lost annual cycle of
the earlier GAN (Reviewer 3, comment 8).

Architecture
------------
G: per-step input [eps_t (8), z_global (8), sin/cos doy (4)] -> Bayesian linear (in)
   -> GRU(64) initialised from an MLP encoding of the previous C=16 days -> Bayesian linear (out)
   -> d anomalies. Blocks of B=64 days are generated autoregressively (block-wise).
D (critic): 1-D CNN on [context ; block] with doy channels, WGAN-GP (lambda=10).

Bayesian treatment: the two Bayesian linear layers carry mean-field Gaussian posteriors
q_theta(W) = N(m, softplus(rho)^2) with prior p(W) = N(0, sigma0^2 = 0.1), trained by
maximising E_q[-critic(G_W(z))] - KL(q||p)/N (variational parameters theta are distinct
from generator weights phi; Reviewer 1, comment 7). One weight sample is drawn per scenario,
so epistemic generator uncertainty propagates to the Monte Carlo ensemble.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

CTX, BLOCK, ZDIM, HID = 16, 64, 8, 64


def doy_feats(doy: np.ndarray) -> np.ndarray:
    w = 2 * np.pi * doy / 365.25
    return np.stack([np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)], -1).astype(np.float32)


class BayesLinear(nn.Module):
    """Mean-field Gaussian linear layer supporting one weight sample per batch element."""

    def __init__(self, n_in, n_out, prior_var=0.1, bayesian=True):
        super().__init__()
        self.bayesian = bayesian
        self.m = nn.Parameter(torch.randn(n_out, n_in) / math.sqrt(n_in))
        self.b = nn.Parameter(torch.zeros(n_out))
        self.rho = nn.Parameter(torch.full((n_out, n_in), -5.0))
        self.prior_sd = math.sqrt(prior_var)

    def sample(self, n):
        if not self.bayesian:
            return self.m.unsqueeze(0).expand(n, -1, -1)
        sd = F.softplus(self.rho)
        return self.m.unsqueeze(0) + sd.unsqueeze(0) * torch.randn(n, *self.m.shape)

    def forward(self, x, W):           # x [n, T, in], W [n, out, in]
        return torch.einsum("nti,noi->nto", x, W) + self.b

    def kl(self):
        if not self.bayesian:
            return torch.zeros(())
        sd = F.softplus(self.rho)
        p = self.prior_sd
        return (torch.log(p / sd) + (sd ** 2 + self.m ** 2) / (2 * p ** 2) - 0.5).sum()


class Generator(nn.Module):
    def __init__(self, d, bayesian=True):
        super().__init__()
        self.d = d
        self.inp = BayesLinear(2 * ZDIM + 4, HID, bayesian=bayesian)
        self.ctx = nn.Sequential(nn.Linear(CTX * d, HID), nn.Tanh())
        self.gru = nn.GRU(HID, HID, batch_first=True)
        self.out = BayesLinear(HID, d, bayesian=bayesian)

    def sample_weights(self, n):
        return self.inp.sample(n), self.out.sample(n)

    def forward(self, ctx, doyf, weights=None):
        n, T = doyf.shape[:2]
        if weights is None:
            weights = self.sample_weights(n)
        eps = torch.randn(n, T, ZDIM)
        zg = torch.randn(n, 1, ZDIM).expand(n, T, ZDIM)
        h = torch.relu(self.inp(torch.cat([eps, zg, doyf], -1), weights[0]))
        h0 = self.ctx(ctx.reshape(n, -1)).unsqueeze(0)
        o, _ = self.gru(h, h0)
        return self.out(o, weights[1])

    def kl(self):
        return self.inp.kl() + self.out.kl()


class Critic(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d + 4, 64, 5, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 64, 5, padding=2, stride=2), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 64, 5, padding=2, stride=2), nn.LeakyReLU(0.2))
        self.lin = nn.Linear(64 * 2, 1)

    def forward(self, seq, doyf):       # seq [n, CTX+BLOCK, d]
        h = self.net(torch.cat([seq, doyf], -1).transpose(1, 2))
        feat = torch.cat([h.mean(-1), h.std(-1)], -1)
        return self.lin(feat).squeeze(-1)


class ScenarioGenerator:
    def __init__(self, bayesian=True, iters=2000, batch=64, n_critic=5, gp=10.0, lr=2e-4, seed=0):
        self.bayesian, self.iters, self.batch = bayesian, iters, batch
        self.n_critic, self.gp, self.lr, self.seed = n_critic, gp, lr, seed

    def fit(self, anom: pd.DataFrame, verbose=False):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        A = anom.values.astype(np.float32)
        D = doy_feats(anom.index.dayofyear.values)
        self.cols, d = list(anom.columns), A.shape[1]
        L = CTX + BLOCK
        starts = np.arange(0, len(A) - L)
        G, C = Generator(d, self.bayesian), Critic(d)
        oG = torch.optim.Adam(G.parameters(), self.lr, betas=(0.5, 0.9))
        oC = torch.optim.Adam(C.parameters(), self.lr, betas=(0.5, 0.9))
        At, Dt = torch.from_numpy(A), torch.from_numpy(D)
        idx = torch.arange(L)
        n_win = len(starts)
        t0 = time.time()
        self.hist = []
        for it in range(self.iters):
            for _ in range(self.n_critic):
                s = torch.from_numpy(np.random.choice(starts, self.batch))
                win = At[s[:, None] + idx]
                dw = Dt[s[:, None] + idx]
                with torch.no_grad():
                    fake = G(win[:, :CTX], dw[:, CTX:])
                fake_seq = torch.cat([win[:, :CTX], fake], 1)
                a = torch.rand(self.batch, 1, 1)
                mix = (a * win + (1 - a) * fake_seq).requires_grad_(True)
                cm = C(mix, dw)
                g = torch.autograd.grad(cm.sum(), mix, create_graph=True)[0]
                gpen = ((g.reshape(self.batch, -1).norm(dim=1) - 1) ** 2).mean()
                lossC = C(fake_seq, dw).mean() - C(win, dw).mean() + self.gp * gpen
                oC.zero_grad(); lossC.backward(); oC.step()
            s = torch.from_numpy(np.random.choice(starts, self.batch))
            win, dw = At[s[:, None] + idx], Dt[s[:, None] + idx]
            fake = G(win[:, :CTX], dw[:, CTX:])
            lossG = -C(torch.cat([win[:, :CTX], fake], 1), dw).mean() + G.kl() / n_win
            oG.zero_grad(); lossG.backward(); oG.step()
            if it % 100 == 0:
                self.hist.append((it, float(lossC), float(lossG)))
                if verbose:
                    print(f"  it {it} lossC {float(lossC):.3f} lossG {float(lossG):.3f} {time.time()-t0:.0f}s")
        self.G = G.eval()
        self.last_ctx = A[-CTX:]
        return self

    @torch.no_grad()
    def sample(self, dates: pd.DatetimeIndex, n: int, ctx: np.ndarray | None = None, seed=0) -> np.ndarray:
        """n scenarios of anomalies over `dates` (contiguous days following the context)."""
        torch.manual_seed(seed)
        ctx = self.last_ctx if ctx is None else ctx
        cur = torch.from_numpy(np.repeat(ctx[None], n, 0).astype(np.float32))
        W = self.G.sample_weights(n)              # one posterior draw per scenario
        D = torch.from_numpy(doy_feats(dates.dayofyear.values))
        out = []
        for k in range(0, len(dates), BLOCK):
            dk = D[k:k + BLOCK].unsqueeze(0).expand(n, -1, -1)
            blk = self.G(cur, dk, W)
            out.append(blk)
            cur = torch.cat([cur, blk], 1)[:, -CTX:]
        return torch.cat(out, 1).numpy()[:, :len(dates)]


def bootstrap_scenarios(anom: pd.DataFrame, dates: pd.DatetimeIndex, n: int, seed=0) -> np.ndarray:
    """Ablation A1 / reference: historical resampling of whole training years (doy aligned)."""
    rng = np.random.default_rng(seed)
    years = sorted(set(anom.index.year))
    years = [y for y in years if (anom.index.year == y).sum() >= 360]
    by_doy = {y: anom[anom.index.year == y].groupby(anom[anom.index.year == y].index.dayofyear).mean()
              for y in years}
    out = np.empty((n, len(dates), anom.shape[1]), np.float32)
    doys = np.minimum(dates.dayofyear.values, 365)
    for i in range(n):
        for yr in sorted(set(dates.year)):
            m = dates.year == yr
            src = by_doy[rng.choice(years)]
            out[i, m] = src.reindex(doys[m]).ffill().bfill().values
    return out


def var1_scenarios(anom: pd.DataFrame, dates, n, seed=0):
    """Reference generator: Gaussian VAR(1) on anomalies."""
    rng = np.random.default_rng(seed)
    X = anom.values
    Y, Z = X[1:], X[:-1]
    A = np.linalg.lstsq(Z, Y, rcond=None)[0]
    E = Y - Z @ A
    Lc = np.linalg.cholesky(np.cov(E.T) + 1e-9 * np.eye(X.shape[1]))
    out = np.empty((n, len(dates), X.shape[1]))
    cur = np.repeat(X[-1][None], n, 0)
    for t in range(len(dates)):
        cur = cur @ A + rng.standard_normal((n, X.shape[1])) @ Lc.T
        out[:, t] = cur
    return out
