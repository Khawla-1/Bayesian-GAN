"""Forecasting models.

1. CondPatchTST  -- proposed conditional model p(z_t | x_{t-L+1:t}, doy_t): a channel-
   independent PatchTST encoder over a 32-day window of the BN-1-selected covariate
   anomalies, with a two-headed Gaussian output (mu_t, log sigma_t) for the standardised
   logit-Kt anomaly z_t. Trained only on observed pairs before the origin (Gaussian NLL).
2. LinearGauss   -- ablation A4: heteroscedastic linear-Gaussian model on the same inputs.
3. UniPatchTST   -- univariate PatchTST baselines (direct 365-day block from the previous
   368 days): point (MSE), MC-dropout and deep-ensemble (Gaussian NLL) variants.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn

LOG2PI = math.log(2 * math.pi)


def gauss_nll(mu, logsig, y):
    # average negative log-likelihood per sample, 0.5*log(2*pi) inside the sum (R1, minor 2)
    return (logsig + 0.5 * ((y - mu) / torch.exp(logsig)) ** 2 + 0.5 * LOG2PI).mean()


def doy_feats(doy):
    w = 2 * np.pi * np.asarray(doy) / 365.25
    return np.stack([np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)], -1).astype(np.float32)


class PatchEncoder(nn.Module):
    def __init__(self, L, patch, d_model=32, heads=4, layers=2, ff=64, dropout=0.1):
        super().__init__()
        self.patch, self.P = patch, L // patch
        self.emb = nn.Linear(patch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.P, d_model))
        enc = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.out_dim = self.P * d_model

    def forward(self, x):                  # x [N, L, C] -> [N, C, P*d]
        N, L, C = x.shape
        p = x.transpose(1, 2).reshape(N * C, self.P, self.patch)
        h = self.enc(self.emb(p) + self.pos)
        return h.reshape(N, C, -1)


class CondNet(nn.Module):
    def __init__(self, k, L=32, patch=8, dropout=0.1):
        super().__init__()
        self.enc = PatchEncoder(L, patch, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(k * self.enc.out_dim + 4, 64), nn.ReLU(), nn.Dropout(dropout))
        self.mu, self.ls = nn.Linear(64, 1), nn.Linear(64, 1)

    def forward(self, x, d):
        h = self.head(torch.cat([self.enc(x).flatten(1), d], -1))
        return self.mu(h).squeeze(-1), self.ls(h).squeeze(-1).clamp(-5, 3)


def windows(A: np.ndarray, L: int) -> np.ndarray:
    """A [T, k] -> [T-L+1, L, k] sliding windows (window i ends at index i+L-1)."""
    from numpy.lib.stride_tricks import sliding_window_view
    return np.ascontiguousarray(sliding_window_view(A, L, axis=0).transpose(0, 2, 1))


def _train(net, Xs, y, Xv, yv, epochs, lr, batch, patience, seed, loss="nll"):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(y)
    best, best_state, bad = np.inf, None, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            out = net(*[X[b] for X in Xs])
            l = gauss_nll(*out, y[b]) if loss == "nll" else ((out - y[b]) ** 2).mean()
            opt.zero_grad(); l.backward(); opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            out = net(*Xv)
            v = float(gauss_nll(*out, yv) if loss == "nll" else ((out - yv) ** 2).mean())
        if v < best - 1e-4:
            best, best_state, bad = v, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return net.eval(), best


class CondPatchTST:
    L = 32

    def __init__(self, seed=0, epochs=60, lr=1e-3, batch=256, patience=8, dropout=0.1):
        self.seed, self.epochs, self.lr, self.batch, self.patience = seed, epochs, lr, batch, patience
        self.dropout = dropout

    def fit(self, Xanom: np.ndarray, doy: np.ndarray, z: np.ndarray, n_val=365):
        """Xanom [T, k] covariate anomalies, z [T] target anomalies (observed days)."""
        W = windows(Xanom, self.L)
        D = doy_feats(doy[self.L - 1:])
        y = z[self.L - 1:].astype(np.float32)
        ok = np.isfinite(y)
        W, D, y = W[ok], D[ok], y[ok]
        tr, va = slice(0, len(y) - n_val), slice(len(y) - n_val, len(y))
        t = lambda a: torch.from_numpy(np.asarray(a, np.float32))
        torch.manual_seed(self.seed)
        self.net = CondNet(Xanom.shape[1], self.L, dropout=self.dropout)
        self.net, self.val_nll = _train(self.net, (t(W[tr]), t(D[tr])), t(y[tr]), (t(W[va]), t(D[va])), t(y[va]),
                                        self.epochs, self.lr, self.batch, self.patience, self.seed)
        with torch.no_grad():
            mu, ls = self.predict_windows(W, D)
        self.train_resid = (y - mu)
        self.train_resid_idx = np.where(ok)[0] + self.L - 1
        return self

    @torch.no_grad()
    def predict_windows(self, W, D, chunk=20000):
        mus, lss = [], []
        for i in range(0, len(W), chunk):
            m, l = self.net(torch.from_numpy(np.asarray(W[i:i + chunk], np.float32)),
                            torch.from_numpy(np.asarray(D[i:i + chunk], np.float32)))
            mus.append(m.numpy()); lss.append(l.numpy())
        return np.concatenate(mus), np.concatenate(lss)

    def predict_paths(self, hist: np.ndarray, paths: np.ndarray, doy: np.ndarray):
        """hist [L-1, k] observed anomalies before origin; paths [S, T, k]; -> mu, sigma [S, T]."""
        S, T, k = paths.shape
        full = np.concatenate([np.repeat(hist[None], S, 0), paths], 1)
        W = np.stack([windows(full[s], self.L) for s in range(S)]).reshape(S * T, self.L, k)
        D = np.tile(doy_feats(doy), (S, 1))
        mu, ls = self.predict_windows(W, D)
        return mu.reshape(S, T), np.exp(ls).reshape(S, T)


class LinearGauss(CondPatchTST):
    """Ablation A4: heteroscedastic linear-Gaussian model (same inputs, no PatchTST)."""

    class Net(nn.Module):
        def __init__(self, k, L):
            super().__init__()
            self.mu = nn.Linear(k * L + 4, 1)
            self.ls = nn.Linear(k * L + 4, 1)

        def forward(self, x, d):
            h = torch.cat([x.flatten(1), d], -1)
            return self.mu(h).squeeze(-1), self.ls(h).squeeze(-1).clamp(-5, 3)

    def fit(self, Xanom, doy, z, n_val=365):
        W = windows(Xanom, self.L)
        D = doy_feats(doy[self.L - 1:])
        y = z[self.L - 1:].astype(np.float32)
        ok = np.isfinite(y)
        W, D, y = W[ok], D[ok], y[ok]
        tr, va = slice(0, len(y) - n_val), slice(len(y) - n_val, len(y))
        t = lambda a: torch.from_numpy(np.asarray(a, np.float32))
        torch.manual_seed(self.seed)
        self.net = self.Net(Xanom.shape[1], self.L)
        self.net, self.val_nll = _train(self.net, (t(W[tr]), t(D[tr])), t(y[tr]), (t(W[va]), t(D[va])), t(y[va]),
                                        self.epochs, 3e-3, self.batch, self.patience, self.seed)
        with torch.no_grad():
            mu, _ = self.predict_windows(W, D)
        self.train_resid = y - mu
        self.train_resid_idx = np.where(ok)[0] + self.L - 1
        return self


# ----------------------------------------------------------------------------------------
# Univariate PatchTST baselines (direct 365-day blocks, recursive across lead years)
# ----------------------------------------------------------------------------------------
class UniNet(nn.Module):
    def __init__(self, L=368, H=365, patch=16, gaussian=True, dropout=0.1):
        super().__init__()
        self.enc = PatchEncoder(L, patch, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.gaussian, self.H = gaussian, H
        self.out = nn.Linear(self.enc.out_dim + 4, H * (2 if gaussian else 1))

    def forward(self, x, d):
        h = torch.cat([self.drop(self.enc(x.unsqueeze(-1)).flatten(1)), d], -1)
        o = self.out(h)
        if self.gaussian:
            return o[:, :self.H], o[:, self.H:].clamp(-5, 3)
        return o


class UniPatchTST:
    L, H = 368, 365

    def __init__(self, kind="point", seed=0, members=5, epochs=40, stride=5):
        self.kind, self.seed, self.members, self.epochs, self.stride = kind, seed, members, epochs, stride

    def fit(self, z: np.ndarray, doy: np.ndarray, n_val_windows=60):
        z = z.astype(np.float32)
        T = len(z)
        starts = np.arange(0, T - self.L - self.H + 1, self.stride)
        X = np.stack([z[s:s + self.L] for s in starts])
        Y = np.stack([z[s + self.L:s + self.L + self.H] for s in starts])
        D = doy_feats(doy[starts + self.L])
        t = lambda a: torch.from_numpy(np.asarray(a, np.float32))
        tr, va = slice(0, len(X) - n_val_windows), slice(len(X) - n_val_windows, len(X))
        n_models = self.members if self.kind == "ensemble" else 1
        self.nets = []
        for m in range(n_models):
            sd = self.seed * 100 + m
            torch.manual_seed(sd)
            net = UniNet(gaussian=self.kind != "point")
            net, _ = _train(net, (t(X[tr]), t(D[tr])), t(Y[tr]), (t(X[va]), t(D[va])), t(Y[va]),
                            self.epochs, 1e-3, 64, 6, sd, loss="mse" if self.kind == "point" else "nll")
            self.nets.append(net)
        return self

    @torch.no_grad()
    def sample(self, ctx: np.ndarray, doy_first: np.ndarray, n_years: int, n_paths: int = 200, seed=0):
        """Recursive sampling of n_years * 365 days. doy_first: doy of first day of each lead year."""
        g = torch.Generator().manual_seed(seed)
        cur = torch.from_numpy(np.repeat(ctx[-self.L:][None], n_paths, 0).astype(np.float32))
        out = []
        for yr in range(n_years):
            d = torch.from_numpy(np.repeat(doy_feats([doy_first[yr]]), n_paths, 0))
            if self.kind == "point":
                self.nets[0].eval()
                y = self.nets[0](cur[:1], d[:1]).expand(n_paths, -1)
            elif self.kind == "mcdropout":
                self.nets[0].train()                       # dropout active at inference
                mu, ls = self.nets[0](cur, d)
                y = mu + torch.exp(ls) * torch.randn(mu.shape, generator=g)
                self.nets[0].eval()
            else:
                mem = torch.randint(len(self.nets), (n_paths,), generator=g)
                mu = torch.empty(n_paths, self.H); ls = torch.empty(n_paths, self.H)
                for m, net in enumerate(self.nets):
                    sel = mem == m
                    if sel.any():
                        mu[sel], ls[sel] = net(cur[sel], d[sel])
                y = mu + torch.exp(ls) * torch.randn(mu.shape, generator=g)
            out.append(y)
            cur = torch.cat([cur, y], 1)[:, -self.L:]
        return torch.cat(out, 1).numpy()
