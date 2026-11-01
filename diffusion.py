"""Conditional denoising-diffusion baseline (compact re-implementation in the spirit of
CSDI / TimeGrad; not the authors' official code). A DDPM (T=50, linear betas) generates
the next 365-day block of the standardised logit-Kt anomaly conditioned on the previous
368 days and on the day of year; lead years beyond the first are sampled recursively."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .forecaster import doy_feats


class ResBlock(nn.Module):
    def __init__(self, ch, dil):
        super().__init__()
        self.conv = nn.Conv1d(ch, 2 * ch, 3, padding=dil, dilation=dil)
        self.cond = nn.Linear(ch, 2 * ch)
        self.out = nn.Conv1d(ch, ch, 1)

    def forward(self, x, c):
        h = self.conv(x) + self.cond(c).unsqueeze(-1)
        a, b = h.chunk(2, 1)
        return x + self.out(torch.tanh(a) * torch.sigmoid(b))


class Denoiser(nn.Module):
    def __init__(self, H=365, L=368, ch=32, T=50):
        super().__init__()
        self.inp = nn.Conv1d(1 + 4, ch, 1)
        self.ctx = nn.Sequential(nn.Conv1d(1, 16, 7, stride=4, padding=3), nn.GELU(),
                                 nn.Conv1d(16, 16, 7, stride=4, padding=3), nn.GELU(),
                                 nn.AdaptiveAvgPool1d(8), nn.Flatten(), nn.Linear(128, ch))
        self.temb = nn.Embedding(T, ch)
        self.blocks = nn.ModuleList([ResBlock(ch, d) for d in (1, 2, 4, 8, 16, 32)])
        self.final = nn.Conv1d(ch, 1, 1)

    def forward(self, x, t, ctx, dfeat):    # x [N,H], ctx [N,L], dfeat [N,H,4]
        h = self.inp(torch.cat([x.unsqueeze(1), dfeat.transpose(1, 2)], 1))
        c = self.temb(t) + self.ctx(ctx.unsqueeze(1))
        for b in self.blocks:
            h = b(h, c)
        return self.final(h).squeeze(1)


class DiffusionForecaster:
    L, H, T = 368, 365, 50

    def __init__(self, seed=0, steps=3000, batch=32, stride=3):
        self.seed, self.steps, self.batch, self.stride = seed, steps, batch, stride
        self.betas = torch.linspace(1e-4, 0.3, self.T)
        self.abar = torch.cumprod(1 - self.betas, 0)

    def fit(self, z, dates):
        torch.manual_seed(self.seed)
        z = z.astype(np.float32)
        starts = np.arange(0, len(z) - self.L - self.H + 1, self.stride)
        X = torch.from_numpy(np.stack([z[s:s + self.L] for s in starts]))
        Y = torch.from_numpy(np.stack([z[s + self.L:s + self.L + self.H] for s in starts]))
        Dd = torch.from_numpy(np.stack([doy_feats(dates[s + self.L:s + self.L + self.H].dayofyear.values)
                                        for s in starts]))
        self.net = Denoiser(self.H, self.L, T=self.T)
        opt = torch.optim.Adam(self.net.parameters(), 1e-3)
        g = torch.Generator().manual_seed(self.seed)
        for it in range(self.steps):
            b = torch.randint(len(X), (self.batch,), generator=g)
            t = torch.randint(self.T, (self.batch,), generator=g)
            e = torch.randn(self.batch, self.H, generator=g)
            ab = self.abar[t].unsqueeze(-1)
            xt = ab.sqrt() * Y[b] + (1 - ab).sqrt() * e
            loss = ((self.net(xt, t, X[b], Dd[b]) - e) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        self.net.eval()
        return self

    @torch.no_grad()
    def _sample_block(self, ctx, dfeat, g):
        n = ctx.shape[0]
        x = torch.randn(n, self.H, generator=g)
        for t in reversed(range(self.T)):
            tt = torch.full((n,), t, dtype=torch.long)
            eps = self.net(x, tt, ctx, dfeat)
            a, ab, b = 1 - self.betas[t], self.abar[t], self.betas[t]
            x = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
            if t > 0:
                x = x + b.sqrt() * torch.randn(n, self.H, generator=g)
        return x

    @torch.no_grad()
    def sample(self, ctx, future_dates, n_years, n_paths=200, seed=0):
        g = torch.Generator().manual_seed(seed)
        cur = torch.from_numpy(np.repeat(ctx[-self.L:][None], n_paths, 0).astype(np.float32))
        out = []
        for yr in range(n_years):
            dd = future_dates[yr * self.H:(yr + 1) * self.H].dayofyear.values
            if len(dd) < self.H:
                dd = np.concatenate([dd, (dd[-1] + np.arange(1, self.H - len(dd) + 1) - 1) % 365 + 1])
            df = torch.from_numpy(np.repeat(doy_feats(dd)[None], n_paths, 0))
            y = self._sample_block(cur, df, g)
            out.append(y)
            cur = torch.cat([cur, y], 1)[:, -self.L:]
        return torch.cat(out, 1).numpy()
