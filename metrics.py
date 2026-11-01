"""Probabilistic and point scores. All scores are computed on GHI in Wh m-2 day-1
unless stated otherwise (CRPS carries the unit of the observed variable)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def crps_ensemble(obs: np.ndarray, ens: np.ndarray) -> np.ndarray:
    """Exact CRPS of the empirical ensemble CDF (energy form), per time step.

    obs: [T], ens: [M, T].  CRPS = E|X-y| - 0.5 E|X-X'|, computed in O(M log M)
    with the sorted-sample identity, identical to properscoring.crps_ensemble.
    """
    ens = np.sort(ens, axis=0)
    m = ens.shape[0]
    term1 = np.mean(np.abs(ens - obs[None, :]), axis=0)
    i = np.arange(1, m + 1)[:, None]
    term2 = np.sum((2 * i - m - 1) * ens, axis=0) / (m * m)
    return term1 - term2


def interval(ens, level=0.90):
    a = (1 - level) / 2
    return np.quantile(ens, a, axis=0), np.quantile(ens, 1 - a, axis=0)


def scores(obs: np.ndarray, ens: np.ndarray, level: float = 0.90) -> dict:
    ok = np.isfinite(obs)
    obs, ens = obs[ok], ens[:, ok]
    lo, hi = interval(ens, level)
    med = np.median(ens, 0)
    mean = ens.mean(0)
    rng = obs.max() - obs.min()
    return dict(
        CRPS=float(crps_ensemble(obs, ens).mean()),
        MAE=float(np.mean(np.abs(obs - med))),
        RMSE=float(np.sqrt(np.mean((obs - mean) ** 2))),
        PICP=float(np.mean((obs >= lo) & (obs <= hi)) * 100),
        MPIW=float(np.mean(hi - lo)),
        PINAW=float(np.mean(hi - lo) / rng),
        n=int(ok.sum()),
    )


def monthly_scores(obs, ens, index, level=0.90):
    """Scores of monthly-mean GHI (planning-relevant aggregation)."""
    import pandas as pd
    o = pd.Series(obs, index=index).resample("MS").mean()
    e = pd.DataFrame(ens.T, index=index).resample("MS").mean().T.values
    cnt = pd.Series(np.isfinite(obs), index=index).resample("MS").sum().values
    keep = cnt >= 20
    return scores(o.values[keep], e[:, keep], level)


def pit(obs, ens):
    ok = np.isfinite(obs)
    return np.mean(ens[:, ok] <= obs[ok][None, :], axis=0)


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1):
    """DM test on loss differential d = a - b with Newey-West HAC variance.
    Negative statistic: model a has lower loss."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    d = d[np.isfinite(d)]
    n = len(d)
    dbar = d.mean()
    lag = int(np.floor(4 * (n / 100) ** (2 / 9))) if h == 1 else h - 1
    gamma0 = np.mean((d - dbar) ** 2)
    var = gamma0
    for k in range(1, lag + 1):
        g = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        var += 2 * (1 - k / (lag + 1)) * g
    stat = dbar / np.sqrt(var / n)
    p = 2 * (1 - stats.norm.cdf(abs(stat)))
    return float(stat), float(p)
