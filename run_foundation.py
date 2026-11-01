#!/usr/bin/env python3
"""Time-series foundation-model baselines on the same rolling origins (paper Sec. IV-H).

* Chronos-Bolt (small), zero-shot: quantile forecasts of Kt (levels 0.1..0.9).
* TimesFM 2.5 (200M), zero-shot: continuous quantile head (deciles 0.1..0.9), 365-day
  blocks generated recursively (the head supports horizons <= 1024).
* MOMENT-1 (small), linear probing: the frozen encoder embeds the previous 512 days of the
  standardised logit-Kt anomaly; a ridge head (fitted on training windows only) maps the
  embedding to the next 365 days; lead years are produced recursively (point forecast).

Quantile forecasts are turned into M=200-member ensembles by sampling the piecewise-linear
quantile function between the outer levels, so CRPS/PICP are computed exactly as for the
other models. Results are written to results/foundation/<origin>.pkl and merged by analysis.py.
Requires network access to huggingface.co and hf.co (model weights).
"""
from __future__ import annotations

import argparse
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ghibgmc.data import Deseasonaliser, expit, load_dataset, logit
from ghibgmc.metrics import crps_ensemble, monthly_scores, scores
from run_experiments import LEADS, ORIGINS

warnings.filterwarnings("ignore")
torch.set_num_threads(int(os.environ.get("NTHREADS", "2")))
M = 200


PHI = 0.5   # lag-1 autocorrelation of the latent copula; set per origin in main()


def q_to_ens(levels, q, seed=0):
    """q [n_levels, H] -> temporally coherent ensemble [M, H].

    Chronos-Bolt and TimesFM return marginal quantiles only (0.1..0.9). (i) The 0.01 and
    0.99 quantiles are added by Gaussian tail extrapolation from the median and the outer
    deciles (otherwise 90% coverage would be capped at 80%). (ii) Members are drawn through
    a Gaussian AR(1) copula whose lag-1 correlation PHI equals that of the pre-origin target
    anomalies, so that monthly aggregates of the ensemble have a realistic spread."""
    from scipy.stats import norm
    levels = np.asarray(levels, float)
    med = np.array([np.interp(0.5, levels, q[:, t]) for t in range(q.shape[1])])
    lo = med + (q[0] - med) * (-2.3263) / (-1.2816) if levels[0] == 0.1 else q[0]
    hi = med + (q[-1] - med) * 2.3263 / 1.2816 if levels[-1] == 0.9 else q[-1]
    L = np.concatenate([[0.01], levels, [0.99]])
    Q = np.vstack([lo, q, hi])
    rng = np.random.default_rng(seed)
    H = Q.shape[1]
    e = np.empty((M, H)); e[:, 0] = rng.standard_normal(M)
    for t in range(1, H):
        e[:, t] = PHI * e[:, t - 1] + np.sqrt(1 - PHI ** 2) * rng.standard_normal(M)
    u = 0.01 + 0.98 * norm.cdf(e)
    return np.stack([np.interp(u[:, t], L, Q[:, t]) for t in range(H)], 1)


def chronos_fc(ctx, H):
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cpu")
    lv = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    q, _ = pipe.predict_quantiles(torch.tensor(ctx[-2048:], dtype=torch.float32)[None],
                                  prediction_length=H, quantile_levels=lv)
    return q_to_ens(np.array(lv), q[0].numpy().T)


def timesfm_fc(ctx, H):
    import timesfm
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    m.compile(timesfm.ForecastConfig(max_context=2048, max_horizon=365, normalize_inputs=True,
                                     use_continuous_quantile_head=True, fix_quantile_crossing=True))
    # the continuous quantile head supports horizons <= 1024: forecast 365-day blocks
    # recursively, extending the context with the block median
    cur, qs = np.asarray(ctx, float), []
    for k in range(int(np.ceil(H / 365))):
        _, qt = m.forecast(horizon=365, inputs=[cur[-2048:]])
        q = np.asarray(qt)[0][:, 1:10].T      # [9, 365]: deciles 0.1..0.9 (index 0 is the mean)
        qs.append(q); cur = np.concatenate([cur, q[4]])
    return q_to_ens(np.linspace(0.1, 0.9, 9), np.concatenate(qs, 1)[:, :H])


class MomentProbe:
    L, Hb = 512, 365

    def __init__(self):
        from momentfm import MOMENTPipeline
        self.m = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-small", model_kwargs={"task_name": "embedding"})
        self.m.init(); self.m.eval()

    @torch.no_grad()
    def embed(self, X):
        out = []
        for i in range(0, len(X), 32):
            out.append(self.m(x_enc=torch.tensor(X[i:i + 32], dtype=torch.float32)[:, None]).embeddings.numpy())
        return np.concatenate(out)

    def fit(self, z):
        st = np.arange(0, len(z) - self.L - self.Hb + 1, 7)
        X = np.stack([z[s:s + self.L] for s in st]); Y = np.stack([z[s + self.L:s + self.L + self.Hb] for s in st])
        E = self.embed(X)
        from sklearn.linear_model import RidgeCV
        self.head = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(E, Y)
        return self

    def predict(self, z, n_years):
        cur, out = z[-self.L:].copy(), []
        for _ in range(n_years):
            y = self.head.predict(self.embed(cur[None]))[0]
            out.append(y); cur = np.concatenate([cur, y])[-self.L:]
        return np.concatenate(out)[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", default="../data/RjimMaatoug.csv")
    ap.add_argument("--met", default="../data/dataset1000.xlsx")
    ap.add_argument("--out", default="results")
    ap.add_argument("--origins", nargs="*", default=ORIGINS)
    a = ap.parse_args()
    out = Path(a.out) / "foundation"; out.mkdir(parents=True, exist_ok=True)
    df = load_dataset(a.cams, a.met)
    END = df.index[-1]
    for origin in a.origins:
        o = pd.Timestamp(origin)
        fut = pd.date_range(o + pd.Timedelta(days=1), min(o + pd.Timedelta(days=LEADS * 365), END))
        n_lead = int(np.ceil(len(fut) / 365))
        blocks = [np.arange(l * 365, min((l + 1) * 365, len(fut))) for l in range(n_lead)]
        tr = df.loc[:o]
        obs, toa = df.loc[fut, "GHI"].values, df.loc[fut, "TOA"].values
        z = pd.DataFrame({"z": logit(tr["Kt_filled"].values)}, index=tr.index)
        zds = Deseasonaliser().fit(z); za = zds.transform(z)["z"].values
        global PHI
        PHI = float(np.corrcoef(za[:-1], za[1:])[0, 1])
        ens = {}
        for lab, fn in [("Chronos-Bolt (zero-shot)", lambda: chronos_fc(tr["Kt_filled"].values, len(fut))),
                        ("TimesFM 2.5 (zero-shot)", lambda: timesfm_fc(tr["Kt_filled"].values, len(fut))),
                        ("MOMENT (linear probe)", lambda: expit(zds.mean(fut, "z")[None] + zds.std(fut, "z")[None] *
                                                                MomentProbe().fit(za).predict(za, n_lead)[:, :len(fut)]))]:
            try:
                ens[lab] = np.clip(fn(), 0, 1)
                print(origin, lab, "ok")
            except Exception as e:  # noqa: BLE001
                print(origin, lab, "FAILED:", type(e).__name__, str(e)[:150])
        rows, daily = [], {}
        for name, kt in ens.items():
            ghi = kt * toa[None]
            for l, b in enumerate(blocks):
                sc = scores(obs[b], ghi[:, b]); ms = monthly_scores(obs[b], ghi[:, b], fut[b])
                rows.append(dict(model=name, seed=0, origin=origin[:4], lead=l + 1, **sc,
                                 **{f"m_{k}": v for k, v in ms.items()}))
            ok = np.isfinite(obs); c = np.full(len(fut), np.nan); c[ok] = crps_ensemble(obs[ok], ghi[:, ok])
            daily[name] = c
        pickle.dump(dict(rows=rows, daily_crps=daily), open(out / f"{origin[:4]}.pkl", "wb"))


if __name__ == "__main__":
    main()
