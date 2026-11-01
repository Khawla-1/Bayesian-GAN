#!/usr/bin/env python3
"""Rolling-origin (walk-forward) evaluation of the Bayesian-GAN-MC framework and baselines.

Protocol (paper Sec. V-A)
  * Origins: 31 Dec of 2015..2022 (8 folds, expanding window). Each fold uses ONLY data up
    to its origin for every fitted object (covariate transforms, BN-1, generator, forecaster,
    drift model, baselines). Hyper-parameters were fixed a priori (no tuning on test folds).
  * Lead blocks: consecutive 365-day blocks after the origin (lead 1..4), truncated at the
    end of the record (2023-11-30). Lead-l scores therefore average over 8, 7, 6, 5 folds.
  * Scores on daily GHI (Wh m-2 day-1): CRPS, MAE (median), RMSE (mean), PICP/PINAW (90%),
    and the same on monthly-mean GHI. Neural models: seeds 0,1,2.

Usage: python run_experiments.py --cams RjimMaatoug.csv --met dataset1000.xlsx --out results
"""
from __future__ import annotations

import argparse
import os
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ghibgmc.data import (CovTransform, Deseasonaliser, expit, load_dataset, logit,
                          prepare_covariates)
from ghibgmc.features import cpt_table, learn_bn1
from ghibgmc.generator import ScenarioGenerator, bootstrap_scenarios
from ghibgmc.forecaster import CondPatchTST, LinearGauss, UniPatchTST
from ghibgmc.diffusion import DiffusionForecaster
from ghibgmc import baselines as B
from ghibgmc.metrics import crps_ensemble, monthly_scores, scores

warnings.filterwarnings("ignore")
torch.set_num_threads(int(os.environ.get("NTHREADS", "2")))

ORIGINS = [f"{y}-12-31" for y in range(2015, 2023)]
SEEDS = [0, 1, 2]
S_EVAL = 200
LEADS = 4
GEN_ITERS = 1200


def drift_paths(resid: np.ndarray, resid_dates: pd.DatetimeIndex, n_years: int, S: int, seed: int):
    """Local-level model on annual means of in-sample residuals; simulated forward."""
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    s = pd.Series(resid, index=resid_dates)
    cnt = s.groupby(s.index.year).count()
    ann = s.groupby(s.index.year).mean()[cnt >= 300]
    r = UnobservedComponents(ann.values, level="llevel").fit(disp=False)
    np.random.seed(seed)
    sim = r.simulate(n_years, repetitions=S, anchor="end")      # [n_years, 1, S] or [n_years, S]
    sim = np.asarray(sim).reshape(n_years, -1)[:, :S]
    return sim.T, r                                                # [S, n_years]


def to_kt(zanom, zds, dates):
    mu = zds.mean(dates, "z")
    sd = zds.std(dates, "z")
    return expit(mu + sd * zanom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", default="../data/RjimMaatoug.csv")
    ap.add_argument("--met", default="../data/dataset1000.xlsx")
    ap.add_argument("--out", default="results")
    ap.add_argument("--origins", nargs="*", default=ORIGINS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    ap.add_argument("--foundation", action="store_true", help="also run Chronos/TimesFM/MOMENT")
    a = ap.parse_args()
    out = Path(a.out); (out / "folds").mkdir(parents=True, exist_ok=True)
    gen_iters = 50 if a.quick else GEN_ITERS

    df = load_dataset(a.cams, a.met)
    cov_all = prepare_covariates(df)
    END = df.index[-1]

    for origin in a.origins:
        fold_file = out / "folds" / f"{origin[:4]}.pkl"
        if fold_file.exists():
            print("skip", origin); continue
        t0 = time.time()
        o = pd.Timestamp(origin)
        fut = pd.date_range(o + pd.Timedelta(days=1), min(o + pd.Timedelta(days=LEADS * 365), END))
        n_lead = int(np.ceil(len(fut) / 365))
        blocks = [np.arange(l * 365, min((l + 1) * 365, len(fut))) for l in range(n_lead)]
        tr = df.loc[:o]
        ghi_obs = df.loc[fut, "GHI"].values
        toa = df.loc[fut, "TOA"].values

        # --- target transform (training only)
        z_tr = pd.DataFrame({"z": logit(tr["Kt_filled"].values)}, index=tr.index)
        zds = Deseasonaliser().fit(z_tr)
        zanom_tr = zds.transform(z_tr)["z"].values
        zanom_tr_obs = np.where(tr["observed"].values, zanom_tr, np.nan)

        # --- covariates (training only) + BN-1
        ct = CovTransform(seed=0).fit(cov_all.loc[:o])
        an_tr = ct.transform(cov_all.loc[:o])
        bn = learn_bn1(an_tr.assign(Kt=zanom_tr))
        sel = bn["markov_blanket"] or list(an_tr.columns)
        print(origin, "BN-1 selected:", sel)
        an_fut_obs = ct.transform(cov_all.loc[fut], seed=1)           # oracle diagnostic only

        # --- scenario generators (fitted once per origin)
        gens = {}
        for name, bayes in [("bayes", True), ("det", False)]:
            g = ScenarioGenerator(bayesian=bayes, iters=gen_iters, n_critic=3, seed=0).fit(an_tr)
            gens[name] = g.sample(fut, S_EVAL, seed=0)
        gens["boot"] = bootstrap_scenarios(an_tr, fut, S_EVAL, seed=0)
        cols = list(an_tr.columns)
        isel = [cols.index(c) for c in sel]
        hist = an_tr.values[-(CondPatchTST.L - 1):]

        ens = {}      # model -> seed -> Kt ensemble [M, T]
        def put(name, seed, kt):
            ens.setdefault(name, {})[seed] = np.clip(kt, 0, 1)
            print(f"    [{time.time()-t0:6.0f}s] {name} seed {seed}", flush=True)

        # --- statistical baselines (deterministic given data)
        kt_tr = tr["Kt"]
        put("Climatology", 0, B.climatology(kt_tr, fut, S_EVAL))
        put("Seasonal naive", 0, B.seasonal_naive(kt_tr, fut))
        put("ARIMA(1,0,1)", 0, to_kt(B.arima(zanom_tr, len(fut), S_EVAL), zds, fut))
        kt_b, _ = B.bsts(pd.Series(z_tr["z"].values, index=tr.index), len(fut), S_EVAL)
        put("BSTS", 0, kt_b)
        try:
            put("Prophet", 0, B.prophet(pd.Series(z_tr["z"].values, index=tr.index), fut, S_EVAL))
        except Exception as e:  # noqa: BLE001
            print("Prophet failed", e)
        if a.foundation:
            for fm, lab in [("chronos", "Chronos-Bolt (zero-shot)"), ("timesfm", "TimesFM 2.5 (zero-shot)"),
                            ("moment", "MOMENT (zero-shot)")]:
                r = B.foundation(fm, tr["Kt_filled"].values, len(fut), S_EVAL)
                if r is not None:
                    put(lab, 0, r)

        drift_info = {}
        for seed in a.seeds:
            ep = 3 if a.quick else 60
            # proposed conditional forecaster + ablation A4
            cond = CondPatchTST(seed=seed, epochs=ep).fit(an_tr.values[:, isel], tr.index.dayofyear.values,
                                                          zanom_tr_obs)
            lin = LinearGauss(seed=seed, epochs=ep).fit(an_tr.values[:, isel], tr.index.dayofyear.values,
                                                        zanom_tr_obs)
            rd = tr.index[cond.train_resid_idx]
            dp, dres = drift_paths(cond.train_resid, rd, n_lead, S_EVAL, seed)
            drift_info[seed] = dict(level_var=float(dres.params[-1]) if len(dres.params) else np.nan,
                                    last_level=float(dres.smoothed_state[0, -1]))
            dday = np.concatenate([np.repeat(dp[:, [l]], len(b), 1) for l, b in enumerate(blocks)], 1)
            rng = np.random.default_rng(seed)
            doy = fut.dayofyear.values

            def run(model, paths, drift=True, aleatoric=True):
                mu, sg = model.predict_paths(hist[:, isel], paths[..., isel], doy)
                zz = mu + (dday[: len(mu)] if drift else 0)
                if aleatoric:
                    zz = zz + sg * rng.standard_normal(zz.shape)
                return to_kt(zz, zds, fut)

            put("Proposed (Bayesian-GAN-MC)", seed, run(cond, gens["bayes"]))
            put("A1: bootstrap scenarios (no GAN)", seed, run(cond, gens["boot"]))
            put("A2: deterministic GAN (no Bayesian prior)", seed, run(cond, gens["det"]))
            put("A3: no Monte Carlo (single mean path)", seed,
                run(cond, np.zeros((1,) + gens["bayes"].shape[1:]), drift=False, aleatoric=False))
            put("A4: linear-Gaussian (no PatchTST)", seed, run(lin, gens["bayes"]))
            put("A5: no drift component", seed, run(cond, gens["bayes"], drift=False))
            orc = np.repeat(an_fut_obs.values[None], S_EVAL, 0)
            put("Oracle covariates (diagnostic)", seed, run(cond, orc, drift=False))

            # univariate deep baselines
            ctx = zanom_tr[-UniPatchTST.L:]
            first = np.array([fut[b[0]].dayofyear for b in blocks])
            for kind, lab in [("point", "PatchTST (point)"), ("mcdropout", "MC-dropout PatchTST"),
                              ("ensemble", "Deep ensemble PatchTST")]:
                if kind == "ensemble" and seed > 0 and a.quick:
                    continue
                m = UniPatchTST(kind, seed=seed, epochs=2 if a.quick else 40,
                                members=2 if a.quick else 5).fit(zanom_tr, tr.index.dayofyear.values)
                zs = m.sample(ctx, first, n_lead, S_EVAL, seed=seed)[:, :len(fut)]
                put(lab, seed, to_kt(zs, zds, fut))
            if seed == a.seeds[0]:      # diffusion baseline: one training run per origin (cost)
                dm = DiffusionForecaster(seed=seed, steps=30 if a.quick else 2000).fit(zanom_tr, tr.index)
                put("Conditional diffusion (CSDI-style)", seed,
                    to_kt(dm.sample(ctx, fut, n_lead, S_EVAL, seed=seed)[:, :len(fut)], zds, fut))
            print(f"  seed {seed} done ({time.time()-t0:.0f}s)")

        # --- scoring
        rows, daily = [], {}
        for name, by_seed in ens.items():
            for seed, kt in by_seed.items():
                ghi = kt * toa[None]
                for l, b in enumerate(blocks):
                    sc = scores(ghi_obs[b], ghi[:, b])
                    ms = monthly_scores(ghi_obs[b], ghi[:, b], fut[b])
                    rows.append(dict(model=name, seed=seed, origin=origin[:4], lead=l + 1,
                                     **sc, **{f"m_{k}": v for k, v in ms.items()}))
                if seed == 0:
                    ok = np.isfinite(ghi_obs)
                    c = np.full(len(fut), np.nan)
                    c[ok] = crps_ensemble(ghi_obs[ok], ghi[:, ok])
                    daily[name] = c
        res = dict(rows=rows, daily_crps=daily, dates=fut, selected=sel, drift=drift_info,
                   bn_edges=list(bn["skeleton"].edges()), cpt=cpt_table(bn),
                   gen_bayes=gens["bayes"][:50], gen_det=gens["det"][:50], gen_boot=gens["boot"][:50],
                   obs_cov_anom=an_fut_obs.values, ghi_obs=ghi_obs, toa=toa,
                   ens_store={k: v[0] for k, v in ens.items()
                              if k in ("Proposed (Bayesian-GAN-MC)", "Climatology", "Oracle covariates (diagnostic)")})
        with open(fold_file, "wb") as f:
            pickle.dump(res, f)
        print(f"{origin} finished in {time.time()-t0:.0f}s")
    print("all folds done")


if __name__ == "__main__":
    main()
