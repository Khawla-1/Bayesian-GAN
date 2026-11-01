#!/usr/bin/env python3
"""Final-stage analyses (paper Secs. V-E to V-G).

1. Monte Carlo ensemble-size sensitivity on the 2019 fold (test 2020-2023):
   S in {50,100,200,500,1000}, 5 independent repetitions each.
2. Sobol variance-based sensitivity (Saltelli/Jansen estimators, SALib) of the 1-year
   (2020) mean predicted GHI to uniform anomaly offsets of each BN-1-selected covariate.
   With second-order indices the cost is N(2d+2) model evaluations.
3. 2024-2045 scenario projection with the model refitted on the whole record
   (S=1000). This is a scenario projection, NOT a verifiable forecast.
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

from ghibgmc.data import CovTransform, Deseasonaliser, expit, load_dataset, logit, prepare_covariates
from ghibgmc.features import learn_bn1
from ghibgmc.forecaster import CondPatchTST
from ghibgmc.generator import ScenarioGenerator
from ghibgmc.metrics import scores
from run_experiments import drift_paths, to_kt, GEN_ITERS

warnings.filterwarnings("ignore")
torch.set_num_threads(int(os.environ.get("NTHREADS", "2")))


def fit_all(df, cov_all, origin, seed=0, iters=GEN_ITERS, epochs=60):
    o = pd.Timestamp(origin)
    tr = df.loc[:o]
    z_tr = pd.DataFrame({"z": logit(tr["Kt_filled"].values)}, index=tr.index)
    zds = Deseasonaliser().fit(z_tr)
    zanom = zds.transform(z_tr)["z"].values
    zobs = np.where(tr["observed"].values, zanom, np.nan)
    ct = CovTransform(seed=0).fit(cov_all.loc[:o])
    an = ct.transform(cov_all.loc[:o])
    bn = learn_bn1(an.assign(Kt=zanom))
    sel = bn["markov_blanket"] or list(an.columns)
    isel = [list(an.columns).index(c) for c in sel]
    gen = ScenarioGenerator(bayesian=True, iters=iters, n_critic=3, seed=seed).fit(an)
    cond = CondPatchTST(seed=seed, epochs=epochs).fit(an.values[:, isel], tr.index.dayofyear.values, zobs)
    return dict(tr=tr, zds=zds, ct=ct, an=an, sel=sel, isel=isel, gen=gen, cond=cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", default="../data/RjimMaatoug.csv")
    ap.add_argument("--met", default="../data/dataset1000.xlsx")
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stages", nargs="*", default=["mc", "sobol", "proj"])
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True, parents=True)
    iters, epochs = (50, 3) if a.quick else (GEN_ITERS, 60)
    df = load_dataset(a.cams, a.met)
    cov_all = prepare_covariates(df)
    END = df.index[-1]

    if "mc" in a.stages or "sobol" in a.stages:
        F = fit_all(df, cov_all, "2019-12-31", iters=iters, epochs=epochs)
        fut = pd.date_range("2020-01-01", END)
        hist = F["an"].values[-(CondPatchTST.L - 1):, F["isel"]]
        toa = df.loc[fut, "TOA"].values
        obs = df.loc[fut, "GHI"].values
        n_lead = int(np.ceil(len(fut) / 365))
        blocks = [np.arange(l * 365, min((l + 1) * 365, len(fut))) for l in range(n_lead)]

    if "mc" in a.stages:
        rows = []
        for S in [50, 100, 200, 500, 1000]:
            for rep in range(5):
                t = time.time()
                paths = F["gen"].sample(fut, S, seed=1000 + rep)
                rd = F["tr"].index[F["cond"].train_resid_idx]
                dp, _ = drift_paths(F["cond"].train_resid, rd, n_lead, S, 1000 + rep)
                dday = np.concatenate([np.repeat(dp[:, [l]], len(b), 1) for l, b in enumerate(blocks)], 1)
                mu, sg = F["cond"].predict_paths(hist, paths[..., F["isel"]], fut.dayofyear.values)
                rng = np.random.default_rng(rep)
                kt = to_kt(mu + dday + sg * rng.standard_normal(mu.shape), F["zds"], fut)
                el = time.time() - t
                sc = scores(obs[blocks[0]], (kt * toa)[:, blocks[0]])
                rows.append(dict(S=S, rep=rep, CRPS=sc["CRPS"], MPIW=sc["MPIW"], PICP=sc["PICP"], time_s=el))
            print("MC S", S, "done")
        pd.DataFrame(rows).to_csv(out / "mc_sensitivity.csv", index=False)

    if "sobol" in a.stages:
        from SALib.sample import sobol as ssob
        from SALib.analyze import sobol as asob
        sel, d = F["sel"], len(F["sel"])
        prob = dict(num_vars=d, names=sel, bounds=[[-1.0, 1.0]] * d)
        N = 256 if a.quick else 1024
        X = ssob.sample(prob, N, calc_second_order=True, seed=0)
        y1 = fut[blocks[0]]
        base = F["ct"].transform(cov_all.loc[y1], seed=1).values[:, F["isel"]]
        Y = np.empty(len(X))
        t = time.time()
        for i in range(0, len(X), 64):
            P = base[None] + X[i:i + 64, None, :]
            mu, _ = F["cond"].predict_paths(hist, P, y1.dayofyear.values)
            Y[i:i + 64] = (to_kt(mu, F["zds"], y1) * df.loc[y1, "TOA"].values[None]).mean(1)
        Si = asob.analyze(prob, Y, calc_second_order=True, seed=0)
        res = pd.DataFrame({"feature": sel, "S1": Si["S1"], "S1_conf": Si["S1_conf"],
                            "ST": Si["ST"], "ST_conf": Si["ST_conf"]})
        res.attrs = {}
        res.to_csv(out / "sobol.csv", index=False)
        json.dump(dict(N=N, d=d, n_eval=int(len(X)), formula="N(2d+2)", sum_S1=float(np.sum(Si["S1"])),
                       output_mean=float(Y.mean()), output_sd=float(Y.std()), seconds=time.time() - t),
                  open(out / "sobol_meta.json", "w"), indent=1)
        print("Sobol", len(X), "evaluations")

    if "proj" in a.stages:
        G = fit_all(df, cov_all, str(END.date()), iters=iters, epochs=epochs)
        fut = pd.date_range(END + pd.Timedelta(days=1), "2045-12-31")
        years = sorted(set(fut.year))
        S, chunk = (100, 50) if a.quick else (1000, 50)
        hist = G["an"].values[-(CondPatchTST.L - 1):, G["isel"]]
        toa = G["tr"].groupby(G["tr"].index.dayofyear)["TOA"].mean().reindex(range(1, 367)).ffill().reindex(
            fut.dayofyear).values
        rd = G["tr"].index[G["cond"].train_resid_idx]
        n_years = len(years)
        dp, dres = drift_paths(G["cond"].train_resid, rd, n_years, S, 7)
        yidx = np.array([years.index(y) for y in fut.year])
        ann, mon = [], []
        for c0 in range(0, S, chunk):
            paths = G["gen"].sample(fut, chunk, seed=c0)
            mu, sg = G["cond"].predict_paths(hist, paths[..., G["isel"]], fut.dayofyear.values)
            rng = np.random.default_rng(c0)
            kt = to_kt(mu + dp[c0:c0 + chunk][:, yidx] + sg * rng.standard_normal(mu.shape), G["zds"], fut)
            ghi = pd.DataFrame((kt * toa[None]).T, index=fut)
            ann.append(ghi.groupby(ghi.index.year).mean().values)
            mon.append(ghi.groupby(ghi.index.month).mean().values)
            print("proj chunk", c0)
        ann = np.concatenate(ann, 1); mon = np.concatenate(mon, 1)
        q = [0.05, 0.25, 0.5, 0.75, 0.95]
        pd.DataFrame(np.quantile(ann, q, axis=1).T, index=years, columns=[f"q{int(x*100)}" for x in q]).assign(
            mean=ann.mean(1)).to_csv(out / "projection_annual.csv")
        pd.DataFrame(np.quantile(mon, q, axis=1).T, index=range(1, 13), columns=[f"q{int(x*100)}" for x in q]).to_csv(
            out / "projection_monthly.csv")
        obs_ann = df["GHI"].groupby(df.index.year).agg(["mean", "count"])
        obs_ann.to_csv(out / "observed_annual.csv")
        json.dump(dict(S=S, selected=G["sel"], drift_last_level=float(dres.smoothed_state[0, -1]),
                       drift_params=dict(zip(dres.model.param_names, map(float, dres.params)))),
                  open(out / "projection_meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
