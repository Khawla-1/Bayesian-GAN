#!/usr/bin/env python3
"""Descriptive statistics, aggregation of rolling-origin results, LaTeX tables and figures.

Reads results/folds/*.pkl (run_experiments.py) and results/*.csv|json (run_final.py) and
writes results/tables/*.tex, results/figs/*.pdf and results/summary.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ghibgmc.data import CovTransform, Deseasonaliser, load_dataset, logit, prepare_covariates
from ghibgmc.generator import var1_scenarios
from ghibgmc.metrics import diebold_mariano, pit

warnings.filterwarnings("ignore")
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4",
         green="#008300", ink="#222222", grey="#8a8a8a", light="#d9d9d9")
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5,
                     "axes.edgecolor": "#888888", "lines.linewidth": 1.2, "legend.frameon": False,
                     "savefig.bbox": "tight", "font.family": "serif"})
PROP = "Proposed (Bayesian-GAN-MC)"
TEXNAME = {"WDs": "$u$ (zonal wind)", "WDc": "$v$ (meridional wind)", "Temp": "Temp", "Hum": "Hum",
           "Prec": "Prec", "Ps": "Ps", "WS": "WS"}
FIGNAME = {"WDs": "u", "WDc": "v"}
CLIM = "Climatology"
ORDER = ["Climatology", "Seasonal naive", "ARIMA(1,0,1)", "Prophet", "BSTS", "PatchTST (point)",
         "MC-dropout PatchTST", "Deep ensemble PatchTST", "Conditional diffusion (CSDI-style)",
         "Chronos-Bolt (zero-shot)", "TimesFM 2.5 (zero-shot)", "MOMENT (linear probe)", PROP]
ABL = [PROP, "A1: bootstrap scenarios (no GAN)", "A2: deterministic GAN (no Bayesian prior)",
       "A3: no Monte Carlo (single mean path)", "A4: linear-Gaussian (no PatchTST)",
       "A5: no drift component", "Oracle covariates (diagnostic)"]


def pm(m, s, f="{:.1f}"):
    return (f + r"\,$\pm$\," + f).format(m, s) if np.isfinite(s) else f.format(m)


def tex_table(df, path, colfmt=None, header=None):
    colfmt = colfmt or "l" + "c" * (df.shape[1] - 1)
    lines = [r"\begin{tabular}{" + colfmt + "}", r"\toprule",
             " & ".join(header or df.columns) + r" \\", r"\midrule"]
    import re
    fix = lambda v: re.sub(r"(?<![\w${^-])-(?=\d)", "$-$", str(v))     # typographic minus
    for _, r in df.iterrows():
        lines.append(" & ".join(fix(v) for v in r.values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(path).write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------------------------------
def descriptive(df, T, F, summ):
    tr = df.loc[:"2019-12-31"]
    # ---- distribution tests (training period only)
    ghi = tr["GHI"].dropna()
    z = pd.DataFrame({"z": logit(tr["Kt_filled"].values)}, index=tr.index)
    za = Deseasonaliser().fit(z).transform(z)["z"].where(tr["observed"]).dropna()
    rows = []
    for name, x in [("GHI (raw, Wh m$^{-2}$)", ghi.values), ("Kt (raw)", tr["Kt"].dropna().values),
                    ("logit-Kt anomaly $z_t$", za.values)]:
        xs = (x - x.mean()) / x.std(ddof=1)
        ks = stats.kstest(xs, "norm")
        k2 = stats.normaltest(x)
        rows.append([name, f"{len(x)}", f"{stats.skew(x):.2f}", f"{stats.kurtosis(x):.2f}",
                     f"{k2.statistic:.1f} ({k2.pvalue:.1e})", f"{ks.statistic:.3f}"])
    tex_table(pd.DataFrame(rows), T / "tab_distribution.tex", "lccccc",
              ["Series (2004--2019)", "$n$", "Skew.", "Ex. kurt.", r"D'Agostino $K^2$ ($p$)",
               r"KS $D$ (standardised)"])
    summ["distribution"] = rows

    # ---- structural tests on monthly Kt anomalies, full record
    from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
    m = df["Kt"].resample("MS").mean()
    an = (m - m.groupby(m.index.month).transform("mean")).dropna()
    rows = []
    for reg, lab in [("c", "A (intercept)"), ("t", "B (trend)"), ("ct", "C (both)")]:
        r = zivot_andrews(an.values, regression=reg, autolag="AIC")
        rows.append([lab, an.index[r[4]].strftime("%Y-%m"), f"{r[0]:.2f}", f"{r[2]['5%']:.2f}",
                     f"{r[1]:.3f}"])
    tex_table(pd.DataFrame(rows), T / "tab_za.tex", "lcccc",
              ["ZA model", "Break date", "Statistic", "5\\% c.v.", "$p$"])
    k = kpss(an.values, regression="ct", nlags="auto")
    ad = adfuller(an.values, autolag="AIC")
    yrs = df["Kt"].groupby(df.index.year)
    ann = yrs.mean()[yrs.count() >= 360]                       # complete years only (2005-2022)
    ts = stats.theilslopes(ann.values, ann.index.values)
    kt_ = stats.kendalltau(ann.index.values, ann.values)
    ghi_ann = df["GHI"].groupby(df.index.year).mean()[yrs.count() >= 360]
    roll = {f"{a}--{a+2}": ghi_ann.loc[a:a + 2].mean() for a in (2005, 2011, 2017, 2020)}
    base = roll["2005--2007"]
    tex_table(pd.DataFrame([[k_, f"{v:,.1f}", f"{100*(v/base-1):+.1f}\\%"] for k_, v in roll.items()]),
              T / "tab_rolling.tex", "lcc", ["Window (complete years)", "Mean daily GHI (Wh m$^{-2}$)", "vs 2005--2007"])
    summ["breaks"] = dict(za=rows, kpss=[float(k[0]), float(k[1])], adf=[float(ad[0]), float(ad[1])],
                          theil_slope_per_yr=float(ts[0]), theil_lo=float(ts[2]), theil_hi=float(ts[3]),
                          kendall_tau=float(kt_.statistic), kendall_p=float(kt_.pvalue),
                          ghi_slope=float(stats.theilslopes(ghi_ann.values, ghi_ann.index.values)[0]),
                          rolling=roll, n_months=len(an), ann_years=[int(ann.index[0]), int(ann.index[-1])])

    # ---- figure: annual mean Kt
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    ax.plot(ann.index, ann.values, "o-", color=C["blue"], ms=3, label="Annual mean $K_t$ (CAMS)")
    ax.plot(ann.index, ts[1] + ts[0] * ann.index.values, "--", color=C["ink"], lw=1,
            label=f"Theil--Sen slope {ts[0]*10:+.4f}/decade")
    ax.set_xticks(range(2005, 2023, 3)); ax.set_xlabel("Year"); ax.set_ylabel("$K_t$")
    ax.legend(loc="upper left", fontsize=7)
    fig.savefig(F / "fig_annual_kt.pdf"); plt.close(fig)


def location_map(F):
    import geopandas as gpd
    w = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    w.plot(ax=ax, color="#f2f2f2", edgecolor="#9a9a9a", lw=0.5)
    w[w.name == "Tunisia"].plot(ax=ax, color="#dbe8f7", edgecolor="#4a4a4a", lw=0.8)
    ax.plot(8.0, 33.0, "o", color=C["orange"], ms=6, mec="white", mew=0.8, zorder=5)
    ax.annotate("Rjim Maatoug\n(CAMS 33.00°N, 8.00°E)", (8.0, 33.0), (8.6, 31.3), fontsize=7,
                arrowprops=dict(arrowstyle="-", color="#555555", lw=0.6))
    ax.plot(9.0649, 33.6739, "s", color=C["blue"], ms=4, mec="white", mew=0.6, zorder=5)
    ax.annotate("MERRA-2 extraction\n(33.67°N, 9.06°E)", (9.06, 33.67), (9.9, 34.6), fontsize=7,
                arrowprops=dict(arrowstyle="-", color="#555555", lw=0.6))
    ax.set_xlim(6.5, 12.5); ax.set_ylim(29.8, 37.8); ax.set_aspect("equal")
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    ax.text(10.3, 37.4, "Mediterranean Sea", fontsize=6, color="#555555", style="italic")
    ax.text(6.7, 30.6, "ALGERIA", fontsize=6, color="#777777"); ax.text(11.0, 30.6, "LIBYA", fontsize=6, color="#777777")
    ax.text(9.2, 35.4, "TUNISIA", fontsize=7, color="#333333")
    fig.savefig(F / "fig_location.pdf"); plt.close(fig)


# ------------------------------------------------------------------------------------------
def load_folds(R):
    folds = {}
    for f in sorted(glob.glob(str(R / "folds" / "*.pkl"))):
        folds[Path(f).stem] = pickle.load(open(f, "rb"))
    for f in sorted(glob.glob(str(R / "foundation" / "*.pkl"))):     # optional foundation models
        k = Path(f).stem
        if k in folds:
            extra = pickle.load(open(f, "rb"))
            folds[k]["rows"] = folds[k]["rows"] + extra["rows"]
            folds[k]["daily_crps"].update(extra["daily_crps"])
    rows = pd.DataFrame([r for v in folds.values() for r in v["rows"]])
    return folds, rows


def aggregate(rows, T, F, summ):
    clim = rows[rows.model == CLIM][["origin", "lead", "CRPS", "m_CRPS"]].rename(
        columns={"CRPS": "c_clim", "m_CRPS": "mc_clim"})
    r = rows.merge(clim, on=["origin", "lead"])
    r["CRPSS"] = 100 * (1 - r.CRPS / r.c_clim)
    r["m_CRPSS"] = 100 * (1 - r.m_CRPS / r.mc_clim)
    # average seeds within fold first, then mean +- sd across folds
    fold = r.groupby(["model", "lead", "origin"]).mean(numeric_only=True).reset_index()
    seedsd = r.groupby(["model", "lead", "origin"]).CRPS.std().groupby(["model", "lead"]).mean()
    agg = fold.drop(columns=["origin"]).groupby(["model", "lead"]).agg(["mean", "std", "count"])
    summ["agg_lead1"] = {m: {k: float(agg.loc[(m, 1), (k, "mean")]) for k in
                             ["CRPS", "MAE", "RMSE", "PICP", "PINAW", "CRPSS", "m_CRPS", "m_CRPSS"]}
                         for m in agg.index.get_level_values(0).unique() if (m, 1) in agg.index}
    summ["n_folds_lead"] = {int(l): int(agg.xs(l, level="lead")[("CRPS", "count")].max())
                            for l in sorted(set(r.lead))}

    def row(m, l=1):
        a = agg.loc[(m, l)]
        g = lambda k, f="{:.1f}": pm(a[(k, "mean")], a[(k, "std")], f)
        return [m, g("CRPS", "{:.0f}"), g("MAE", "{:.0f}"), g("RMSE", "{:.0f}"), g("PICP"),
                g("PINAW", "{:.3f}"), g("CRPSS"), g("m_CRPS", "{:.0f}"), g("m_CRPSS")]
    hdr = ["Model", "CRPS", "MAE", "RMSE", "PICP (\\%)", "PINAW", "CRPSS (\\%)", "CRPS$_{\\mathrm{mon}}$",
           "CRPSS$_{\\mathrm{mon}}$ (\\%)"]
    present = [m for m in ORDER if (m, 1) in agg.index]
    tex_table(pd.DataFrame([row(m) for m in present]), T / "tab_main.tex", "l" + "c" * 8, hdr)
    tex_table(pd.DataFrame([row(m) for m in ABL if (m, 1) in agg.index]), T / "tab_ablation.tex",
              "l" + "c" * 8, hdr)
    summ["seed_sd_crps_lead1"] = {m: float(seedsd.get((m, 1), np.nan)) for m in present + ABL}

    # CRPSS by lead table and figure
    key = [CLIM, "BSTS", "Prophet", "Deep ensemble PatchTST", "Conditional diffusion (CSDI-style)", PROP,
           "Oracle covariates (diagnostic)"]
    key = [m for m in key if (m, 1) in agg.index]
    leads = sorted(set(r.lead))
    trows = []
    for m in key:
        trows.append([m] + [pm(agg.loc[(m, l), ("CRPSS", "mean")], agg.loc[(m, l), ("CRPSS", "std")])
                            + " / " + pm(agg.loc[(m, l), ("m_CRPSS", "mean")], agg.loc[(m, l), ("m_CRPSS", "std")])
                            for l in leads])
    tex_table(pd.DataFrame(trows), T / "tab_lead.tex", "l" + "c" * len(leads),
              ["Model"] + [f"Lead {l} ($n$={summ['n_folds_lead'][l]})" for l in leads])
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=False)
    cols = [C["grey"], C["yellow"], C["magenta"], C["green"], C["aqua"], C["blue"], C["orange"]]
    for ax, k, ttl in [(axs[0], "CRPSS", "Daily GHI"), (axs[1], "m_CRPSS", "Monthly-mean GHI")]:
        for i, m in enumerate(key):
            mu = [agg.loc[(m, l), (k, "mean")] for l in leads]
            sd = [agg.loc[(m, l), (k, "std")] for l in leads]
            ls = ":" if "Oracle" in m else "-"
            ax.errorbar(np.array(leads) + (i - 3) * 0.04, mu, yerr=sd, fmt="o" + ls, ms=3, capsize=2,
                        color=cols[i % len(cols)], label=m.replace(" (Bayesian-GAN-MC)", ""), lw=1)
        ax.axhline(0, color=C["ink"], lw=0.6)
        ax.set_xticks(leads); ax.set_xlabel("Lead year"); ax.set_ylabel("CRPSS vs climatology (%)"); ax.set_title(ttl)
    axs[1].set_ylim(-60, 40)
    axs[1].text(0.02, 0.03, "axis clipped at -60%", transform=axs[1].transAxes, fontsize=6, color="#666666")
    axs[1].legend(fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.savefig(F / "fig_crpss_lead.pdf"); plt.close(fig)
    return agg


def dm_tests(folds, T, summ):
    rows = []
    comp = [CLIM, "BSTS", "Deep ensemble PatchTST", "Conditional diffusion (CSDI-style)",
            "A1: bootstrap scenarios (no GAN)"]
    for other in comp:
        res = []
        for l in range(1, 5):
            a, b = [], []
            for f in folds.values():
                idx = np.arange((l - 1) * 365, min(l * 365, len(f["dates"])))
                if len(idx) == 0 or other not in f["daily_crps"]:
                    continue
                a.append(f["daily_crps"][PROP][idx]); b.append(f["daily_crps"][other][idx])
            if not a:
                res.append("--"); continue
            st, p = diebold_mariano(np.concatenate(a), np.concatenate(b))
            res.append(f"{st:+.2f} ({p:.3f})")
        rows.append([other] + res)
    tex_table(pd.DataFrame(rows), T / "tab_dm.tex", "lcccc",
              ["Proposed vs.", "Lead 1", "Lead 2", "Lead 3", "Lead 4"])
    summ["dm"] = rows


def generator_diag(df, folds, T, F, summ):
    f = folds.get("2019")
    if f is None:
        return
    cov = prepare_covariates(df)
    ct = CovTransform(seed=0).fit(cov.loc[:"2019-12-31"])
    an_tr = ct.transform(cov.loc[:"2019-12-31"])
    fut = f["dates"]
    obs = f["obs_cov_anom"]
    gens = dict(BayesGAN=f["gen_bayes"], DetGAN=f["gen_det"], Bootstrap=f["gen_boot"],
                VAR1=var1_scenarios(an_tr, fut, 50))
    names = list(an_tr.columns)
    acf1 = lambda x: np.corrcoef(x[:-1], x[1:])[0, 1]

    def summary(X):
        X = X.reshape(-1, X.shape[-1]) if X.ndim == 3 else X
        return X

    # window-level discriminative score (real test windows vs synthetic), logistic regression
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    def feats(arr):  # arr [T, d] -> windows of 64 days summary features
        out = []
        for s in range(0, arr.shape[0] - 64, 16):
            w = arr[s:s + 64]
            cc = np.corrcoef(w.T)[np.triu_indices(w.shape[1], 1)]
            out.append(np.concatenate([w.mean(0), w.std(0), [acf1(w[:, j]) for j in range(w.shape[1])], cc]))
        return np.nan_to_num(np.array(out))
    real = feats(obs)
    rows = []
    obs_cc = np.corrcoef(obs.T)
    for g, X in gens.items():
        sd_ratio = X.reshape(-1, X.shape[-1]).std(0) / obs.std(0)
        a_g = np.array([[acf1(X[i, :, j]) for j in range(X.shape[-1])] for i in range(min(20, len(X)))]).mean(0)
        a_o = np.array([acf1(obs[:, j]) for j in range(obs.shape[1])])
        ks = np.array([stats.ks_2samp(X[:, :, j].ravel(), obs[:, j]).statistic for j in range(obs.shape[1])])
        cc_err = np.mean(np.abs(np.mean([np.corrcoef(X[i].T) for i in range(min(20, len(X)))], 0) - obs_cc)
                         [np.triu_indices(obs.shape[1], 1)])
        syn = np.concatenate([feats(X[i]) for i in range(5)])
        n = min(len(real), len(syn))
        Xc = np.concatenate([real[:n], syn[:n]]); yc = np.r_[np.zeros(n), np.ones(n)]
        acc = cross_val_score(LogisticRegression(max_iter=2000, C=0.5), Xc, yc, cv=5).mean()
        rows.append([g, f"{np.mean(np.abs(sd_ratio-1)):.3f}", f"{np.mean(np.abs(a_g-a_o)):.3f}",
                     f"{ks.mean():.3f}", f"{cc_err:.3f}", f"{abs(acc-0.5):.3f}"])
    labels = {"BayesGAN": "Bayesian WGAN-GP (proposed)", "DetGAN": "Deterministic WGAN-GP",
              "Bootstrap": "Year-block bootstrap", "VAR1": "Gaussian VAR(1)"}
    rows = [[labels[r[0]]] + r[1:] for r in rows]
    tex_table(pd.DataFrame(rows), T / "tab_generator.tex", "lccccc",
              ["Generator", r"$\overline{|\sigma_g/\sigma_o-1|}$", r"$\overline{|\Delta\rho_1|}$",
               r"$\overline{\mathrm{KS}}$", r"$\overline{|\Delta r_{ij}|}$", r"$|\mathrm{acc}-0.5|$"])
    summ["generator"] = rows

    # figure: annual cycle in physical units (obs 2020-2023 vs generated band)
    phys_g = ct.inverse(f["gen_bayes"], fut)
    phys_o = cov.loc[fut].values
    fig, axs = plt.subplots(1, 4, figsize=(7.1, 1.8))
    for ax, (j, lab) in zip(axs, [(0, "Temperature (°C)"), (1, "Rel. humidity (%)"), (3, "Pressure (kPa)"),
                                  (4, "Wind speed (m/s)")]):
        doy = fut.dayofyear.values
        ud = np.unique(doy)
        qs = np.array([np.quantile(phys_g[:, doy == d, j].ravel(), [0.05, 0.5, 0.95]) for d in ud])
        q05, q50, q95 = (pd.Series(qs[:, i], index=ud) for i in range(3))
        sm = lambda s: s.rolling(7, center=True, min_periods=1).mean()
        ax.fill_between(q50.index, sm(q05), sm(q95), color=C["blue"], alpha=0.25, lw=0, label="Generated 5--95%")
        ax.plot(q50.index, sm(q50), color=C["blue"], lw=1, label="Generated median")
        ax.plot(doy, phys_o[:, j], ".", color=C["ink"], ms=0.8, alpha=0.5, label="Observed 2020--2023")
        ax.set_title(lab, fontsize=7.5); ax.set_xlabel("Day of year")
    axs[0].legend(fontsize=5.5, loc="lower center", markerscale=6)
    fig.tight_layout(); fig.savefig(F / "fig_generator.pdf"); plt.close(fig)


def backtest_fig(df, folds, F, summ):
    f = folds.get("2019")
    if f is None:
        return
    fut = f["dates"]; obs = f["ghi_obs"]; toa = f["toa"]
    P = f["ens_store"][PROP] * toa[None]; Cl = f["ens_store"][CLIM] * toa[None]
    fig, axs = plt.subplots(2, 1, figsize=(7.0, 3.6), sharex=True)
    ax = axs[0]
    lo, hi = np.quantile(P, [0.05, 0.95], 0)
    ax.fill_between(fut, lo, hi, color=C["blue"], alpha=0.25, lw=0, label="Proposed 90% interval")
    ax.plot(fut, np.median(P, 0), color=C["blue"], lw=0.8, label="Proposed median")
    ax.plot(fut, obs, color=C["ink"], lw=0.4, label="Observed (CAMS)")
    ax.set_ylabel("Daily GHI (Wh m$^{-2}$)"); ax.legend(fontsize=6, ncol=3, loc="lower left")
    ax = axs[1]
    mo = pd.Series(obs, index=fut).resample("MS").mean()
    mp = pd.DataFrame(P.T, index=fut).resample("MS").mean()
    mc = pd.DataFrame(Cl.T, index=fut).resample("MS").mean()
    ax.fill_between(mp.index, mc.quantile(0.05, axis=1), mc.quantile(0.95, axis=1), color=C["orange"], alpha=0.2,
                    lw=0, label="Climatology 90%")
    ax.fill_between(mp.index, mp.quantile(0.05, axis=1), mp.quantile(0.95, axis=1), color=C["blue"], alpha=0.3,
                    lw=0, label="Proposed 90%")
    ax.plot(mo.index, mo.values, "o-", color=C["ink"], ms=2, lw=0.8, label="Observed monthly mean")
    ax.set_ylabel("Monthly-mean GHI"); ax.legend(fontsize=6, ncol=3, loc="lower left")
    fig.tight_layout(); fig.savefig(F / "fig_backtest.pdf"); plt.close(fig)

    # PIT histograms (lead 1, pooled over folds)
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.8), sharey=True)
    for ax, (m, col) in zip(axs, [(PROP, C["blue"]), (CLIM, C["orange"]), ("Oracle covariates (diagnostic)", C["aqua"])]):
        pv = []
        for fo in folds.values():
            if m in fo["ens_store"]:
                b = np.arange(0, min(365, len(fo["dates"])))
                pv.append(pit(fo["ghi_obs"][b], fo["ens_store"][m][:, b] * fo["toa"][None, b]))
        pv = np.concatenate(pv)
        ax.hist(pv, bins=10, range=(0, 1), color=col, edgecolor="white", lw=1, density=True)
        ax.axhline(1, color=C["ink"], lw=0.6, ls="--")
        ax.set_title(m.replace(" (Bayesian-GAN-MC)", "").replace(" (diagnostic)", ""), fontsize=7.5)
        ax.set_xlabel("PIT")
    axs[0].set_ylabel("Density")
    fig.tight_layout(); fig.savefig(F / "fig_pit.pdf"); plt.close(fig)


def bn_fig(folds, T, F, summ):
    import networkx as nx
    sel = {k: v["selected"] for k, v in folds.items()}
    allv = ["Temp", "Hum", "Prec", "Ps", "WS", "WDs", "WDc"]
    freq = {v: sum(v in s for s in sel.values()) for v in allv}
    summ["bn1_selection"] = dict(per_origin=sel, freq=freq, n=len(sel))
    tex_table(pd.DataFrame([[TEXNAME[v], f"{freq[v]}/{len(sel)}"] for v in allv]), T / "tab_bn1.tex", "lc",
              ["Candidate input", "Selected (origins)"])
    f = folds.get("2019") or list(folds.values())[-1]
    G = nx.Graph(); G.add_edges_from(f["bn_edges"]); G.add_nodes_from(allv + ["Kt"])
    pos = nx.circular_layout(sorted(set(G.nodes) - {"Kt"}))
    pos["Kt"] = np.array([0.0, 0.0])
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    ek = [e for e in G.edges if "Kt" in e]; eo = [e for e in G.edges if "Kt" not in e]
    nx.draw_networkx_edges(G, pos, edgelist=eo, edge_color="#bdbdbd", width=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=ek, edge_color=C["blue"], width=1.8, ax=ax)
    colors = [C["orange"] if n == "Kt" else (C["blue"] if n in f["selected"] else "#eeeeee") for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=650, edgecolors="#555555", linewidths=0.6, ax=ax)
    lab = {n: ("$z_t$\n(Kt anom.)" if n == "Kt" else FIGNAME.get(n, n)) for n in G.nodes}
    nx.draw_networkx_labels(G, pos, lab, font_size=6.5,
                            font_color="black", ax=ax)
    ax.axis("off"); fig.savefig(F / "fig_bn1.pdf"); plt.close(fig)
    cpt = f["cpt"]
    summ["cpt_shape"] = list(cpt.shape)


def final_tables(R, T, F, summ, df):
    if (R / "mc_sensitivity.csv").exists():
        m = pd.read_csv(R / "mc_sensitivity.csv").groupby("S").agg(["mean", "std"])
        rows = [[int(s), pm(r[("CRPS", "mean")], r[("CRPS", "std")], "{:.1f}"),
                 pm(r[("MPIW", "mean")], r[("MPIW", "std")], "{:.0f}"),
                 pm(r[("PICP", "mean")], r[("PICP", "std")], "{:.1f}"), f"{r[('time_s', 'mean')]:.1f}"]
                for s, r in m.iterrows()]
        tex_table(pd.DataFrame(rows), T / "tab_mc.tex", "ccccc",
                  ["$S$", "CRPS (Wh m$^{-2}$)", "Mean 90\\% width", "PICP (\\%)", "Time (s)"])
        summ["mc"] = rows
    if (R / "sobol.csv").exists():
        s = pd.read_csv(R / "sobol.csv"); meta = json.load(open(R / "sobol_meta.json"))
        rows = [[TEXNAME.get(r.feature, r.feature), pm(r.S1, r.S1_conf, "{:.3f}"), pm(r.ST, r.ST_conf, "{:.3f}")]
                for r in s.itertuples()]
        rows.append([r"$\sum_i S_i$", f"{s.S1.sum():.3f}", f"{s.ST.sum():.3f}"])
        tex_table(pd.DataFrame(rows), T / "tab_sobol.tex", "lcc",
                  ["Input (offset)", "First order $S_i$", "Total order $S_{Ti}$"])
        summ["sobol"] = dict(rows=rows, meta=meta)
    if (R / "projection_annual.csv").exists():
        p = pd.read_csv(R / "projection_annual.csv", index_col=0)
        p = p[p.index >= 2024]          # Dec 2023 alone is not an annual mean
        o = pd.read_csv(R / "observed_annual.csv", index_col=0)
        o = o[o["count"] >= 360]
        fig, ax = plt.subplots(figsize=(3.5, 2.2))
        ax.fill_between(p.index, p.q5, p.q95, color=C["blue"], alpha=0.2, lw=0, label="Scenario 5--95%")
        ax.fill_between(p.index, p.q25, p.q75, color=C["blue"], alpha=0.35, lw=0, label="Scenario 25--75%")
        ax.plot(p.index, p.q50, color=C["blue"], lw=1.2, label="Scenario median")
        ax.plot(o.index, o["mean"], "o-", color=C["ink"], ms=2.5, lw=0.8, label="Observed (complete years)")
        ax.set_xticks(range(2005, 2046, 10)); ax.set_xlabel("Year"); ax.set_ylabel("Annual-mean daily GHI (Wh m$^{-2}$)")
        ax.legend(fontsize=6, loc="upper left")
        fig.savefig(F / "fig_projection.pdf"); plt.close(fig)
        pick = [y for y in (2025, 2030, 2035, 2040, 2045) if y in p.index]
        rows = [[y, f"{p.loc[y,'q50']:,.0f}", f"[{p.loc[y,'q5']:,.0f}, {p.loc[y,'q95']:,.0f}]",
                 f"{p.loc[y,'q95']-p.loc[y,'q5']:,.0f}"] for y in pick]
        tex_table(pd.DataFrame(rows), T / "tab_projection.tex", "cccc",
                  ["Year", "Median", "90\\% scenario range", "Width"])
        summ["projection"] = dict(rows=rows, obs_mean=float(o["mean"].mean()), meta=json.load(open(R / "projection_meta.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", default="../data/RjimMaatoug.csv")
    ap.add_argument("--met", default="../data/dataset1000.xlsx")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    R = Path(a.out); T = R / "tables"; F = R / "figs"
    T.mkdir(parents=True, exist_ok=True); F.mkdir(parents=True, exist_ok=True)
    df = load_dataset(a.cams, a.met)
    summ = dict(period=[str(df.index[0].date()), str(df.index[-1].date())], n_days=len(df),
                n_missing_ghi=int((~df.observed).sum()))
    descriptive(df, T, F, summ)
    location_map(F)
    folds, rows = load_folds(R)
    if len(rows):
        rows.to_csv(R / "all_scores.csv", index=False)
        aggregate(rows, T, F, summ)
        dm_tests(folds, T, summ)
        generator_diag(df, folds, T, F, summ)
        backtest_fig(df, folds, F, summ)
        bn_fig(folds, T, F, summ)
    final_tables(R, T, F, summ, df)
    json.dump(summ, open(R / "summary.json", "w"), indent=1, default=str)
    print(json.dumps({k: summ[k] for k in summ if k in ("period", "n_days", "breaks", "n_folds_lead")}, indent=1, default=str))


if __name__ == "__main__":
    main()
