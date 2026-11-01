"""Data loading and deterministic preprocessing.

Sources
-------
* CAMS Radiation Service v4.6 (daily sums, Wh m-2): GHI, BNI, DHI, TOA, reliability.
  File: RjimMaatoug.csv (site 33.00 N, 8.00 E), 2004-02-01 .. 2023-11-30.
* NASA POWER / MERRA-2 daily meteorology: T2M, RH2M, PRECTOTCORR, PS, WS10M, WD10M
  (and ALLSKY_KT, which is NOT used as a model input, see paper Sec. III-A).
  File: dataset1000.xlsx (columns Date, Kt, Temp, Hum, Prec, Ps, WS, WD, GHI, BNI, Re).

Target
------
The model target is the CAMS clearness index Kt = GHI / TOA, where TOA is the
deterministic top-of-atmosphere irradiation. GHI forecasts are recovered exactly as
GHI = Kt * TOA. No radiative variable (Kt_MERRA2, BNI, DHI) enters the inputs, so the
target cannot leak into the predictors.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CAMS_COLS = ["per", "TOA", "CS_GHI", "CS_BHI", "CS_DHI", "CS_BNI", "GHI", "BHI", "DHI", "BNI", "Rel"]
MET_COLS = ["Temp", "Hum", "Prec", "Ps", "WS", "WD"]
START, END = "2004-02-01", "2023-11-30"
KT_EPS = 0.02


def load_cams(path: str | Path) -> pd.DataFrame:
    c = pd.read_csv(path, sep=";", comment="#", header=None, names=CAMS_COLS)
    c = c[c["per"].astype(str).str.match(r"^\d{4}-")]
    c.index = pd.to_datetime(c["per"].str.slice(0, 10))
    c = c.drop(columns="per").astype(float)
    c.index.name = "Date"
    return c


def load_met(path: str | Path) -> pd.DataFrame:
    x = pd.read_excel(path)
    x = x.set_index(pd.to_datetime(x["Date"])).drop(columns="Date")
    x.index.name = "Date"
    return x[MET_COLS + ["Kt"]].rename(columns={"Kt": "Kt_MERRA2"})


def load_dataset(cams_csv: str | Path, met_xlsx: str | Path) -> pd.DataFrame:
    """Merged daily frame restricted to the period actually covered by CAMS."""
    cams = load_cams(cams_csv)
    met = load_met(met_xlsx)
    df = met.join(cams[["TOA", "GHI", "BNI", "DHI", "Rel"]], how="inner").loc[START:END]
    df["Kt"] = df["GHI"] / df["TOA"]
    df["observed"] = df["GHI"].notna()          # 8 missing CAMS days are never scored
    df["Kt_filled"] = df["Kt"].interpolate(limit_direction="both")
    df["doy"] = df.index.dayofyear
    # TOA for any calendar day (needed for future projections): smooth harmonic fit
    df["TOA_fit"] = toa_model(df)(df.index)
    return df


def toa_model(df: pd.DataFrame, harmonics: int = 4):
    """Least-squares harmonic fit of daily TOA vs day of year (TOA is astronomical)."""
    t = df.index.dayofyear.values
    X = fourier(t, harmonics)
    beta, *_ = np.linalg.lstsq(X, df["TOA"].values, rcond=None)

    def f(idx: pd.DatetimeIndex) -> np.ndarray:
        return fourier(idx.dayofyear.values, harmonics) @ beta
    return f


def fourier(doy: np.ndarray, k: int, period: float = 365.25) -> np.ndarray:
    cols = [np.ones_like(doy, dtype=float)]
    for h in range(1, k + 1):
        w = 2 * np.pi * h * doy / period
        cols += [np.sin(w), np.cos(w)]
    return np.column_stack(cols)


def logit(kt):
    kt = np.clip(kt, KT_EPS, 1 - KT_EPS)
    return np.log(kt / (1 - kt))


def expit(z):
    return 1.0 / (1.0 + np.exp(-z))


class Deseasonaliser:
    """Harmonic day-of-year climatology of mean and std, fitted on training data only."""

    def __init__(self, harmonics: int = 3):
        self.k = harmonics

    def fit(self, frame: pd.DataFrame):
        doy = frame.index.dayofyear.values
        X = fourier(doy, self.k)
        self.mu_, self.sd_ = {}, {}
        for col in frame.columns:
            y = frame[col].values.astype(float)
            ok = np.isfinite(y)
            bm, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
            r = y - X @ bm
            bs, *_ = np.linalg.lstsq(X[ok], np.log(r[ok] ** 2 + 1e-8), rcond=None)
            # log-variance regression is biased by E[log chi2_1] = -1.27; correct it
            self.mu_[col], self.sd_[col] = bm, bs
            sd = np.exp(0.5 * (X[ok] @ bs + 1.2704))
            self.sd_scale_ = getattr(self, "sd_scale_", {})
            self.sd_scale_[col] = np.std(r[ok] / sd)
        self.cols_ = list(frame.columns)
        return self

    def mean(self, idx, col):
        return fourier(idx.dayofyear.values, self.k) @ self.mu_[col]

    def std(self, idx, col):
        X = fourier(idx.dayofyear.values, self.k)
        return np.exp(0.5 * (X @ self.sd_[col] + 1.2704)) * self.sd_scale_[col]

    def transform(self, frame):
        out = pd.DataFrame(index=frame.index)
        for col in self.cols_:
            out[col] = (frame[col].values - self.mean(frame.index, col)) / self.std(frame.index, col)
        return out

    def inverse(self, anomalies: np.ndarray, idx: pd.DatetimeIndex) -> np.ndarray:
        """anomalies [..., T, d] -> physical values."""
        mu = np.stack([self.mean(idx, c) for c in self.cols_], -1)
        sd = np.stack([self.std(idx, c) for c in self.cols_], -1)
        return anomalies * sd + mu


def prepare_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Physical covariates in modelling scale (precipitation log1p, wind direction as vector)."""
    cov = pd.DataFrame(index=df.index)
    cov["Temp"] = df["Temp"]
    cov["Hum"] = df["Hum"]
    cov["Prec"] = np.log1p(df["Prec"].clip(lower=0))
    cov["Ps"] = df["Ps"]
    cov["WS"] = df["WS"]
    cov["WDs"] = df["WS"] * np.sin(np.deg2rad(df["WD"]))   # zonal-like component
    cov["WDc"] = df["WS"] * np.cos(np.deg2rad(df["WD"]))   # meridional-like component
    return cov


class CovTransform:
    """Origin-specific transform of physical covariates to standardised anomalies.

    * Precipitation (zero-inflated): randomised probability-integral transform with the
      training-period monthly empirical CDF, then a normal score (continuous latent),
      inverted by monthly empirical quantiles (restores the exact dry-day frequency).
    * Other covariates: harmonic day-of-year mean/std deseasonalisation.
    All statistics are fitted on data before the forecast origin.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed

    def _prec_latent(self, s: pd.Series, rng) -> pd.Series:
        from scipy.stats import norm
        out = np.empty(len(s))
        for m in range(1, 13):
            msk = s.index.month == m
            ref = self.prec_ref_[m]
            v = s.values[msk]
            lo = np.searchsorted(ref, v, side="left") / len(ref)
            hi = np.searchsorted(ref, v, side="right") / len(ref)
            u = lo + (hi - lo) * rng.uniform(size=len(v))
            out[msk] = norm.ppf(np.clip(u, 1e-4, 1 - 1e-4))
        return pd.Series(out, index=s.index)

    def fit(self, cov: pd.DataFrame):
        self.prec_ref_ = {m: np.sort(cov["Prec"].values[cov.index.month == m]) for m in range(1, 13)}
        lat = cov.copy()
        lat["Prec"] = self._prec_latent(cov["Prec"], np.random.default_rng(self.seed))
        self.ds_ = Deseasonaliser().fit(lat)
        self.cols_ = list(cov.columns)
        return self

    def transform(self, cov: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
        lat = cov.copy()
        lat["Prec"] = self._prec_latent(cov["Prec"], np.random.default_rng(self.seed if seed is None else seed))
        return self.ds_.transform(lat)

    def inverse(self, anomalies: np.ndarray, idx: pd.DatetimeIndex) -> np.ndarray:
        from scipy.stats import norm
        phys = self.ds_.inverse(anomalies, idx)
        j = self.cols_.index("Prec")
        u = norm.cdf(phys[..., j])
        out = np.empty_like(u)
        for m in range(1, 13):
            msk = idx.month == m
            ref = self.prec_ref_[m]
            out[..., msk] = np.quantile(ref, u[..., msk])
        phys[..., j] = out
        return phys
