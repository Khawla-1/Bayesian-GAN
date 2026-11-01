"""Statistical and foundation-model baselines. Every function returns an ensemble of the
clearness index Kt with shape [M, T] for the target dates (GHI = Kt * TOA afterwards)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .data import expit, logit

warnings.filterwarnings("ignore")


def climatology(kt_train: pd.Series, dates: pd.DatetimeIndex, M=200, window=7, seed=0):
    """Analog-year climatology: each member is a whole pre-origin year (day-of-year aligned)
    shifted by a random offset of up to +-7 days, so members are temporally coherent and
    monthly/annual aggregates of the ensemble have a realistic spread. The daily marginal
    is the pooled +-7-day climatological distribution."""
    rng = np.random.default_rng(seed)
    kt = kt_train.interpolate(limit_direction="both")
    years = [y for y in sorted(set(kt.index.year)) if (kt.index.year == y).sum() >= 360]
    clim = {y: kt[kt.index.year == y].groupby(kt[kt.index.year == y].index.dayofyear).mean()
                 .reindex(range(1, 367)).interpolate(limit_direction="both").values for y in years}
    doy = np.minimum(dates.dayofyear.values, 366) - 1
    out = np.empty((M, len(dates)))
    for m in range(M):
        shift = rng.integers(-window, window + 1)
        # a new analog year is drawn for every target calendar year
        for yr in sorted(set(dates.year)):
            sel = dates.year == yr
            src = clim[years[rng.integers(len(years))]]
            out[m, sel] = src[(doy[sel] + shift) % 366]
    return out


def seasonal_naive(kt_train: pd.Series, dates):
    last = kt_train.dropna()
    last = last[last.index > last.index[-1] - pd.Timedelta(days=366)]
    m = last.groupby(last.index.dayofyear).mean().reindex(range(1, 367)).interpolate(limit_direction="both")
    return m.reindex(dates.dayofyear).values[None, :]


def arima(zanom_train: np.ndarray, H: int, M=200, seed=0):
    """ARIMA(1,0,1) on z; coherent predictive paths simulated from the fitted model."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    r = SARIMAX(zanom_train, order=(1, 0, 1), trend="c").fit(disp=False)
    np.random.seed(seed)
    return np.asarray(r.simulate(H, repetitions=M, anchor="end")).reshape(H, -1)[:, :M].T


def bsts(z_train: pd.Series, H: int, M=200, seed=0):
    """Structural time series (local linear trend + 3 annual harmonics + AR(1)) estimated
    by maximum likelihood with the Kalman filter (statsmodels UnobservedComponents)."""
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    mod = UnobservedComponents(z_train.values, level="lltrend",
                               freq_seasonal=[{"period": 365.25, "harmonics": 3}], autoregressive=1)
    r = mod.fit(disp=False, maxiter=200)
    # coherent paths from the BSTS predictive distribution (simulation smoother, plug-in ML
    # variances): y = mu + gamma + AR + eps
    np.random.seed(seed)
    zs = np.asarray(r.simulate(H, repetitions=M, anchor="end")).reshape(H, -1)[:, :M].T
    return expit(zs), r


def prophet(z_train: pd.Series, dates, M=200, seed=0):
    import logging
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    from prophet import Prophet
    np.random.seed(seed)
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                uncertainty_samples=M)
    m.fit(pd.DataFrame({"ds": z_train.index, "y": z_train.values}))
    s = m.predictive_samples(pd.DataFrame({"ds": dates}))["yhat"]       # [T, M]
    return expit(s.T)


def foundation(name: str, kt_ctx: np.ndarray, H: int, M=200):
    """Zero-shot foundation models. Requires the model weights (Hugging Face Hub).
    Returns None when weights cannot be obtained (the run is then reported as not run)."""
    try:
        if name == "chronos":
            import torch
            from chronos import BaseChronosPipeline
            pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cpu")
            q, _ = pipe.predict_quantiles(torch.tensor(kt_ctx[-2048:], dtype=torch.float32)[None],
                                          prediction_length=H, quantile_levels=list(np.linspace(0.05, 0.95, 19)))
            q = q[0].numpy().T                                          # [19, H]
            u = np.random.default_rng(0).uniform(0.05, 0.95, (M, 1))
            return np.stack([np.interp(u[:, 0], np.linspace(0.05, 0.95, 19), q[:, t]) for t in range(H)], 1)
        if name == "timesfm":
            import timesfm
            tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
            tfm.compile(timesfm.ForecastConfig(max_context=2048, max_horizon=H, use_continuous_quantile_head=True))
            pt, qt = tfm.forecast(horizon=H, inputs=[kt_ctx[-2048:]])
            q = qt[0][:, 1:10].T                                        # deciles 0.1..0.9
            u = np.random.default_rng(0).uniform(0.1, 0.9, M)
            return np.stack([np.interp(u, np.linspace(0.1, 0.9, 9), q[:, t]) for t in range(H)], 1)
        if name == "moment":
            import torch
            from momentfm import MOMENTPipeline
            mdl = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-large",
                                                 model_kwargs={"task_name": "forecasting", "forecast_horizon": H})
            mdl.init()
            x = torch.tensor(kt_ctx[-512:], dtype=torch.float32)[None, None]
            pt = mdl(x_enc=x).forecast[0, 0].detach().numpy()
            return pt[None]                                             # point forecast (zero-shot head)
    except Exception as e:  # noqa: BLE001
        print(f"[foundation:{name}] not run: {type(e).__name__}: {str(e)[:120]}")
        return None
