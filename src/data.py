from __future__ import annotations

import numpy as np
import pandas as pd


def make_sales(days: int = 730, seed: int = 0, drift_after: int | None = None) -> pd.DataFrame:
    """Vendas diárias com tendência, sazonalidade semanal/anual, promoções e ruído.
    drift_after: a partir deste dia a demanda muda de regime (para simular drift)."""
    rng = np.random.default_rng(seed)
    t = np.arange(days)
    promo = (rng.random(days) < 0.12).astype(int)
    temp = 22 + 6 * np.sin(2 * np.pi * t / 365) + rng.normal(0, 1.5, days)
    base = 100 + 0.01 * t + 15 * np.sin(2 * np.pi * t / 7) + 10 * np.sin(2 * np.pi * t / 365)
    y = base + 30 * promo + 1.2 * (temp - 22) + rng.normal(0, 5, days)
    if drift_after is not None:
        m = t >= drift_after
        y[m] = y[m] * 1.35 + 40 * promo[m]
        temp[m] += 5
        promo[m] = (rng.random(m.sum()) < 0.30).astype(int)
    return pd.DataFrame({"date": pd.date_range("2024-01-01", periods=days), "promo": promo,
                         "temp": temp, "units": y})


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["dow"] = d["date"].dt.dayofweek
    d["doy"] = d["date"].dt.dayofyear
    d["lag_7"] = d["units"].shift(7)
    d["lag_14"] = d["units"].shift(14)
    d["roll_7"] = d["units"].shift(1).rolling(7).mean()
    return d.dropna().reset_index(drop=True)


FEATURES = ["promo", "temp", "dow", "doy", "lag_7", "lag_14", "roll_7"]
TARGET = "units"
