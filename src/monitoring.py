"""Detecção de drift: PSI (feature) + KS (feature) + degradação de erro (MAPE)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PSI_WARN, PSI_ALERT = 0.10, 0.25


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def seasonal_match(ref: pd.DataFrame, cur: pd.DataFrame, col: str = "doy") -> pd.DataFrame:
    """Restringe a referência ao mesmo período do ano da janela atual.
    Sem isso, sazonalidade normal é confundida com drift."""
    lo, hi = cur[col].min(), cur[col].max()
    matched = ref[(ref[col] >= lo) & (ref[col] <= hi)]
    return matched if len(matched) >= 30 else ref


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    ref = seasonal_match(ref, cur)
    rows = []
    for f in features:
        p = psi(ref[f].to_numpy(), cur[f].to_numpy())
        ks = ks_2samp(ref[f], cur[f])
        status = "ALERT" if p >= PSI_ALERT else "WARN" if p >= PSI_WARN else "OK"
        rows.append({"feature": f, "psi": round(p, 4), "ks_stat": round(float(ks.statistic), 4),
                     "ks_pvalue": float(ks.pvalue), "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def needs_retrain(report: pd.DataFrame, mape_ref: float, mape_cur: float, tol: float = 1.5, min_alerts: int = 2) -> bool:
    # Um único alerta de PSI é ruído comum (features de baixa variância, como médias móveis,
    # deixam o PSI hipersensível). Exige corroboração: >= min_alerts features OU erro degradado.
    return bool((report["status"] == "ALERT").sum() >= min_alerts or mape_cur > mape_ref * tol)
