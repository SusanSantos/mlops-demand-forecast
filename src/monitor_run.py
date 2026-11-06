"""Simula produção: avalia o modelo em dados novos (com ou sem drift) e decide retreino."""
from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd

from src.data import FEATURES, TARGET, add_features, make_sales
from src.monitoring import drift_report, needs_retrain
from src.train import mape


def main(drift: bool) -> None:
    model = joblib.load("model.joblib")
    ref = pd.read_csv("reference.csv")
    cur = add_features(make_sales(730, seed=1, drift_after=500 if drift else None)).iloc[-180:]
    rep = drift_report(ref, cur, FEATURES)
    with open("metrics.json") as f:
        m_ref = json.load(f)["mape_model"]  # MAPE de holdout (não in-sample)
    m_cur = mape(cur[TARGET], model.predict(cur[FEATURES]))
    print(rep.to_string(index=False))
    print(f"\nMAPE ref={m_ref:.3%}  atual={m_cur:.3%}")
    print("RETREINAR" if needs_retrain(rep, m_ref, m_cur) else "Modelo saudável")
    rep.to_csv("drift_report.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift", action="store_true")
    main(ap.parse_args().drift)
