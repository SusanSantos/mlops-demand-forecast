"""Treino com MLflow: compara baseline ingênuo vs. modelo e registra métricas/artefatos."""
from __future__ import annotations

import argparse
import json

import joblib
import mlflow
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from src.data import FEATURES, TARGET, add_features, make_sales


def mape(y, p) -> float:
    y, p = np.asarray(y), np.asarray(p)
    return float(np.mean(np.abs(y - p) / np.abs(y)))


def train(model_path: str = "model.joblib", tracking_uri: str = "sqlite:///mlflow.db") -> dict:
    df = add_features(make_sales(730, seed=0))
    cut = int(len(df) * 0.8)  # split temporal, nunca aleatório
    tr, te = df.iloc[:cut], df.iloc[cut:]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("demand-forecast")
    with mlflow.start_run():
        base_pred = te["lag_7"]  # baseline: mesma data da semana anterior
        m_base = mape(te[TARGET], base_pred)
        model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0)
        model.fit(tr[FEATURES], tr[TARGET])
        pred = model.predict(te[FEATURES])
        metrics = {"mape_baseline": m_base, "mape_model": mape(te[TARGET], pred),
                   "mae_model": float(mean_absolute_error(te[TARGET], pred))}
        mlflow.log_params({"model": "GradientBoosting", "n_estimators": 300, "max_depth": 3})
        mlflow.log_metrics(metrics)
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
    ref = tr[FEATURES + [TARGET]]
    ref.to_csv("reference.csv", index=False)
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    print(train())
