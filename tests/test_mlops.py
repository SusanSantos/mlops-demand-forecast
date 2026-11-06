import numpy as np
import pytest

from src.data import FEATURES, add_features, make_sales
from src.monitoring import drift_report, needs_retrain, psi
from src.train import mape, train


def test_psi_zero_for_same_distribution():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x) < 1e-6


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(0, 1, 5000), rng.normal(2, 1, 5000)) > 0.25


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    import os
    d = tmp_path_factory.mktemp("run")
    cwd = os.getcwd()
    os.chdir(d)
    try:
        m = train("model.joblib", f"sqlite:///{d}/mlflow.db")
    finally:
        os.chdir(cwd)
    return m


def test_model_beats_baseline(trained):
    assert trained["mape_model"] < trained["mape_baseline"]


def test_no_drift_vs_drift():
    ref = add_features(make_sales(730, seed=0)).iloc[:500]
    ok = add_features(make_sales(730, seed=1)).iloc[-180:]
    bad = add_features(make_sales(730, seed=1, drift_after=500)).iloc[-180:]
    r_ok, r_bad = drift_report(ref, ok, FEATURES), drift_report(ref, bad, FEATURES)
    assert (r_ok["status"] == "ALERT").sum() < 2
    assert (r_bad["status"] == "ALERT").sum() >= 2
    assert needs_retrain(r_bad, 0.05, 0.05)
    assert not needs_retrain(r_ok, 0.05, 0.05)


def test_mape():
    assert mape([100], [90]) == pytest.approx(0.1)
