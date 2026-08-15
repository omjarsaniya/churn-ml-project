import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pipeline import create_pipeline  # noqa: E402


def make_synthetic_data(n=100, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "tenure": rng.integers(0, 72, n),
        "MonthlyCharges": rng.uniform(20, 120, n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
    })
    y = pd.Series(rng.integers(0, 2, n))
    return X, y


def test_pipeline_fits_without_error():
    X, y = make_synthetic_data()
    pipeline = create_pipeline(X)
    pipeline.fit(X, y)  # should not raise


def test_pipeline_predict_proba_is_valid():
    X, y = make_synthetic_data()
    pipeline = create_pipeline(X)
    pipeline.fit(X, y)

    probs = pipeline.predict_proba(X)[:, 1]
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0
    assert len(probs) == len(X)


def test_pipeline_predict_returns_binary_labels():
    X, y = make_synthetic_data()
    pipeline = create_pipeline(X)
    pipeline.fit(X, y)

    preds = pipeline.predict(X)
    assert set(np.unique(preds)) <= {0, 1}


def test_pipeline_handles_unseen_categories_at_inference():
    """OneHotEncoder(handle_unknown='ignore') should not crash on a category
    it never saw during training — this matters because production traffic
    will eventually include a value the training data didn't have."""
    X, y = make_synthetic_data()
    pipeline = create_pipeline(X)
    pipeline.fit(X, y)

    X_new = X.iloc[[0]].copy()
    X_new["Contract"] = "Some Brand New Contract Type"
    probs = pipeline.predict_proba(X_new)  # should not raise
    assert probs.shape == (1, 2)
