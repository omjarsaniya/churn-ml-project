import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from preprocess import clean_data, split_data  # noqa: E402


def make_raw_df():
    return pd.DataFrame({
        "customerID": ["0001", "0002", "0003"],
        "TotalCharges": ["100.5", " ", "300"],   # note the blank string — real data has these
        "gender": ["Male", "Female", "Male"],
        "Churn": ["Yes", "No", "No"],
    })


def test_clean_data_drops_customer_id():
    df = clean_data(make_raw_df())
    assert "customerID" not in df.columns


def test_clean_data_converts_blank_total_charges_to_numeric():
    df = clean_data(make_raw_df())
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])
    assert df["TotalCharges"].isnull().sum() == 0


def test_clean_data_fills_blank_with_median_not_zero():
    df = clean_data(make_raw_df())
    # the blank row should be filled with the median of (100.5, 300) = 200.25,
    # not silently dropped or zeroed
    assert df.loc[1, "TotalCharges"] == pytest.approx(200.25)


def test_split_data_maps_churn_to_binary():
    df = clean_data(make_raw_df())
    X, y = split_data(df)
    assert "Churn" not in X.columns
    assert set(y.unique()) <= {0, 1}
    assert y.tolist() == [1, 0, 0]
