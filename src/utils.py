# src/utils.py
# RetailPulse – Shared Utility Functions
#
# Small helper functions used across multiple modules.
# Keeping them here avoids code duplication.

import os
import sys

import pandas as pd
import numpy as np


def ensure_directories(*paths: str) -> None:
    """
    Create directories if they do not already exist.

    Args:
        *paths: One or more directory paths to create.
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)


def format_currency(value: float, symbol: str = "£") -> str:
    """
    Format a numeric value as a currency string.

    Examples:
        format_currency(1234567.89) → "£1,234,568"
        format_currency(999.5, "$") → "$1,000"

    Args:
        value:  Numeric value to format.
        symbol: Currency symbol prefix.

    Returns:
        Formatted currency string.
    """
    return f"{symbol}{value:,.0f}"


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Percentage Error (MAPE).

    MAPE = mean(|actual - predicted| / |actual|) × 100

    A small epsilon is added to the denominator to avoid division by zero.

    Args:
        y_true: Array of actual values.
        y_pred: Array of predicted values.

    Returns:
        MAPE as a percentage (e.g., 8.5 means 8.5% error).
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)


def load_csv_safe(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Load a CSV file with a clear error message if the file is missing.

    Args:
        filepath: Path to the CSV file.
        **kwargs: Additional keyword arguments passed to pd.read_csv.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: With a helpful message if the file is missing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            "Ensure you have run the pipeline (python run_pipeline.py) first."
        )
    return pd.read_csv(filepath, **kwargs)


def describe_dataframe(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Print a quick summary of a DataFrame (shape, dtypes, missing values).

    Useful for debugging data issues.

    Args:
        df:   The DataFrame to describe.
        name: Label to display in the output.
    """
    print(f"\n── {name} ──")
    print(f"  Shape   : {df.shape}")
    print(f"  Columns : {list(df.columns)}")
    missing = df.isnull().sum()
    if missing.any():
        print(f"  Missing :\n{missing[missing > 0].to_string()}")
    else:
        print("  Missing : None ✓")
    print(f"  Dtypes  :\n{df.dtypes.to_string()}")
