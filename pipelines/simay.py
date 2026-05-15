import numpy as np
import pandas as pd


def label_uploaded_raw_data(input_df):
    labeled_df = input_df.copy()
    labeled_df["time"] = pd.to_numeric(labeled_df["time"], errors="coerce")
    labeled_df["delta_lambda_noisy"] = pd.to_numeric(labeled_df["delta_lambda_noisy"], errors="coerce")
    labeled_df = labeled_df.dropna(subset=["time", "delta_lambda_noisy"]).reset_index(drop=True)

    mean_value = float(labeled_df["delta_lambda_noisy"].mean())
    std_value = float(labeled_df["delta_lambda_noisy"].std())
    if not np.isfinite(std_value) or std_value == 0.0:
        std_value = max(abs(mean_value) * 0.05, 1e-6)

    lower_threshold = mean_value - 0.5 * std_value
    upper_threshold = mean_value + 0.5 * std_value
    labeled_df["label"] = np.where(
        labeled_df["delta_lambda_noisy"] <= lower_threshold,
        "normal",
        np.where(labeled_df["delta_lambda_noisy"] <= upper_threshold, "mild_damage", "severe_damage"),
    )
    return labeled_df


def run_simay_pipeline(input_df, force_relabel=False):
    simay_df = input_df.copy()
    simay_df["time"] = pd.to_numeric(simay_df["time"], errors="coerce")
    simay_df["delta_lambda_noisy"] = pd.to_numeric(simay_df["delta_lambda_noisy"], errors="coerce")
    simay_df = simay_df.dropna(subset=["time", "delta_lambda_noisy"]).reset_index(drop=True)

    if "label" in simay_df.columns and not force_relabel:
        simay_df["label"] = simay_df["label"].astype(str).str.strip()
    else:
        simay_df = label_uploaded_raw_data(simay_df)
    return simay_df
