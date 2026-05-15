import pandas as pd


def run_aleyna_pipeline(input_df, window_size=15):
    working_df = input_df.copy()
    if "delta_lambda_noisy" not in working_df.columns:
        raise ValueError("Aleyna adımı için 'delta_lambda_noisy' sütunu gerekli.")

    noisy = pd.to_numeric(working_df["delta_lambda_noisy"], errors="coerce")
    filtered = noisy.rolling(window=window_size, center=True, min_periods=1).mean()
    working_df["delta_lambda_filtered"] = filtered
    return working_df
