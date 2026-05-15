import numpy as np
import pandas as pd


def compute_signal_features(signal_values, time_values):
    signal = np.asarray(signal_values, dtype=np.float64)
    time = np.asarray(time_values, dtype=np.float64)

    mean_shift = float(np.mean(signal))
    std_value = float(np.std(signal))
    rms_value = float(np.sqrt(np.mean(np.square(signal))))
    max_value = float(np.max(signal))
    min_value = float(np.min(signal))
    peak_to_peak = max_value - min_value
    amplitude = max_value - mean_shift

    if len(time) > 1:
        ts = float(np.mean(np.diff(time)))
        fs = 1.0 / ts if ts > 0 else 1.0
    else:
        fs = 1.0

    y_fft = np.fft.fft(signal)
    length = len(signal)
    p2 = np.abs(y_fft / max(length, 1))
    p1 = p2[: length // 2 + 1]
    freq = fs * np.arange(0, length // 2 + 1) / max(length, 1)
    spectral_energy = float(np.sum(np.square(p1)))
    dominant_frequency = float(freq[np.argmax(p1[1:]) + 1]) if len(p1) > 1 else 0.0

    return {
        "MeanShift": mean_shift,
        "StdDev": std_value,
        "RMS": rms_value,
        "MaxValue": max_value,
        "MinValue": min_value,
        "PeakToPeak": peak_to_peak,
        "Amplitude": amplitude,
        "SpectralEnergy": spectral_energy,
        "DominantFrequency": dominant_frequency,
    }


def run_gizem_pipeline(filtered_df):
    if "delta_lambda_filtered" not in filtered_df.columns:
        raise ValueError("Gizem adımı için 'delta_lambda_filtered' sütunu gerekli.")
    if "time" not in filtered_df.columns:
        raise ValueError("Gizem adımı için 'time' sütunu gerekli.")

    gizem_df = filtered_df.copy()
    if "label" in gizem_df.columns:
        label_series = gizem_df["label"].astype(str).str.strip()
    else:
        label_series = pd.Series(["unknown"] * len(gizem_df), index=gizem_df.index)

    label_map = {"normal": 0, "mild_damage": 1, "severe_damage": 2}
    gizem_df["numeric_label"] = label_series.map(label_map).fillna(-1).astype(int)
    feature_row = compute_signal_features(gizem_df["delta_lambda_filtered"].values, gizem_df["time"].values)
    feature_table = pd.DataFrame([feature_row])
    return gizem_df, feature_table
