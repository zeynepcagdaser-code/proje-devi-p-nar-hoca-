def _normalize_column_name(name):
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def _canonical_column_name(name):
    normalized = _normalize_column_name(name)
    alias_map = {
        "time": "time",
        "zaman": "time",
        "timestamp": "time",
        "delta_lambda_noisy": "delta_lambda_noisy",
        "delta_lambda": "delta_lambda_noisy",
        "fbg_signal": "delta_lambda_noisy",
        "fgb_signal": "delta_lambda_noisy",
        "signal": "delta_lambda_noisy",
        "sinyal": "delta_lambda_noisy",
        "delta_lambda_simulink": "delta_lambda_simulink",
        "simulink_signal": "delta_lambda_simulink",
        "label": "label",
        "damage_label": "label",
        "damageclass": "label",
        "damage_class": "label",
        "class": "label",
        "sinif": "label",
        "hasar": "label",
    }
    return alias_map.get(normalized, normalized)


def validate_raw_upload_columns(df):
    renamed = {}
    for col in df.columns:
        canonical = _canonical_column_name(col)
        if canonical != col:
            renamed[col] = canonical
    if renamed:
        df.rename(columns=renamed, inplace=True)

    required_columns = {"time", "delta_lambda_noisy"}
    optional_columns = {"delta_lambda_simulink", "label"}
    allowed_columns = required_columns | optional_columns

    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        raise ValueError("Ham veri icin zorunlu sutunlar eksik: " + ", ".join(missing_required))

    disallowed = [col for col in df.columns if col not in allowed_columns]
    if disallowed:
        raise ValueError(
            "Yuklenen dosya ham veri formatinda olmali. Izinli sutunlar: "
            + ", ".join(sorted(allowed_columns))
            + ". Uygun olmayan sutunlar: "
            + ", ".join(disallowed)
        )
