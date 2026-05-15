def validate_raw_upload_columns(df):
    required_columns = {"time", "delta_lambda_noisy"}
    optional_columns = {"delta_lambda_simulink", "label"}
    allowed_columns = required_columns | optional_columns

    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        raise ValueError("Ham veri için zorunlu sütunlar eksik: " + ", ".join(missing_required))

    disallowed = [col for col in df.columns if col not in allowed_columns]
    if disallowed:
        raise ValueError(
            "Yüklenen dosya ham veri formatında olmalı. İzinli sütunlar: "
            + ", ".join(sorted(allowed_columns))
            + ". Uygun olmayan sütunlar: "
            + ", ".join(disallowed)
        )
