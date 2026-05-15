from .data_source import MODEL_DIR, PROJECT_DIR


def artifact_path(filename):
    model_path = MODEL_DIR / filename
    flat_path = PROJECT_DIR / filename
    if model_path.exists():
        return model_path
    if flat_path.exists():
        return flat_path
    return model_path


def artifacts_exist(artifact_names):
    return all(artifact_path(filename).exists() for filename in artifact_names.values())
