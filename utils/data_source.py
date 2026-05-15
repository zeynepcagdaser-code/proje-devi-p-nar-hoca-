from pathlib import Path

import pandas as pd

DEFAULT_CSV_CANDIDATES = [
    "simay/fbg_simulink_labeled_dataset (4).csv",
    "../simay/fbg_simulink_labeled_dataset (4).csv",
    "aleyna/fbg_simulink_labeled_dataset (4).csv",
    "../aleyna/fbg_simulink_labeled_dataset (4).csv",
    "fbg_simulink_labeled_dataset.csv",
    "fbg_simulink_labeled_dataset(1).csv",
    "fbg_filtered_dataset.csv",
]


def get_project_dir():
    script_dir = Path(__file__).resolve().parent.parent
    candidates = [script_dir, script_dir.parent, Path.cwd()]
    for candidate in candidates:
        for csv_name in DEFAULT_CSV_CANDIDATES:
            if (candidate / Path(csv_name)).exists():
                return candidate
    return script_dir


def resolve_default_csv_path(project_dir):
    for csv_name in DEFAULT_CSV_CANDIDATES:
        candidate = project_dir / Path(csv_name)
        if candidate.exists():
            return candidate
    return project_dir / Path(DEFAULT_CSV_CANDIDATES[0])


PROJECT_DIR = get_project_dir()
MODEL_DIR = PROJECT_DIR / "models"
DEFAULT_CSV = resolve_default_csv_path(PROJECT_DIR)


def load_dataset(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file), uploaded_file.name
    return pd.read_csv(DEFAULT_CSV), str(DEFAULT_CSV)


def source_label(source):
    return Path(source).name
