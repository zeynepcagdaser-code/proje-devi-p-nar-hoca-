import json
import os
import random
from collections import Counter

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    LSTM,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling1D,
)
from tensorflow.keras.models import Sequential, load_model
from pipelines import run_aleyna_pipeline, run_gizem_pipeline, run_simay_pipeline
from utils import MODEL_DIR, PROJECT_DIR, artifact_path, artifacts_exist, load_dataset, source_label, validate_raw_upload_columns

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass


st.set_page_config(page_title="FBG LSTM + CNN Dashboard", layout="wide")


LSTM_ARTIFACTS = {
    "model": "fbg_lstm_model.keras",
    "scaler": "scaler.joblib",
    "encoder": "label_encoder.joblib",
    "config": "model_config.json",
}
LSTM_RECIPE_VERSION = "v1_recipe_search"

CNN_ARTIFACTS = {
    "model": "fbg_1dcnn_model.keras",
    "scaler": "cnn_scaler.joblib",
    "encoder": "cnn_label_encoder.joblib",
    "config": "cnn_model_config.json",
}
CNN_RECIPE_VERSION = "v3_recipe_search"

PIPELINE_OUTPUT_DIR = PROJECT_DIR / "pipeline_outputs"


def persist_pipeline_outputs(simay_df, aleyna_df, gizem_df, gizem_features_df):
    PIPELINE_OUTPUT_DIR.mkdir(exist_ok=True)
    simay_path = PIPELINE_OUTPUT_DIR / "simay_labeled_output.csv"
    aleyna_path = PIPELINE_OUTPUT_DIR / "aleyna_filtered_output.csv"
    gizem_path = PIPELINE_OUTPUT_DIR / "gizem_processed_output.csv"
    gizem_features_path = PIPELINE_OUTPUT_DIR / "gizem_features_output.csv"

    simay_df.to_csv(simay_path, index=False)
    aleyna_df.to_csv(aleyna_path, index=False)
    gizem_df.to_csv(gizem_path, index=False)
    gizem_features_df.to_csv(gizem_features_path, index=False)

    return simay_path, aleyna_path, gizem_path, gizem_features_path


def validate_columns(df, feature_column, needs_label):
    required = [feature_column]
    if needs_label:
        required.append("label")
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("Eksik sütunlar: " + ", ".join(missing))


def make_windows(df, feature_column, label_column, window_size, stride=1):
    x_values = pd.to_numeric(df[feature_column], errors="coerce")

    if label_column and label_column in df.columns:
        valid_mask = x_values.notna() & df[label_column].notna()
    else:
        valid_mask = x_values.notna()

    series = x_values[valid_mask].astype(float).values
    labels = None
    if label_column and label_column in df.columns:
        labels = df.loc[valid_mask, label_column].astype(str).str.strip().values

    if len(series) < window_size:
        raise ValueError("Veri uzunluğu pencere boyutundan küçük.")

    x_windows = []
    y_windows = []

    for start in range(0, len(series) - window_size + 1, stride):
        end = start + window_size
        x_windows.append(series[start:end])

        if labels is not None:
            window_labels = labels[start:end]
            dominant_label = Counter(window_labels).most_common(1)[0][0]
            y_windows.append(dominant_label)

    x_windows = np.array(x_windows, dtype=np.float32)[..., np.newaxis]

    if labels is None:
        return x_windows, None

    return x_windows, np.array(y_windows)


def scale_lstm_train_test(x_train, x_test):
    scaler = StandardScaler()
    feature_count = x_train.shape[2]
    x_train_2d = x_train.reshape(-1, feature_count)
    x_test_2d = x_test.reshape(-1, feature_count)
    x_train_scaled = scaler.fit_transform(x_train_2d).reshape(x_train.shape)
    x_test_scaled = scaler.transform(x_test_2d).reshape(x_test.shape)
    return x_train_scaled, x_test_scaled, scaler


def scale_cnn_full_windows(x_windows):
    scaler = MinMaxScaler()
    feature_count = x_windows.shape[2]
    x_2d = x_windows.reshape(-1, feature_count)
    x_scaled = scaler.fit_transform(x_2d).reshape(x_windows.shape)
    return x_scaled, scaler


def build_lstm_model(window_size, feature_count, class_count, recipe):
    model = Sequential(
        [
            Input(shape=(window_size, feature_count)),
            LSTM(recipe["lstm_units"]),
            Dropout(recipe["dropout"]),
            Dense(recipe["dense_units"], activation="relu"),
            Dense(class_count, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=recipe["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def lstm_recipes():
    return [
        {
            "name": "lstm_w16_s2_e40",
            "window_size": 16,
            "stride": 2,
            "epochs": 40,
            "lstm_units": 64,
            "dense_units": 32,
            "dropout": 0.25,
            "learning_rate": 0.0010,
        },
        {
            "name": "lstm_w32_s2_e50",
            "window_size": 32,
            "stride": 2,
            "epochs": 50,
            "lstm_units": 64,
            "dense_units": 32,
            "dropout": 0.30,
            "learning_rate": 0.0007,
        },
        {
            "name": "lstm_w48_s4_e60",
            "window_size": 48,
            "stride": 4,
            "epochs": 60,
            "lstm_units": 96,
            "dense_units": 48,
            "dropout": 0.30,
            "learning_rate": 0.0007,
        },
    ]


def build_cnn_model(window_size, feature_count, class_count, recipe):
    model = Sequential(
        [
            Input(shape=(window_size, feature_count)),
            Conv1D(16, kernel_size=3, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(recipe["dropout1"]),
            Conv1D(32, kernel_size=3, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(recipe["dropout2"]),
            Flatten(),
            Dense(recipe["dense_units"], activation="relu"),
            Dropout(recipe["dropout3"]),
            Dense(class_count, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=recipe["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def cnn_recipes():
    return [
        {
            "name": "stable_balanced",
            "dropout1": 0.10,
            "dropout2": 0.20,
            "dropout3": 0.25,
            "dense_units": 96,
            "learning_rate": 0.0003,
            "use_class_weight": True,
        },
        {
            "name": "high_capacity_low_lr",
            "dropout1": 0.05,
            "dropout2": 0.15,
            "dropout3": 0.20,
            "dense_units": 128,
            "learning_rate": 0.0005,
            "use_class_weight": False,
        },
        {
            "name": "balanced_plus_low_lr",
            "dropout1": 0.10,
            "dropout2": 0.20,
            "dropout3": 0.20,
            "dense_units": 96,
            "learning_rate": 0.0002,
            "use_class_weight": True,
        },
    ]


def save_artifacts(artifact_names, model, scaler, encoder, config, save_model=True):
    MODEL_DIR.mkdir(exist_ok=True)
    if save_model:
        model.save(MODEL_DIR / artifact_names["model"])
    joblib.dump(scaler, MODEL_DIR / artifact_names["scaler"])
    joblib.dump(encoder, MODEL_DIR / artifact_names["encoder"])
    (MODEL_DIR / artifact_names["config"]).write_text(
        json.dumps(config, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )


def load_artifacts(artifact_names):
    model = load_model(artifact_path(artifact_names["model"]))
    scaler = joblib.load(artifact_path(artifact_names["scaler"]))
    encoder = joblib.load(artifact_path(artifact_names["encoder"]))
    config = json.loads(artifact_path(artifact_names["config"]).read_text(encoding="utf-8"))
    return model, scaler, encoder, config


def store_active_model(model_type, model, scaler, encoder, config):
    st.session_state["active_model_type"] = model_type
    st.session_state["active_model"] = model
    st.session_state["active_scaler"] = scaler
    st.session_state["active_encoder"] = encoder
    st.session_state["active_config"] = config


def active_model_ready():
    return all(
        key in st.session_state
        for key in ["active_model_type", "active_model", "active_scaler", "active_encoder", "active_config"]
    )


def predict_window(values):
    model = st.session_state["active_model"]
    scaler = st.session_state["active_scaler"]
    encoder = st.session_state["active_encoder"]
    config = st.session_state["active_config"]

    window_size = int(config["window_size"])
    feature_count = int(config["feature_count"])
    values = np.array(values, dtype=np.float32).reshape(window_size, feature_count)
    values_2d = values.reshape(-1, feature_count)
    values_scaled = scaler.transform(values_2d).reshape(1, window_size, feature_count)

    probabilities = model.predict(values_scaled, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_label = encoder.inverse_transform([predicted_index])[0]
    confidence = float(probabilities[predicted_index])
    return predicted_label, confidence, probabilities


def plot_signal(df, feature_column):
    fig, ax = plt.subplots(figsize=(12, 4))
    x_axis = df["time"] if "time" in df.columns else df.index
    ax.plot(x_axis, df[feature_column], label=feature_column)
    ax.set_xlabel("Zaman")
    ax.set_ylabel("Sinyal değeri")
    ax.set_title("FBG Sinyal Grafiği")
    ax.grid(True)
    ax.legend()
    return fig


def plot_history(history, title_prefix):
    def smooth_curve(values, alpha=0.35):
        if not values:
            return values
        smoothed = [values[0]]
        for val in values[1:]:
            smoothed.append(alpha * val + (1 - alpha) * smoothed[-1])
        return smoothed

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    train_acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    axes[0].plot(train_acc, label="Eğitim Doğruluğu (ham)", alpha=0.25)
    axes[0].plot(val_acc, label="Doğrulama Doğruluğu (ham)", alpha=0.25)
    axes[0].plot(smooth_curve(train_acc), label="Eğitim Doğruluğu", linewidth=2.2)
    axes[0].plot(smooth_curve(val_acc), label="Doğrulama Doğruluğu", linewidth=2.2)
    axes[0].set_title(f"{title_prefix} Doğruluk Grafiği")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(train_loss, label="Eğitim Kaybı (ham)", alpha=0.25)
    axes[1].plot(val_loss, label="Doğrulama Kaybı (ham)", alpha=0.25)
    axes[1].plot(smooth_curve(train_loss), label="Eğitim Kaybı", linewidth=2.2)
    axes[1].plot(smooth_curve(val_loss), label="Doğrulama Kaybı", linewidth=2.2)
    axes[1].set_title(f"{title_prefix} Kayıp Grafiği")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm, class_names, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(cm, display_labels=class_names)
    display.plot(cmap="Blues", values_format="d", ax=ax)
    ax.set_title(title)
    return fig


def show_results(model_name, test_accuracy, test_loss, window_count, history, cm, encoder, y_test, y_pred):
    st.success(f"{model_name} modeli eğitildi ve kaydedildi.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Test Accuracy", round(float(test_accuracy), 4))
    col2.metric("Test Loss", round(float(test_loss), 4))
    col3.metric("Pencere Sayısı", int(window_count))

    st.pyplot(plot_history(history, model_name))
    st.pyplot(plot_confusion_matrix(cm, encoder.classes_, f"{model_name} Confusion Matrix"))

    report = classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)


def compute_best_epoch(history):
    val_loss = history.history.get("val_loss", [])
    if not val_loss:
        return None
    return int(np.argmin(val_loss) + 1)


def has_validation_degradation(history):
    val_accuracy = history.history.get("val_accuracy", [])
    train_accuracy = history.history.get("accuracy", [])
    val_loss = history.history.get("val_loss", [])
    if len(val_accuracy) < 8 or len(val_loss) < 8 or len(train_accuracy) < 8:
        return False

    best_epoch_idx = int(np.argmin(val_loss))
    val_acc_best = float(val_accuracy[best_epoch_idx])
    train_acc_best = float(train_accuracy[best_epoch_idx])
    generalization_gap = train_acc_best - val_acc_best

    recent_window = val_loss[-4:]
    recent_increase = all(recent_window[idx] >= recent_window[idx - 1] for idx in range(1, len(recent_window)))
    loss_degraded = float(val_loss[-1]) > float(np.min(val_loss)) * 1.15

    return generalization_gap > 0.08 and recent_increase and loss_degraded


def plot_cnn_history(history, best_epoch):
    def smooth_curve(values, alpha=0.35):
        if not values:
            return values
        smoothed = [values[0]]
        for val in values[1:]:
            smoothed.append(alpha * val + (1 - alpha) * smoothed[-1])
        return smoothed

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    train_acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    axes[0].plot(train_acc, label="Eğitim Doğruluğu (ham)", alpha=0.25)
    axes[0].plot(val_acc, label="Doğrulama Doğruluğu (ham)", alpha=0.25)
    axes[0].plot(smooth_curve(train_acc), label="Eğitim Doğruluğu", linewidth=2.2)
    axes[0].plot(smooth_curve(val_acc), label="Doğrulama Doğruluğu", linewidth=2.2)
    axes[0].set_title("CNN Doğruluk Grafiği")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Doğruluk")
    axes[0].grid(True)
    if best_epoch is not None:
        axes[0].axvline(best_epoch - 1, color="orange", linestyle="--", linewidth=1.5, label="En iyi epoch")
        axes[0].text(best_epoch - 1, axes[0].get_ylim()[1] * 0.95, f"En iyi epoch: {best_epoch}", color="orange")
    axes[0].legend()

    axes[1].plot(train_loss, label="Eğitim Kaybı (ham)", alpha=0.25)
    axes[1].plot(val_loss, label="Doğrulama Kaybı (ham)", alpha=0.25)
    axes[1].plot(smooth_curve(train_loss), label="Eğitim Kaybı", linewidth=2.2)
    axes[1].plot(smooth_curve(val_loss), label="Doğrulama Kaybı", linewidth=2.2)
    axes[1].set_title("CNN Kayıp Grafiği")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Kayıp")
    axes[1].grid(True)
    if best_epoch is not None:
        axes[1].axvline(best_epoch - 1, color="orange", linestyle="--", linewidth=1.5, label="En iyi epoch")
    axes[1].legend()

    fig.tight_layout()
    return fig


def show_cnn_results(test_accuracy, test_loss, window_count, best_epoch, macro_f1, history, cm, encoder, y_test, y_pred):
    st.success("CNN modeli eğitildi ve kaydedildi.")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Test Accuracy", round(float(test_accuracy), 4))
    col2.metric("Test Loss", round(float(test_loss), 4))
    col3.metric("Pencere Sayısı", int(window_count))
    col4.metric("En İyi Epoch", int(best_epoch) if best_epoch is not None else "-")
    col5.metric("Macro F1", round(float(macro_f1), 4))

    st.pyplot(plot_cnn_history(history, best_epoch))
    if best_epoch is not None:
        st.info(f"EarlyStopping aktif. En iyi epoch: {best_epoch}")

    if has_validation_degradation(history):
        st.warning("Model validation set üzerinde bozulma göstermiştir. Overfitting olabilir.")
    else:
        st.success("CNN modeli stabil şekilde eğitildi.")

    st.pyplot(plot_confusion_matrix(cm, encoder.classes_, "1D-CNN Confusion Matrix"))

    report = classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)


st.title("FBG Sensör Verisi ile LSTM + CNN Hasar Tespiti Dashboard")

st.sidebar.header("Veri ve Model")
data_mode = st.sidebar.radio(
    "Veri kaynağı",
    ["Simulink verisini kullan", "Makine öğrenimi için yeni veri yükle"],
)

uploaded_file = None
if data_mode == "Makine öğrenimi için yeni veri yükle":
    uploaded_file = st.sidebar.file_uploader("Ham veri CSV yükle", type=["csv"])
    st.sidebar.caption("Ham veri formatı: time, delta_lambda_noisy (opsiyonel: delta_lambda_simulink, label)")

try:
    if data_mode == "Makine öğrenimi için yeni veri yükle":
        if uploaded_file is None:
            st.info("Lütfen ham veri CSV dosyası yükleyin.")
            st.stop()
        df, data_source = load_dataset(uploaded_file)
        validate_raw_upload_columns(df)
        if "label" not in df.columns:
            st.error("Makine ogrenimi icin yuklenen veride `label` sutunu zorunludur.")
            st.stop()
    else:
        df, data_source = load_dataset(None)
except FileNotFoundError:
    st.error("Varsayılan CSV bulunamadı.")
    st.stop()
except ValueError as exc:
    st.error(str(exc))
    st.stop()

is_uploaded_mode = uploaded_file is not None
raw_uploaded_df = df.copy() if is_uploaded_mode else None
uploaded_missing_label = is_uploaded_mode and ("label" not in df.columns)
uploaded_has_label_raw = is_uploaded_mode and ("label" in df.columns)

simay_generated_df = None
aleyna_generated_df = None
gizem_generated_df = None
gizem_features_df = None
pipeline_output_paths = None

try:
    simay_generated_df = run_simay_pipeline(df, force_relabel=False)
    aleyna_generated_df = run_aleyna_pipeline(simay_generated_df, window_size=15)
    gizem_generated_df, gizem_features_df = run_gizem_pipeline(aleyna_generated_df)
    pipeline_output_paths = persist_pipeline_outputs(
        simay_generated_df,
        aleyna_generated_df,
        gizem_generated_df,
        gizem_features_df,
    )
except Exception as exc:
    st.error(f"Veri işleme hattı (Simay -> Aleyna -> Gizem) çalıştırılamadı: {exc}")
    st.stop()

if uploaded_missing_label and simay_generated_df is not None:
    labeled_csv_bytes = simay_generated_df.to_csv(index=False).encode("utf-8")
    st.sidebar.success("Yuklenen veri etiketlendi ve Aleyna adimi icin hazir.")
    st.sidebar.download_button(
        "Etiketli CSV Indir",
        data=labeled_csv_bytes,
        file_name="uploaded_labeled_by_simay.csv",
        mime="text/csv",
    )


model_training_df = gizem_generated_df

numeric_columns = model_training_df.select_dtypes(include=["number"]).columns.tolist()
feature_candidates = [column for column in numeric_columns if column not in {"time", "numeric_label"}]

if not feature_candidates:
    st.error("CSV içinde sayısal sinyal sütunu bulunamadı.")
    st.stop()

default_feature_index = 0
if "delta_lambda_filtered" in feature_candidates:
    default_feature_index = feature_candidates.index("delta_lambda_filtered")

feature_column = feature_candidates[default_feature_index]

has_label = "label" in model_training_df.columns
show_label_distribution = has_label and ((not is_uploaded_mode) or uploaded_has_label_raw)

st.sidebar.subheader("LSTM Ayarları")
lstm_window_size = st.sidebar.slider("LSTM pencere boyutu", 10, 128, 32, 2)
lstm_stride = st.sidebar.slider("LSTM stride", 1, 32, 8, 1)
lstm_epochs = st.sidebar.slider("LSTM epoch sayısı", 5, 60, 20, 5)

st.sidebar.subheader("CNN Ayarları")
cnn_window_size = 16
cnn_stride = 1
cnn_epochs = 70
st.sidebar.caption(
    "CNN hiperparametre profili sabit: pencere=16, stride=1, max epoch=70. "
    "En iyi epoch eğitim sırasında EarlyStopping ile otomatik belirlenir."
)


if is_uploaded_mode:
    summary_df = df
else:
    summary_df = simay_generated_df if simay_generated_df is not None else df

if is_uploaded_mode:
    col1, col2, col3 = st.columns(3)
    col1.metric("Satir Sayisi", len(summary_df))
    col2.metric("Sutun Sayisi", len(summary_df.columns))
    col3.metric("Yuklenen Dosyada Etiket Var mi?", "Evet" if uploaded_has_label_raw else "Hayir")
else:
    col1, col2 = st.columns(2)
    col1.metric("Satir Sayisi", len(summary_df))
    col2.metric("Sutun Sayisi", len(summary_df.columns))

if gizem_generated_df is not None:
    source_text = f"Gizem çıktısı (Simay -> Aleyna -> Gizem) | Kaynak: {source_label(data_source)}"
else:
    source_text = str(data_source)
st.caption(f"Kullanılan veri: {source_text}")
if pipeline_output_paths is not None:
    st.caption(
        "Pipeline çıktıları kaydedildi: "
        f"{pipeline_output_paths[0].name}, {pipeline_output_paths[1].name}, "
        f"{pipeline_output_paths[2].name}, {pipeline_output_paths[3].name}"
    )

if is_uploaded_mode:
    overview_tab, aleyna_tab, gizem_tab, lstm_tab, cnn_tab, compare_tab, live_tab = st.tabs(
        ["Veri İnceleme", "Sinyal İşleme Filtreleme", "Özellik Çıkarımı (Feature Engineering)", "LSTM Eğitimi", "CNN Eğitimi", "Adil Karşılaştırma", "Canlı Tahmin"]
    )
else:
    overview_tab, simay_tab, aleyna_tab, gizem_tab, lstm_tab, cnn_tab, compare_tab, live_tab = st.tabs(
        ["Veri İnceleme", "Fiziksel Modelleme ve Donanım", "Sinyal İşleme Filtreleme", "Özellik Çıkarımı (Feature Engineering)", "LSTM Eğitimi", "CNN Eğitimi", "Adil Karşılaştırma", "Canlı Tahmin"]
    )


with overview_tab:
    st.subheader("Makine Öğrenimi Veri Görüntüleme")
    preview_rows = st.slider("Gösterilecek satır sayısı", 5, min(len(model_training_df), 100), 20)
    max_start = max(len(model_training_df) - preview_rows, 0)
    start_row = st.slider("Başlangıç satırı", 0, max_start, 0)
    end_row = min(start_row + preview_rows, len(model_training_df))
    st.caption(f"Gösterilen aralık: {start_row} - {end_row - 1} / Toplam satır: {len(model_training_df)}")
    st.dataframe(model_training_df.iloc[start_row:end_row], use_container_width=True)

    st.subheader("Sinyal Grafigi")
    overview_graph_df = df if is_uploaded_mode else (simay_generated_df if simay_generated_df is not None else df)
    overview_feature_candidates = ["delta_lambda_noisy", "delta_lambda_simulink", feature_column]
    overview_feature = next((column for column in overview_feature_candidates if column in overview_graph_df.columns), None)
    if overview_feature is not None:
        st.pyplot(plot_signal(overview_graph_df, overview_feature))
        st.caption(f"Grafikte gosterilen kaynak sutun: {overview_feature}")
    else:
        st.info("Grafik icin uygun sinyal sutunu bulunamadi.")

    if show_label_distribution:
        st.subheader("Etiket Dağılımı")
        label_counts = model_training_df["label"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(label_counts.index, label_counts.values)
        ax.set_xlabel("Etiket")
        ax.set_ylabel("Adet")
        ax.set_title("Sınıf Dağılımı")
        ax.grid(axis="y")
        st.pyplot(fig)
        st.dataframe(label_counts.rename("adet"), use_container_width=True)
    else:
        st.info("Etiket oluşturulamadıysa eğitim yapılamaz; bu durumda yalnızca kayıtlı modelle tahmin yapılabilir.")

with lstm_tab:
    st.subheader("LSTM Model Eğitimi")
    st.write(
        "LSTM, zaman serisindeki ardışık ölçümler arasındaki ilişkiyi öğrenir. "
        "Bu bölüm mevcut panelindeki RNN/LSTM tabanlı hasar sınıflandırma akışıdır."
    )

    lstm_train_disabled = not has_label or len(model_training_df) <= lstm_window_size
    if lstm_train_disabled:
        st.warning("LSTM eğitimi için CSV içinde label sütunu olmalı ve veri pencere boyutundan uzun olmalı.")

    lstm_action_col1, lstm_action_col2 = st.columns(2)
    with lstm_action_col1:
        retrain_lstm_clicked = st.button("LSTM Modelini Eğit ve Kaydet", disabled=lstm_train_disabled)
    with lstm_action_col2:
        use_saved_lstm_clicked = st.button("Kayıtlı LSTM Modelini Yükle", disabled=not artifacts_exist(LSTM_ARTIFACTS))

    if retrain_lstm_clicked:
        validate_columns(model_training_df, feature_column, needs_label=True)

        best_selection = None
        recipe_results = []

        with st.spinner("LSTM recipe search çalışıyor, en iyi model seçiliyor..."):
            for recipe in lstm_recipes():
                tf.keras.backend.clear_session()
                random.seed(SEED)
                np.random.seed(SEED)
                tf.random.set_seed(SEED)

                try:
                    x_windows, y_text = make_windows(
                        model_training_df,
                        feature_column,
                        "label",
                        recipe["window_size"],
                        stride=recipe["stride"],
                    )
                except ValueError:
                    continue

                encoder = LabelEncoder()
                y_numeric = encoder.fit_transform(y_text)

                x_train_val, x_test, y_train_val, y_test = train_test_split(
                    x_windows,
                    y_numeric,
                    test_size=0.2,
                    random_state=SEED,
                    stratify=y_numeric,
                )
                x_train, x_val, y_train, y_val = train_test_split(
                    x_train_val,
                    y_train_val,
                    test_size=0.2,
                    random_state=SEED,
                    stratify=y_train_val,
                )

                scaler = StandardScaler()
                feature_count = x_train.shape[2]
                x_train_scaled = scaler.fit_transform(x_train.reshape(-1, feature_count)).reshape(x_train.shape)
                x_val_scaled = scaler.transform(x_val.reshape(-1, feature_count)).reshape(x_val.shape)
                x_test_scaled = scaler.transform(x_test.reshape(-1, feature_count)).reshape(x_test.shape)

                model = build_lstm_model(
                    window_size=x_train_scaled.shape[1],
                    feature_count=x_train_scaled.shape[2],
                    class_count=len(encoder.classes_),
                    recipe=recipe,
                )

                checkpoint_path = MODEL_DIR / f"{recipe['name']}_best_lstm.keras"
                callbacks = [
                    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
                    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
                    ModelCheckpoint(
                        str(checkpoint_path),
                        monitor="val_loss",
                        save_best_only=True,
                        mode="min",
                    ),
                ]

                history = model.fit(
                    x_train_scaled,
                    y_train,
                    validation_data=(x_val_scaled, y_val),
                    epochs=recipe["epochs"],
                    batch_size=32,
                    shuffle=False,
                    callbacks=callbacks,
                    verbose=0,
                )

                best_model_for_recipe = load_model(checkpoint_path)
                val_pred = np.argmax(best_model_for_recipe.predict(x_val_scaled, verbose=0), axis=1)
                val_macro_f1 = float(f1_score(y_val, val_pred, average="macro"))
                val_loss_min = float(np.min(history.history.get("val_loss", [9999])))

                recipe_results.append(
                    {
                        "name": recipe["name"],
                        "window_size": recipe["window_size"],
                        "stride": recipe["stride"],
                        "epochs": recipe["epochs"],
                        "val_macro_f1": val_macro_f1,
                        "val_loss_min": val_loss_min,
                        "best_epoch": compute_best_epoch(history),
                    }
                )

                rank_key = (-val_loss_min, val_macro_f1)
                if best_selection is None or rank_key > best_selection["rank_key"]:
                    test_loss, test_accuracy = best_model_for_recipe.evaluate(x_test_scaled, y_test, verbose=0)
                    test_probabilities = best_model_for_recipe.predict(x_test_scaled, verbose=0)
                    y_pred = np.argmax(test_probabilities, axis=1)
                    cm = confusion_matrix(y_test, y_pred)
                    report_dict = classification_report(
                        y_test,
                        y_pred,
                        target_names=encoder.classes_,
                        output_dict=True,
                        zero_division=0,
                    )
                    best_selection = {
                        "rank_key": rank_key,
                        "model": best_model_for_recipe,
                        "history": history,
                        "scaler": scaler,
                        "encoder": encoder,
                        "recipe": recipe,
                        "window_count": len(x_windows),
                        "test_loss": test_loss,
                        "test_accuracy": test_accuracy,
                        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
                        "cm": cm,
                        "y_test": y_test,
                        "y_pred": y_pred,
                        "feature_count": int(x_train_scaled.shape[2]),
                    }

        if best_selection is None:
            st.error("LSTM recipe search için uygun pencere üretilemedi. Veri uzunluğunu artırmayı deneyin.")
        else:
            best_epoch = compute_best_epoch(best_selection["history"])
            config = {
                "model_type": "LSTM",
                "feature_column": feature_column,
                "label_column": "label",
                "window_size": int(best_selection["recipe"]["window_size"]),
                "stride": int(best_selection["recipe"]["stride"]),
                "feature_count": int(best_selection["feature_count"]),
                "class_names": best_selection["encoder"].classes_.tolist(),
                "source": source_label(data_source),
                "scaling": "StandardScaler",
                "seed": SEED,
                "recipe_version": LSTM_RECIPE_VERSION,
                "best_recipe": best_selection["recipe"]["name"],
                "recipe_results": recipe_results,
            }

            save_artifacts(
                LSTM_ARTIFACTS,
                best_selection["model"],
                best_selection["scaler"],
                best_selection["encoder"],
                config,
            )
            store_active_model(
                "LSTM",
                best_selection["model"],
                best_selection["scaler"],
                best_selection["encoder"],
                config,
            )
            st.dataframe(pd.DataFrame(recipe_results), use_container_width=True)
            st.info(f"Seçilen en iyi LSTM recipe: {best_selection['recipe']['name']}")
            show_results(
                "LSTM",
                best_selection["test_accuracy"],
                best_selection["test_loss"],
                best_selection["window_count"],
                best_selection["history"],
                best_selection["cm"],
                best_selection["encoder"],
                best_selection["y_test"],
                best_selection["y_pred"],
            )

    if use_saved_lstm_clicked:
        model, scaler, encoder, config = load_artifacts(LSTM_ARTIFACTS)
        store_active_model("LSTM", model, scaler, encoder, config)
        st.success("Kayıtlı LSTM modeli aktif edildi.")
        if config.get("recipe_version") != LSTM_RECIPE_VERSION:
            st.warning("Kayıtlı LSTM modeli eski ayarlarla eğitilmiş olabilir. En iyi sonuç için yeniden eğitim önerilir.")

    if not artifacts_exist(LSTM_ARTIFACTS):
        st.info("Henüz kayıtlı LSTM modeli bulunamadı. Önce LSTM modelini eğitip kaydet.")

with cnn_tab:
    st.subheader("1D-CNN Model Eğitimi")
    st.write(
        "CNN eğitim akışı sabit seed, sabit split ve en iyi model checkpoint'i ile daha stabil çalışacak şekilde ayarlandı."
    )

    cnn_train_disabled = not has_label or len(model_training_df) <= cnn_window_size
    if cnn_train_disabled:
        st.warning("CNN eğitimi için CSV içinde label sütunu olmalı ve veri pencere boyutundan uzun olmalı.")

    has_saved_cnn = artifacts_exist(CNN_ARTIFACTS)
    if has_saved_cnn and st.session_state.get("active_model_type") != "1D-CNN":
        model, scaler, encoder, config = load_artifacts(CNN_ARTIFACTS)
        store_active_model("1D-CNN", model, scaler, encoder, config)
        st.info("Varsayılan olarak kayıtlı CNN modeli aktif edildi.")
        if config.get("recipe_version") != CNN_RECIPE_VERSION:
            st.warning("Kayıtlı CNN modeli eski eğitim ayarlarıyla üretilmiş olabilir. En iyi sonuç için yeniden eğit.")

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button("Kayıtlı CNN Modelini Kullan", disabled=not has_saved_cnn):
            model, scaler, encoder, config = load_artifacts(CNN_ARTIFACTS)
            store_active_model("1D-CNN", model, scaler, encoder, config)
            st.success("Kayıtlı CNN modeli aktif edildi.")
            if config.get("recipe_version") != CNN_RECIPE_VERSION:
                st.warning("Bu model eski ayarlarla eğitilmiş olabilir. Confusion matrix için yeniden eğitim önerilir.")

    with action_col2:
        retrain_clicked = st.button("CNN Modelini Yeniden Eğit ve Kaydet", disabled=cnn_train_disabled)

    if not has_saved_cnn:
        st.info("Henüz kayıtlı CNN modeli bulunamadı. İlk eğitimde en iyi model checkpoint ile kaydedilecek.")

    if retrain_clicked:
        validate_columns(model_training_df, feature_column, needs_label=True)
        x_windows, y_text = make_windows(model_training_df, feature_column, "label", cnn_window_size, stride=cnn_stride)
        x_scaled, scaler = scale_cnn_full_windows(x_windows)

        encoder = LabelEncoder()
        y_numeric = encoder.fit_transform(y_text)

        x_train_val, x_test, y_train_val, y_test = train_test_split(
            x_scaled,
            y_numeric,
            test_size=0.2,
            random_state=SEED,
            stratify=y_numeric,
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val,
            y_train_val,
            test_size=0.2,
            random_state=SEED,
            stratify=y_train_val,
        )

        class_values = np.unique(y_train)
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=class_values,
            y=y_train,
        )
        class_weight_dict = dict(zip(class_values, class_weights))

        MODEL_DIR.mkdir(exist_ok=True)
        recipe_results = []
        best_selection = None

        with st.spinner("CNN recipe search çalışıyor, en iyi model seçiliyor..."):
            for recipe in cnn_recipes():
                tf.keras.backend.clear_session()
                random.seed(SEED)
                np.random.seed(SEED)
                tf.random.set_seed(SEED)

                model = build_cnn_model(
                    window_size=x_scaled.shape[1],
                    feature_count=x_scaled.shape[2],
                    class_count=len(encoder.classes_),
                    recipe=recipe,
                )

                checkpoint_path = MODEL_DIR / f"{recipe['name']}_best.keras"
                callbacks = [
                    EarlyStopping(
                        monitor="val_loss",
                        patience=6,
                        min_delta=1e-4,
                        restore_best_weights=True,
                    ),
                    ReduceLROnPlateau(
                        monitor="val_loss",
                        factor=0.4,
                        patience=3,
                        min_lr=1e-6,
                    ),
                    ModelCheckpoint(
                        str(checkpoint_path),
                        monitor="val_loss",
                        save_best_only=True,
                        mode="min",
                    ),
                ]

                history = model.fit(
                    x_train,
                    y_train,
                    validation_data=(x_val, y_val),
                    epochs=cnn_epochs,
                    batch_size=16,
                    callbacks=callbacks,
                    class_weight=class_weight_dict if recipe["use_class_weight"] else None,
                    shuffle=False,
                    verbose=0,
                )

                best_model_for_recipe = load_model(checkpoint_path)
                val_pred = np.argmax(best_model_for_recipe.predict(x_val, verbose=0), axis=1)
                val_macro_f1 = float(f1_score(y_val, val_pred, average="macro"))
                val_loss_min = float(np.min(history.history.get("val_loss", [9999])))

                recipe_results.append(
                    {
                        "name": recipe["name"],
                        "val_macro_f1": val_macro_f1,
                        "val_loss_min": val_loss_min,
                        "best_epoch": compute_best_epoch(history),
                    }
                )

                rank_key = (-val_loss_min, val_macro_f1)
                if best_selection is None or rank_key > best_selection["rank_key"]:
                    best_selection = {
                        "rank_key": rank_key,
                        "recipe": recipe,
                        "history": history,
                        "checkpoint_path": checkpoint_path,
                    }

        best_model = load_model(best_selection["checkpoint_path"])
        best_epoch = compute_best_epoch(best_selection["history"])

        test_loss, test_accuracy = best_model.evaluate(x_test, y_test, verbose=0)
        probabilities = best_model.predict(x_test, verbose=0)
        y_pred = np.argmax(probabilities, axis=1)
        cm = confusion_matrix(y_test, y_pred)

        report_dict = classification_report(
            y_test,
            y_pred,
            target_names=encoder.classes_,
            output_dict=True,
            zero_division=0,
        )
        macro_f1 = float(report_dict["macro avg"]["f1-score"])

        config = {
            "model_type": "1D-CNN",
            "feature_column": feature_column,
            "label_column": "label",
            "window_size": int(cnn_window_size),
            "stride": int(cnn_stride),
            "feature_count": int(x_scaled.shape[2]),
            "class_names": encoder.classes_.tolist(),
            "source": source_label(data_source),
            "scaling": "MinMaxScaler",
            "seed": SEED,
            "recipe_version": CNN_RECIPE_VERSION,
            "best_recipe": best_selection["recipe"]["name"],
            "recipe_results": recipe_results,
        }

        save_artifacts(CNN_ARTIFACTS, best_model, scaler, encoder, config, save_model=False)
        store_active_model("1D-CNN", best_model, scaler, encoder, config)
        st.dataframe(pd.DataFrame(recipe_results), use_container_width=True)
        st.info(f"Seçilen en iyi recipe: {best_selection['recipe']['name']}")
        show_cnn_results(
            test_accuracy=test_accuracy,
            test_loss=test_loss,
            window_count=len(x_windows),
            best_epoch=best_epoch,
            macro_f1=macro_f1,
            history=best_selection["history"],
            cm=cm,
            encoder=encoder,
            y_test=y_test,
            y_pred=y_pred,
        )

with compare_tab:
    st.subheader("LSTM vs CNN (Aynı Koşullarda)")
    st.write(
        "Bu bölümde iki model aynı veri, aynı pencere, aynı stride, aynı split ve aynı seed ile eğitilip karşılaştırılır."
    )

    fair_window_size = 16
    fair_stride = 1
    fair_epochs = 40
    st.caption(
        f"Sabit karşılaştırma ayarları: window={fair_window_size}, stride={fair_stride}, epoch={fair_epochs}, seed={SEED}"
    )

    fair_train_disabled = not has_label or len(model_training_df) <= fair_window_size
    if fair_train_disabled:
        st.warning("Karşılaştırma için CSV içinde label sütunu olmalı ve veri pencere boyutundan uzun olmalı.")

    if st.button("Adil Karşılaştırmayı Çalıştır", disabled=fair_train_disabled):
        validate_columns(model_training_df, feature_column, needs_label=True)
        x_windows, y_text = make_windows(model_training_df, feature_column, "label", fair_window_size, stride=fair_stride)

        encoder = LabelEncoder()
        y_numeric = encoder.fit_transform(y_text)

        x_train_val, x_test, y_train_val, y_test = train_test_split(
            x_windows,
            y_numeric,
            test_size=0.2,
            random_state=SEED,
            stratify=y_numeric,
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val,
            y_train_val,
            test_size=0.2,
            random_state=SEED,
            stratify=y_train_val,
        )

        class_values = np.unique(y_train)
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=class_values,
            y=y_train,
        )
        class_weight_dict = dict(zip(class_values, class_weights))

        with st.spinner("LSTM ve CNN aynı koşullarda eğitiliyor..."):
            tf.keras.backend.clear_session()
            random.seed(SEED)
            np.random.seed(SEED)
            tf.random.set_seed(SEED)

            feature_count = x_train.shape[2]

            lstm_scaler = StandardScaler()
            x_train_lstm = lstm_scaler.fit_transform(x_train.reshape(-1, feature_count)).reshape(x_train.shape)
            x_val_lstm = lstm_scaler.transform(x_val.reshape(-1, feature_count)).reshape(x_val.shape)
            x_test_lstm = lstm_scaler.transform(x_test.reshape(-1, feature_count)).reshape(x_test.shape)

            lstm_recipe = {
                "lstm_units": 64,
                "dense_units": 32,
                "dropout": 0.25,
                "learning_rate": 0.0010,
            }
            lstm_model = build_lstm_model(
                window_size=x_train_lstm.shape[1],
                feature_count=x_train_lstm.shape[2],
                class_count=len(encoder.classes_),
                recipe=lstm_recipe,
            )
            lstm_callbacks = [
                EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
            ]
            lstm_history = lstm_model.fit(
                x_train_lstm,
                y_train,
                validation_data=(x_val_lstm, y_val),
                epochs=fair_epochs,
                batch_size=32,
                callbacks=lstm_callbacks,
                class_weight=class_weight_dict,
                shuffle=False,
                verbose=0,
            )
            lstm_test_loss, lstm_test_acc = lstm_model.evaluate(x_test_lstm, y_test, verbose=0)
            lstm_pred = np.argmax(lstm_model.predict(x_test_lstm, verbose=0), axis=1)
            lstm_macro_f1 = float(f1_score(y_test, lstm_pred, average="macro"))
            lstm_cm = confusion_matrix(y_test, lstm_pred)

            tf.keras.backend.clear_session()
            random.seed(SEED)
            np.random.seed(SEED)
            tf.random.set_seed(SEED)

            cnn_scaler = MinMaxScaler()
            x_train_cnn = cnn_scaler.fit_transform(x_train.reshape(-1, feature_count)).reshape(x_train.shape)
            x_val_cnn = cnn_scaler.transform(x_val.reshape(-1, feature_count)).reshape(x_val.shape)
            x_test_cnn = cnn_scaler.transform(x_test.reshape(-1, feature_count)).reshape(x_test.shape)

            cnn_recipe = cnn_recipes()[0]
            cnn_model = build_cnn_model(
                window_size=x_train_cnn.shape[1],
                feature_count=x_train_cnn.shape[2],
                class_count=len(encoder.classes_),
                recipe=cnn_recipe,
            )
            cnn_callbacks = [
                EarlyStopping(monitor="val_loss", patience=6, min_delta=1e-4, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=3, min_lr=1e-6),
            ]
            cnn_history = cnn_model.fit(
                x_train_cnn,
                y_train,
                validation_data=(x_val_cnn, y_val),
                epochs=fair_epochs,
                batch_size=16,
                callbacks=cnn_callbacks,
                class_weight=class_weight_dict if cnn_recipe["use_class_weight"] else None,
                shuffle=False,
                verbose=0,
            )
            cnn_test_loss, cnn_test_acc = cnn_model.evaluate(x_test_cnn, y_test, verbose=0)
            cnn_pred = np.argmax(cnn_model.predict(x_test_cnn, verbose=0), axis=1)
            cnn_macro_f1 = float(f1_score(y_test, cnn_pred, average="macro"))
            cnn_cm = confusion_matrix(y_test, cnn_pred)

        metrics_df = pd.DataFrame(
            [
                {
                    "Model": "LSTM",
                    "Test Accuracy": round(float(lstm_test_acc), 4),
                    "Test Loss": round(float(lstm_test_loss), 4),
                    "Macro F1": round(float(lstm_macro_f1), 4),
                },
                {
                    "Model": "CNN",
                    "Test Accuracy": round(float(cnn_test_acc), 4),
                    "Test Loss": round(float(cnn_test_loss), 4),
                    "Macro F1": round(float(cnn_macro_f1), 4),
                },
            ]
        )
        st.dataframe(metrics_df, use_container_width=True)

        winner = "LSTM" if lstm_macro_f1 >= cnn_macro_f1 else "CNN"
        st.info(f"Adil karşılaştırma sonucu (Macro F1): {winner} önde.")

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.pyplot(plot_history(lstm_history, "LSTM"))
        with chart_col2:
            st.pyplot(plot_cnn_history(cnn_history, compute_best_epoch(cnn_history)))

        cm_col1, cm_col2 = st.columns(2)
        with cm_col1:
            st.pyplot(plot_confusion_matrix(lstm_cm, encoder.classes_, "LSTM Confusion Matrix"))
        with cm_col2:
            st.pyplot(plot_confusion_matrix(cnn_cm, encoder.classes_, "CNN Confusion Matrix"))

if not is_uploaded_mode:
    with simay_tab:
        if is_uploaded_mode:
            st.info("Disaridan etiketli veri geldi")
        else:
            st.subheader("Fiziksel Modelleme & Donanım")
            st.write("FBG sensör yapısı, Bragg kayması ve fiziksel simülasyon parametreleri.")

            try:
                simay_df = simay_generated_df if simay_generated_df is not None else run_simay_pipeline(df, force_relabel=False)

                s_col1, s_col2, s_col3 = st.columns(3)
                s_col1.metric("Kayit Sayisi", int(len(simay_df)))
                s_col2.metric("Sutun Sayisi", int(len(simay_df.columns)))
                s_col3.metric("Sinif Sayisi", int(simay_df["label"].nunique()))

                simay_rows = st.slider("Gosterilecek satir sayisi (Simay)", 5, min(len(simay_df), 100), 20)
                simay_max_start = max(len(simay_df) - simay_rows, 0)
                simay_start = st.slider("Baslangic satiri (Simay)", 0, simay_max_start, 0)
                simay_end = min(simay_start + simay_rows, len(simay_df))
                st.caption(f"Simay araligi: {simay_start} - {simay_end - 1} / Toplam satir: {len(simay_df)}")
                st.dataframe(simay_df.iloc[simay_start:simay_end], use_container_width=True)

                fig1, ax1 = plt.subplots(figsize=(12, 4))
                if "delta_lambda_simulink" in simay_df.columns:
                    ax1.plot(simay_df["time"], simay_df["delta_lambda_simulink"], label="Simulink Signal", linewidth=2)
                ax1.plot(simay_df["time"], simay_df["delta_lambda_noisy"], label="Noisy Signal", alpha=0.65)
                ax1.set_title("Simay - Etiketlenen Ham Sinyal")
                ax1.set_xlabel("Time")
                ax1.set_ylabel("Delta Lambda")
                ax1.grid(True)
                ax1.legend()
                st.pyplot(fig1)

                fig2, ax2 = plt.subplots(figsize=(6, 4))
                label_counts = simay_df["label"].value_counts()
                ax2.bar(label_counts.index, label_counts.values)
                ax2.set_title("Simay - Etiket Dagilimi")
                ax2.set_xlabel("Label")
                ax2.set_ylabel("Adet")
                ax2.grid(axis="y")
                st.pyplot(fig2)
                st.dataframe(label_counts.rename("adet"), use_container_width=True)
            except Exception as exc:
                st.error(f"Simay adimi calistirilamadi: {exc}")

with aleyna_tab:
    st.subheader("Sinyal İşleme & Filtreleme")
    st.write("Gürültü temizleme (Denoising) ve optik sinyali AI için hazır hale getirme.")

    try:
        if simay_generated_df is None:
            raise ValueError("Simay adımı üretilemediği için Aleyna adımı başlatılamadı.")
        aleyna_df = aleyna_generated_df if aleyna_generated_df is not None else run_aleyna_pipeline(simay_generated_df, window_size=15)
        st.success("Aleyna adımı tamamlandı.")
        st.dataframe(aleyna_df.head(20), use_container_width=True)

        fig, ax = plt.subplots(figsize=(12, 4))
        x_axis = aleyna_df["time"] if "time" in aleyna_df.columns else aleyna_df.index
        ax.plot(x_axis, aleyna_df["delta_lambda_noisy"], label="Noisy Signal", alpha=0.45)
        ax.plot(x_axis, aleyna_df["delta_lambda_filtered"], label="Filtered Signal", linewidth=2)
        ax.set_title("Aleyna - FBG Signal Filtering")
        ax.set_xlabel("Time")
        ax.set_ylabel("Delta Lambda")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
    except Exception as exc:
        st.error(f"Aleyna adımı çalıştırılamadı: {exc}")

with gizem_tab:
    st.subheader("Özellik Çıkarımı (Feature Engineering)")
    st.write("Veriyi sayısallaştırma, tepe noktası analizi ve veri setini etiketleme.")

    try:
        if aleyna_generated_df is None:
            raise ValueError("Aleyna adımı üretilemediği için Gizem adımı başlatılamadı.")
        gizem_labeled_df = gizem_generated_df if gizem_generated_df is not None else run_gizem_pipeline(aleyna_generated_df)[0]
        gizem_feature_table = gizem_features_df if gizem_features_df is not None else run_gizem_pipeline(aleyna_generated_df)[1]

        st.success("Gizem adımı tamamlandı.")
        st.write("Global özellik tablosu:")
        st.dataframe(gizem_feature_table, use_container_width=True)

        st.write("Gizem etiketlenmiş veri önizlemesi:")
        st.dataframe(gizem_labeled_df.head(20), use_container_width=True)
    except Exception as exc:
        st.error(f"Gizem adımı çalıştırılamadı: {exc}")

with live_tab:
    st.subheader("Canlı Tahmin Paneli")

    load_col1, load_col2 = st.columns(2)
    with load_col1:
        if st.button("LSTM'i Tahmin İçin Aktif Et", disabled=not artifacts_exist(LSTM_ARTIFACTS)):
            model, scaler, encoder, config = load_artifacts(LSTM_ARTIFACTS)
            store_active_model("LSTM", model, scaler, encoder, config)
            st.success("LSTM modeli aktif edildi.")

    with load_col2:
        if st.button("CNN'i Tahmin İçin Aktif Et", disabled=not artifacts_exist(CNN_ARTIFACTS)):
            model, scaler, encoder, config = load_artifacts(CNN_ARTIFACTS)
            store_active_model("1D-CNN", model, scaler, encoder, config)
            st.success("CNN modeli aktif edildi.")

    if not active_model_ready():
        st.warning("Tahmin için önce LSTM veya CNN modelini eğit ya da kayıtlı modeli yükle.")
    else:
        model_type = st.session_state["active_model_type"]
        config = st.session_state["active_config"]
        active_feature = config["feature_column"]
        active_window_size = int(config["window_size"])

        st.write("Aktif model:")
        st.json(config)

        if active_feature not in model_training_df.columns:
            st.error(f"Aktif model {active_feature} sütununu bekliyor, ancak bu CSV içinde yok.")
        else:
            st.write(
                f"{model_type} modeli son {active_window_size} ölçümü kullanarak tahmin yapacak. "
                f"Kullanılan sütun: {active_feature}"
            )

            prediction_mode = st.radio(
                "Tahmin veri kaynağı",
                ["CSV icindeki son olcumler", "Tahmin icin yeni CSV yukle", "Manuel deger gir"],
                horizontal=True,
            )

            if prediction_mode == "CSV içindeki son ölçümler":
                if len(model_training_df) < active_window_size:
                    st.error("CSV, modelin beklediği pencere boyutundan kısa.")
                else:
                    latest_values = model_training_df[active_feature].tail(active_window_size).values
                    st.line_chart(pd.DataFrame({active_feature: latest_values}))

                    if st.button("Son Ölçümlerle Tahmin Et"):
                        predicted_label, confidence, probabilities = predict_window(latest_values)
                        st.success(f"Tahmin: {predicted_label}")
                        st.metric("Güven", round(confidence, 4))
                        probability_df = pd.DataFrame(
                            {
                                "sınıf": st.session_state["active_encoder"].classes_,
                                "olasılık": probabilities,
                            }
                        )
                        st.dataframe(probability_df, use_container_width=True)

            elif prediction_mode == "Tahmin icin yeni CSV yukle":
                prediction_file = st.file_uploader("Tahmin icin CSV yukle", type=["csv"], key="live_prediction_csv")
                if prediction_file is not None:
                    try:
                        prediction_df = pd.read_csv(prediction_file)
                    except Exception as exc:
                        st.error(f"Yuklenen tahmin CSV dosyasi okunamadi: {exc}")
                        prediction_df = None

                    if prediction_df is not None:
                        if active_feature not in prediction_df.columns:
                            st.error(
                                f"Yuklenen tahmin verisinde `{active_feature}` sutunu yok. "
                                "Model bu sutunu bekliyor."
                            )
                        else:
                            prediction_signal = pd.to_numeric(prediction_df[active_feature], errors="coerce").dropna()
                            if len(prediction_signal) < active_window_size:
                                st.error(
                                    f"Yuklenen tahmin verisi en az {active_window_size} deger icermeli. "
                                    f"Bulunan gecerli deger sayisi: {len(prediction_signal)}"
                                )
                            else:
                                latest_values = prediction_signal.tail(active_window_size).values
                                st.line_chart(pd.DataFrame({active_feature: latest_values}))
                                if st.button("Yuklenen Veriyle Tahmin Et"):
                                    predicted_label, confidence, probabilities = predict_window(latest_values)
                                    st.success(f"Tahmin: {predicted_label}")
                                    st.metric("Guven", round(confidence, 4))
                                    probability_df = pd.DataFrame(
                                        {
                                            "sinif": st.session_state["active_encoder"].classes_,
                                            "olasilik": probabilities,
                                        }
                                    )
                                    st.dataframe(probability_df, use_container_width=True)

            else:
                st.write(
                    f"{active_window_size} adet değeri virgülle ayırarak gir. "
                    "Örnek: 6.2, 6.3, 6.4"
                )
                manual_text = st.text_area("Sinyal değerleri")

                if st.button("Manuel Değerlerle Tahmin Et"):
                    try:
                        manual_values = [
                            float(value.strip())
                            for value in manual_text.replace("\n", ",").split(",")
                            if value.strip()
                        ]

                        if len(manual_values) != active_window_size:
                            st.error(
                                f"Model {active_window_size} değer bekliyor, "
                                f"sen {len(manual_values)} değer girdin."
                            )
                        else:
                            predicted_label, confidence, probabilities = predict_window(manual_values)
                            st.success(f"Tahmin: {predicted_label}")
                            st.metric("Güven", round(confidence, 4))
                            probability_df = pd.DataFrame(
                                {
                                    "sınıf": st.session_state["active_encoder"].classes_,
                                    "olasılık": probabilities,
                                }
                            )
                            st.dataframe(probability_df, use_container_width=True)

                    except ValueError:
                        st.error("Lütfen sadece sayısal değerler gir.")
