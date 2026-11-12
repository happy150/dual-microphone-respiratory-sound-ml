from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import librosa
import matplotlib
import numpy as np
import soundfile as sf
import streamlit as st
import tensorflow as tf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_AUDIO_DIR = PROJECT_ROOT / "data" / "raw"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

TARGET_SR = 4000
CLIP_SECONDS = 5.0
TARGET_SAMPLES = int(TARGET_SR * CLIP_SECONDS)
N_FFT = 512
HOP_LENGTH = 128
N_MELS = 64
FMIN = 20
FMAX = 2000
TARGET_FRAMES = 157

CRACKLE_THRESHOLD = 0.45
WHEEZE_THRESHOLD = 0.60

MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "weighted_3ch_cnn": {
        "label": "Weighted 3-channel CNN",
        "path": NOTEBOOKS_DIR / "best_3ch_multilabel.keras",
    },
    "patient_balanced_3ch_cnn": {
        "label": "Patient-balanced 3-channel CNN",
        "path": NOTEBOOKS_DIR / "best_3ch_patient_balanced.keras",
    },
    "original_cnn_lstm_m0": {
        "label": "Original CNN-LSTM (M0)",
        "path": NOTEBOOKS_DIR / "best_cnn_lstm_patient_weighted.keras",
    },
    "device_aligned_cnn_lstm_m2": {
        "label": "Device-aligned CNN-LSTM (M2)",
        "path": NOTEBOOKS_DIR / "best_cnn_lstm_device_aligned.keras",
    },
}

MODEL_KEYS = list(MODEL_CATALOG.keys())
DEFAULT_MODEL_KEY = "original_cnn_lstm_m0"

HISTORICAL_METRICS = {
    "Validation accuracy": 56.55,
    "Validation macro F1": 50.41,
    "Test accuracy": 45.83,
    "Test macro F1": 35.84,
    "Crackle test ROC-AUC": 0.5885,
    "Wheeze test ROC-AUC": 0.7276,
    "Device-alignment crackle validation AUC": 0.5555,
    "Device-alignment wheeze validation AUC": 0.9013,
}


@st.cache_resource
def load_model_registry() -> Dict[str, tf.keras.Model]:
    registry: Dict[str, tf.keras.Model] = {}
    for key, meta in MODEL_CATALOG.items():
        path = meta["path"]
        if not path.exists():
            continue
        registry[key] = tf.keras.models.load_model(str(path), compile=False)
    return registry


@st.cache_resource
def load_model_once(model_key: str = DEFAULT_MODEL_KEY) -> tf.keras.Model:
    registry = load_model_registry()
    model = registry.get(model_key)
    if model is None:
        raise ValueError(f"Model {model_key} is not available in this project.")
    return model


def fit_audio_length(y: np.ndarray, target_samples: int) -> np.ndarray:
    if len(y) > target_samples:
        start = (len(y) - target_samples) // 2
        y = y[start : start + target_samples]
    elif len(y) < target_samples:
        total_padding = target_samples - len(y)
        left_padding = total_padding // 2
        right_padding = total_padding - left_padding
        y = np.pad(y, (left_padding, right_padding), mode="constant")
    return y.astype(np.float32)


def fit_logmel_frames(log_mel: np.ndarray, target_frames: int = TARGET_FRAMES) -> np.ndarray:
    source_frames = log_mel.shape[1]
    if source_frames > target_frames:
        start = (source_frames - target_frames) // 2
        log_mel = log_mel[:, start : start + target_frames]
    elif source_frames < target_frames:
        total_padding = target_frames - source_frames
        left_padding = total_padding // 2
        right_padding = total_padding - left_padding
        log_mel = np.pad(
            log_mel,
            ((0, 0), (left_padding, right_padding)),
            mode="constant",
            constant_values=0.0,
        )
    return log_mel.astype(np.float32)


def extract_logmel(segment: np.ndarray) -> np.ndarray:
    mel_power = librosa.feature.melspectrogram(
        y=segment,
        sr=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel_power, ref=np.max)
    log_mel = np.clip(log_mel, -80.0, 0.0)
    log_mel = (log_mel + 80.0) / 80.0
    return log_mel.astype(np.float32)


def build_three_channel_input(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    segment = fit_audio_length(audio, TARGET_SAMPLES)
    log_mel = extract_logmel(segment)
    log_mel = fit_logmel_frames(log_mel, TARGET_FRAMES)

    width = min(9, log_mel.shape[1] if log_mel.shape[1] % 2 == 1 else log_mel.shape[1] - 1)
    if width < 3:
        raise ValueError(f"Too few valid frames for delta extraction: {log_mel.shape[1]}")

    delta = librosa.feature.delta(log_mel, width=width, order=1, axis=-1)
    delta_delta = librosa.feature.delta(log_mel, width=width, order=2, axis=-1)

    x = np.zeros((1, 64, TARGET_FRAMES, 3), dtype=np.float32)
    x[0, :, :, 0] = log_mel
    x[0, :, :, 1] = delta
    x[0, :, :, 2] = delta_delta
    return x


def normalize_model_output(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    if probs.shape[-1] == 1:
        probs = np.concatenate([1.0 - probs, probs], axis=-1)
    probs = probs[:, :2]
    return probs.astype(np.float32)


def decode_prediction(probabilities: np.ndarray, crackle_threshold: float = CRACKLE_THRESHOLD, wheeze_threshold: float = WHEEZE_THRESHOLD) -> Tuple[str, float, float]:
    probs = normalize_model_output(probabilities)
    crackle_prob = float(probs[0, 0])
    wheeze_prob = float(probs[0, 1])

    crackle_pred = crackle_prob >= crackle_threshold
    wheeze_pred = wheeze_prob >= wheeze_threshold

    if not crackle_pred and not wheeze_pred:
        label = "Normal"
    elif crackle_pred and not wheeze_pred:
        label = "Crackle"
    elif wheeze_pred and not crackle_pred:
        label = "Wheeze"
    else:
        label = "Both crackle and wheeze"

    return label, crackle_prob, wheeze_prob


def render_feature_channels(log_mel: np.ndarray, delta: np.ndarray, delta_delta: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    channel_data = [log_mel, delta, delta_delta]
    titles = ["Log-Mel", "Delta", "Delta-Delta"]

    for ax, data, title in zip(axes, channel_data, titles):
        im = ax.imshow(data, aspect="auto", origin="lower", cmap="magma")
        ax.set_title(title)
        ax.set_xlabel("Time frames")
        ax.set_ylabel("Mel bins")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    st.pyplot(fig)


def list_audio_files() -> list[Path]:
    return sorted(RAW_AUDIO_DIR.glob("*.wav"))


def load_single_model_prediction(model_key: str, audio: np.ndarray) -> tuple[str, float, float, str]:
    registry = load_model_registry()
    model = registry.get(model_key)
    if model is None:
        raise ValueError(f"Model {model_key} is not available in this project.")

    x_input = build_three_channel_input(audio)
    probabilities = model.predict(x_input, verbose=0)
    label, crackle_prob, wheeze_prob = decode_prediction(probabilities)
    return label, crackle_prob, wheeze_prob, MODEL_CATALOG[model_key]["label"]


def main() -> None:
    st.set_page_config(page_title="Respiratory Sound Analysis", layout="wide")
    st.title("Respiratory Sound Analysis")
    st.caption("Model comparison using the project’s saved checkpoint family: weighted 3-channel CNN, patient-balanced 3-channel CNN, original CNN-LSTM (M0), and device-aligned CNN-LSTM (M2).")

    registry = load_model_registry()
    available_models = [key for key in MODEL_KEYS if key in registry]

    st.sidebar.header("Model selection")
    st.sidebar.write(f"Available checkpoints: {len(available_models)}/{len(MODEL_KEYS)}")
    for key in MODEL_KEYS:
        if key in registry:
            st.sidebar.markdown(f"- {MODEL_CATALOG[key]['label']}: available")
        else:
            st.sidebar.markdown(f"- {MODEL_CATALOG[key]['label']}: missing")

    compare_all = st.sidebar.checkbox("Compare all available models", value=False)
    selected_key = st.sidebar.selectbox(
        "Select a model",
        options=MODEL_KEYS,
        format_func=lambda key: MODEL_CATALOG[key]["label"],
        index=MODEL_KEYS.index(DEFAULT_MODEL_KEY) if DEFAULT_MODEL_KEY in MODEL_KEYS else 0,
    )

    wav_files = list_audio_files()
    if not wav_files:
        st.error("No WAV files were found in the project data/raw directory.")
        return

    selected_path = st.selectbox("Select an ICBHI recording", wav_files, format_func=lambda p: p.name)

    audio, sr = sf.read(str(selected_path), dtype="float32")
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio_segment = fit_audio_length(audio, TARGET_SAMPLES)
    log_mel = extract_logmel(audio_segment)
    log_mel = fit_logmel_frames(log_mel, TARGET_FRAMES)
    delta = librosa.feature.delta(log_mel, width=min(9, log_mel.shape[1] if log_mel.shape[1] % 2 == 1 else log_mel.shape[1] - 1), order=1, axis=-1)
    delta_delta = librosa.feature.delta(log_mel, width=min(9, log_mel.shape[1] if log_mel.shape[1] % 2 == 1 else log_mel.shape[1] - 1), order=2, axis=-1)

    cols = st.columns(2)
    with cols[0]:
        st.subheader("Input recording")
        st.write(f"Recording: {selected_path.name}")
        st.write(f"Sample rate: {sr} Hz")
        st.line_chart(np.asarray(audio), use_container_width=True)

    if compare_all:
        st.subheader("Model comparison")
        comparison_rows = []
        for key in MODEL_KEYS:
            if key not in registry:
                continue
            model = registry[key]
            probabilities = model.predict(build_three_channel_input(audio), verbose=0)
            label, crackle_prob, wheeze_prob = decode_prediction(probabilities)
            comparison_rows.append(
                {
                    "Model": MODEL_CATALOG[key]["label"],
                    "Crackle probability": round(crackle_prob, 4),
                    "Wheeze probability": round(wheeze_prob, 4),
                    "Prediction": label,
                }
            )

        if comparison_rows:
            st.table(comparison_rows)
        else:
            st.warning("No model checkpoints are available for comparison in this workspace.")

    selected_model = registry.get(selected_key)
    if selected_model is None:
        st.error(f"The selected checkpoint for {MODEL_CATALOG[selected_key]['label']} is missing. Please restore the file to continue.")
        return

    with cols[1]:
        st.subheader("Selected model output")
        probabilities = selected_model.predict(build_three_channel_input(audio), verbose=0)
        label, crackle_prob, wheeze_prob = decode_prediction(probabilities)
        st.metric("Model", MODEL_CATALOG[selected_key]["label"])
        st.metric("Crackle probability", f"{crackle_prob:.4f}")
        st.metric("Wheeze probability", f"{wheeze_prob:.4f}")
        st.metric("Predicted class", label)

    st.subheader("Three-channel feature representation")
    render_feature_channels(log_mel, delta, delta_delta)

    st.subheader("Reference metrics from the project notebooks")
    metric_cols = st.columns(3)
    for idx, (name, value) in enumerate(HISTORICAL_METRICS.items()):
        with metric_cols[idx % 3]:
            st.metric(name, f"{value:.4f}" if isinstance(value, float) else str(value))

    st.caption("The app loads the actual saved Keras checkpoints from the project and compares them on the same input audio. It does not re-train or claim scores that are not recorded in the project notebooks.")


if __name__ == "__main__":
    main()
