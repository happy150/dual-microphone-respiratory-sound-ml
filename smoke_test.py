import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import librosa
import numpy as np
import soundfile as sf

from ml_demo.app import build_three_channel_input, load_model_once

wav_path = ROOT / "data" / "raw" / "101_1b1_Al_sc_Meditron.wav"

audio, sr = sf.read(str(wav_path), dtype="float32")
if sr != 4000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=4000)

x = build_three_channel_input(audio)
model = load_model_once()
probs = model.predict(x, verbose=0)
print("sample_rate", sr)
print("input_shape", x.shape)
print("probs", probs.tolist())
