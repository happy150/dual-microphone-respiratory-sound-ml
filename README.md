# ML demo

This folder contains an additive demo dashboard for the respiratory sound classification project.

It does not modify the original notebooks, checkpoints, or dataset files.

## Run

```bash
cd C:/Users/Admin/Respiratory_TinyML
.\.venv\Scripts\python.exe -m streamlit run .\ml_demo\app.py
```

## Notes

- Uses the original saved checkpoint: `notebooks/best_cnn_lstm_patient_weighted.keras`
- Uses the preprocessing path documented in `notebooks/02_build_dataset.ipynb`
- Validates a real WAV file from `data/raw`
- Shows waveform, three feature channels, probabilities, and historical evaluation metrics
