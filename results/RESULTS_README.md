# Recorded ML Results

These CSV files summarize experiment outputs documented in the project report and saved notebook outputs.

## Important reproducibility note
- `validation_model_comparison.csv` contains historical aggregate validation results across the 649-cycle validation set.
- `selected_model_metrics.csv` contains the recorded validation and evaluated-test metrics for the selected CNN-LSTM, including frozen event thresholds.
- `device_alignment_historical.csv` contains historical device-wise M0 vs M2 validation aggregates.
- These tables are documentation of recorded results, not a claim that the original experiments were fully rerun from the restored assets.
- The current repository includes a saved test prediction CSV, but some historical sample-level arrays and feature tensors used for the original alignment comparison were not preserved.
