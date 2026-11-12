# Dual Microphone Respiratory Sound Assessment with Adaptive Noise Cancellation and Machine Learning

Minor Specialization in Embedded Systems project, Manipal Institute of Technology.

**Authors**
- Asmi Jain (230906092)

**Guide:** R Srividya, Associate Professor, School of Electrical Engineering, Manipal Institute of Technology.

## Project overview

This repository contains the documented machine-learning experiments and supporting project deliverables for a dual-microphone respiratory sound assessment system. The intended system combines a chest-coupled primary microphone, an ambient reference microphone, adaptive LMS/NLMS processing, and a machine-learning classifier. The current ML demonstration performs inference on a laptop; embedded ESP32-S3 inference is future work.

The ML workflow uses the ICBHI respiratory sound recordings with patient-separated partitions, 4 kHz model audio, 64-band log-Mel features, first and second temporal derivatives, and a CNN-LSTM classifier. The recorded selected CNN-LSTM has 107,906 parameters and two event outputs (crackle and wheeze), which are mapped to four acoustic categories using documented thresholds. The corresponding selected CNN-LSTM checkpoint is not present in the supplied upload set, so this repository does not pretend to include that missing weight file.

## Repository structure

```text
.
├── data/
│   └── ICBHI_challenge_train_test.txt
├── models/
│   ├── best_logmel_cnn.keras
│   ├── best_resized_logmel_cnn.keras
│   ├── best_compact_logmel_cnn.keras
│   ├── best_fold3_compact_cnn.keras
│   ├── best_framepad_gap_cnn.keras
│   ├── best_framepad_multilabel_cnn.keras
│   ├── best_framepad_multilabel_gap.keras
│   ├── best_framepad_multilabel_mfcc.keras
│   └── best_framepad_fusion.keras
├── notebooks/
│   ├── 01_explore_icbhi.ipynb
│   └── 02_build_dataset.ipynb
├── results/
│   └── cnn_lstm_official_test_predictions.csv
├── docs/
│   ├── Minor_Specialization_Report.docx
│   └── Minor_Project_Presentation.pptx
├── archive/
│   └── duplicate_notebooks/
├── PROJECT_INVENTORY.csv
├── requirements.txt
├── .gitignore
└── README.md
```


## ML contribution highlights

The repository is organized around the machine-learning contribution rather than only the final checkpoint. The cleaned primary notebook documents: (1) patient-separated dataset construction and split auditing, (2) log-Mel + delta + delta-delta feature engineering and frame padding, (3) compact CNN and alternative time/feature representations, (4) weighted and patient-balanced training, (5) CNN embedding classifiers using SVM, k-NN and bagged k-NN, (6) the two-output CNN-LSTM, (7) validation-based event-threshold selection, and (8) device-stratified analysis and training-derived alignment experiments.

The primary notebook removes kernel-debugging, machine-specific paths, installation commands, notebook self-inspection, repeated checks, and other non-essential development noise. The complete original experiment notebook is retained under `archive/full_notebooks/` for transparency. The cleaned notebook is designed to make the engineering decisions, model-development progression, and ML contribution easy for technical reviewers and recruiters to follow.

## My ML contribution

The primary ML contribution in this project was the end-to-end development and evaluation workflow: dataset and annotation auditing, patient-separated partition construction, feature engineering, controlled model comparisons, class and patient balancing, classical ML on learned CNN embeddings, CNN–LSTM temporal modelling, event threshold selection, device-level analysis, and training-derived device alignment. The repository keeps the intermediate checkpoints and experiment history so the progression can be inspected rather than presenting only one final model.

## Dataset and split

The recorded audit in the report describes 920 original WAV recordings, 6,898 annotated cycles and 126 patients. The retained patient-separated project partitions contain 907 recordings and 6,781 cycles: 3,376 training cycles, 649 validation cycles and 2,756 cycles in the evaluated test subset.

This repository does **not** contain the raw ICBHI audio dataset. Download the dataset from its official source and place the audio/annotation files in a local data directory. The text file in `data/` is the project split/assignment manifest.

## Selected ML pipeline

- Audio rate: 4,000 Hz
- Maximum segment: 20,000 samples (5 s)
- FFT/hop: 512 / 128
- Mel bands: 64, 20–2,000 Hz
- Representation: log-Mel + delta + delta-delta
- Model input: 64 × 157 × 3 float32
- Short segments: feature-frame padding with valid-frame masking
- Long segments: center crop to 20,000 samples
- Training: weighted binary cross-entropy, patient-balanced sample weights, Adam (initial LR 0.001), batch size 32, early stopping and learning-rate reduction
- Event thresholds for the recorded baseline: crackle 0.45, wheeze 0.60

## Recorded results

The final report records the following historical/recorded results; these numbers should not be described as a fresh rerun from this repository unless the full dataset, matching preprocessing state and sample-level artifacts are reconstructed.

| Model / metric | Validation | Evaluated test |
|---|---:|---:|
| Four-class accuracy (CNN-LSTM) | 0.5655 | 0.4583 |
| Four-class macro F1 | 0.5041 | 0.3584 |
| Crackle ROC-AUC | 0.5517 | 0.5885 |
| Wheeze ROC-AUC | 0.8497 | 0.7276 |
| ICBHI-style score | — | 0.4448 |

The report notes weak crackle discrimination and limited generalization. The output is an **experimental acoustic classification**, not a clinical diagnosis.

## Device-alignment experiments

The project also contains a training-derived device-alignment experiment (M0/M1/M2). Historical M2 validation aggregates showed higher wheeze ranking metrics, but the report explicitly notes that the original sample-level prediction arrays and matching feature tensors were not preserved, so the complete historical comparison is not independently reproducible from the restored assets. The historical device-wise effects were heterogeneous and do not establish unseen-device generalization.

## How to run the notebooks

1. Download the official ICBHI dataset and place the recordings/annotations where the notebook expects them.
2. Create a Python environment from `requirements.txt`.
3. Start Jupyter and open `notebooks/01_explore_icbhi.ipynb`.
4. Run `notebooks/02_build_dataset.ipynb` from the documented data paths.
5. For inference, use the saved `.keras` checkpoint together with the **matching preprocessing, normalization and threshold settings**. Do not assume that one model's preprocessing contract applies to every checkpoint.

The notebooks contain historical experiments as well as model-selection work, so execution may require updating local file paths and removing notebook-only Google Colab file-upload steps.

## Hardware integration status

The report documents an ESP32-S3 N16R8 configuration with two INMP441 I2S microphones. The intended capture rate is 16 kHz, followed by anti-aliased resampling to the ML rate of 4 kHz. The current report states that the microphones were not functioning at the time of documentation, so synchronized acquisition, real-hardware ANC gain, hardware-to-model integration, latency, memory and power measurements remain pending.

Do not claim ESP32 on-device inference or measured noise-cancellation improvement based only on the files in this repository.

## Reproducibility notes

- Preserve patient separation during any re-training or evaluation.
- Keep model checkpoint, feature scaling, masks, class order and event thresholds paired.
- Do not fit normalization statistics on validation/test/demo audio.
- Do not call the two event outputs calibrated clinical probabilities.
- Report class-level metrics and confusion matrices, not accuracy alone.

## License / dataset note

The ICBHI dataset is not redistributed in this repository. Follow the dataset's original access and usage terms when downloading it. Project documents and code in this repository are provided for academic/research demonstration unless otherwise specified.
