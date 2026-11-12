# Model inventory

The supplied checkpoints are retained in `models/` so the experiment history is preserved.

| Checkpoint | Role / note |
|---|---|
| `best_logmel_cnn.keras` | Log-Mel CNN experiment |
| `best_resized_logmel_cnn.keras` | Time-resized Log-Mel CNN experiment |
| `best_compact_logmel_cnn.keras` | Compact Log-Mel CNN experiment |
| `best_fold3_compact_cnn.keras` | Fold-specific compact CNN checkpoint |
| `best_framepad_gap_cnn.keras` | Frame-padding / GAP CNN experiment |
| `best_framepad_multilabel_cnn.keras` | Frame-padding multilabel CNN experiment |
| `best_framepad_multilabel_gap.keras` | Frame-padding multilabel GAP experiment |
| `best_framepad_multilabel_mfcc.keras` | Frame-padding MFCC multilabel experiment |
| `best_framepad_fusion.keras` | Feature/model fusion experiment |

The final report identifies a recorded CNN-LSTM baseline as the selected four-class development model, with 107,906 parameters. That exact selected CNN-LSTM checkpoint is **not among the uploaded checkpoint files** used to build this GitHub package. The supplied checkpoints should therefore be treated as experiment artifacts unless their exact architecture and preprocessing contract are verified from the corresponding notebook/model metadata.
