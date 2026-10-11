# Results

All figures here are generated from `metrics.json` (see
`../Code/src/make_figures.py`), so the plots always match the numbers in the
report.

| File | What it shows |
|---|---|
| `metrics.json` | Accuracy / macro-F1 / count-MAE, confusion matrices, and Random Forest feature importances from the M2 run |
| `model_comparison.png` | Accuracy and macro-F1 for the three models (leave-one-session-out) |
| `confusion_matrix_threshold_amp-variance.png` | Threshold baseline confusion matrix |
| `confusion_matrix_random_forest.png` | Random Forest confusion matrix |
| `confusion_matrix_mlp.png` | MLP confusion matrix |
| `feature_importance.png` | Random Forest feature importances (12 features) |
| `amp_variance_vs_people.png` | Amplitude variance (feature 0) vs. number of people |

**Dataset:** RadioPoints WiFi-CrowdCounting (Intel 5300, 3 rooms, 0–2 people),
cited in the report. These are real measured CSI results, but collected by the
dataset authors, not on our own testbed.

**Protocol:** leave-one-session-out cross-validation over 2914 analysis windows;
no window from a held-out session is seen during training.

Best model at M2: **Random Forest — 53.3% accuracy, 0.546 macro-F1, 0.468
count-MAE**, versus a 33% three-class chance level and a 51.5% non-AI
threshold baseline.
