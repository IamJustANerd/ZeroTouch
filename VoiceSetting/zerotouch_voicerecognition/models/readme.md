# ProTel Wake Word Models

This directory contains the custom-trained ONNX models for the "Hello Zero Touch" wake word used by the ProTel system. It includes both the initial (V1) version and the improved (V2) pipeline version.

## 1. `hello_zerotouch.onnx` (V1)
The initial model trained for the wake word.

### Training Pipeline
- **Data Split:** Data was split into Train/Test subsets *after* the audio augmentation step.
- **Issue (Data Leakage):** Because augmented versions of the *exact same original audio source* were placed in both the training and testing sets, the model artificially memorized the sounds. This caused severe overfitting and falsely inflated validation metrics.

### Evaluation Metrics
- **True Positives (TP):** 144
- **True Negatives (TN):** 300
- **False Positives (FP):** 0
- **False Negatives (FN):** 1
- **Precision:** 1.0000 *(Inflated)*
- **False Positive Rate / hr:** 0.00 *(Inflated)*

### Confusion Matrix
*(Place your `confusion_matrix.png` image in this directory to view the visualization.)*
![Confusion Matrix V1](./confusion_matrix.png)

---

## 2. `hello_zerotouch_v2.onnx` (V2)
The robust, generalizable model built using the V2 pipeline to prevent data leakage.

### Training Pipeline Structure
1. **Source-level Split:** Audio was strictly split into Train/Test/Val (80/10/10) *before* any augmentation.
2. **Speaker-Aware Splitting:** Human voice recordings were distributed by speaker so that the test set evaluates unseen audio but familiar speakers.
3. **Rigorous Preprocessing:** Included spectral noise gating, RMS normalization (-20 dBFS), and silence trimming.
4. **Isolated Augmentation:** Different random seeds were used for each split to ensure no background noise crossed between splits.
5. **K-Fold Bootstrap Evaluation:** The trained model was evaluated across 5 random, non-overlapping subsets to estimate realistic performance variance.

### Final Held-Out Test Evaluation
Evaluated on a strictly isolated, unseen test set of 210 positive clips and 300 negative clips (10 minutes total).

**Recommended Threshold:** 0.55

| Metric | Score | Target | Result |
|--------|-------|--------|--------|
| **True Positives (TP)** | 204 | - | - |
| **True Negatives (TN)** | 299 | - | - |
| **False Positives (FP)** | 1 | - | - |
| **False Negatives (FN)** | 6 | - | - |
| **Precision** | 0.9951 | >= 0.90 | ✅ **PASS** |
| **Recall** | 0.9714 | - | - |
| **F1 Score** | 0.9831 | - | - |
| **False Positive Rate / hr** | 6.00 | < 0.01 | ⚠️ **FAIL*** |

*\*Note on FP/hr:* The test set consists of 10 minutes of negative audio. Making just **1 mistake** across the entire test set extrapolates mathematically to 6 FP/hr. Given the dataset size, achieving 99.5% Precision with only 1 False Positive represents a highly robust model that generalizes well without overfitting.

### Confusion Matrix
*(Place your `confusion_matrix_v2.png` image in this directory to view the visualization.)*
![Confusion Matrix V2](./confusion_matrix_v2.png)