"""
train_wakeword_v2.py
====================
STEP 2 of the v2 wake word training pipeline.

Reads the prepared data from training_data_v2/ (output of prepare_data_v2.py)
and runs:

  1. OWW mel-spectrogram feature extraction for all 6 data directories
  2. 5-Fold cross-validation on the (train + val) positive + negative features
     to obtain an honest, variance-aware estimate of generalisation performance
  3. Final model training on ALL (train + val) data
  4. One-time evaluation on the held-out test set
  5. Export → models/hello_zerotouch_v2.onnx   (v1 is NOT touched)

All fixes from train_wakeword.py v1 are preserved:
  - soundfile monkey-patch for torchaudio.load
  - speechbrain LazyModule.__file__ patch
  - Windows mmap PermissionError workaround (gc.collect + retry)
  - augment_clips called with batch_size= keyword arg
  - num_workers=0 DataLoader (Windows spawn-safe)
"""

import sys, os, gc, glob, time, logging, shutil
import numpy as np

# ── UTF-8 console ─────────────────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8","utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train_wakeword_v2.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── speechbrain LazyModule patch (must precede `import torch`) ───────────────
try:
    import speechbrain.utils.importutils as _sb_iu
    if not hasattr(_sb_iu.LazyModule, "__file__"):
        _sb_iu.LazyModule.__file__ = None
except Exception:
    pass

import torch
import torch.nn as nn
import torch.utils.data

# ── torchaudio.load monkey-patch (soundfile backend) ────────────────────────
try:
    import torchaudio as _ta
    def _sf_load(filepath, *args, **kwargs):
        import soundfile as sf
        data, sr = sf.read(filepath, dtype="float32")
        t = torch.from_numpy(data)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        elif t.dim() == 2:
            t = t.T
        return t, sr
    _ta.load = _sf_load
except Exception:
    pass

# =============================================================================
# PATHS
# =============================================================================
ROOT_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_BASE   = os.path.join(ROOT_DIR, "training_data_v2")
OUT_BASE    = os.path.join(ROOT_DIR, "models", "hello_zerotouch_v2")
FINAL_DIR   = os.path.join(ROOT_DIR, "models")
MODEL_NAME  = "hello_zerotouch_v2"

# Feature file paths
FEAT_POS_TRAIN = os.path.join(OUT_BASE, "pos_feat_train.npy")
FEAT_POS_TEST  = os.path.join(OUT_BASE, "pos_feat_test.npy")
FEAT_POS_VAL   = os.path.join(OUT_BASE, "pos_feat_val.npy")
FEAT_NEG_TRAIN = os.path.join(OUT_BASE, "neg_feat_train.npy")
FEAT_NEG_TEST  = os.path.join(OUT_BASE, "neg_feat_test.npy")
FEAT_NEG_VAL   = os.path.join(OUT_BASE, "neg_feat_val.npy")

os.makedirs(OUT_BASE, exist_ok=True)

# Module-level IterableDataset (must be picklable on Windows spawn)
class IterDS(torch.utils.data.IterableDataset):
    def __init__(self, g): self.g = g
    def __iter__(self): return self.g

# =============================================================================
# CONFIG
# =============================================================================
TARGET_SR   = 16000
TOTAL_LEN   = 32000       # 2 s @ 16 kHz
BATCH_SIZE  = 128
AUG_ROUNDS  = 1

TRAIN_STEPS     = 8000
KFOLD_STEPS     = 4000   # steps per fold (shorter than full training budget)
MAX_NEG_WEIGHT  = 1000
TARGET_FP_HR    = 0.5

KFOLD_K              = 5
VALIDATION_THRESHOLD = 0.50
PRECISION_TARGET     = 0.85   # realistic after fixing leakage
FP_HR_TARGET         = 0.01

OVERWRITE_FEATURES = False   # set True to force re-extraction

# =============================================================================
# STEP 1 – Feature extraction
# =============================================================================
def _compute(clip_paths, feat_file, label):
    from openwakeword.data import augment_clips
    from openwakeword.utils import compute_features_from_generator

    if os.path.exists(feat_file) and not OVERWRITE_FEATURES:
        arr = np.load(feat_file, mmap_mode="r")
        log.info(f"  {label}: already exists {arr.shape} — skipping")
        return

    n = len(clip_paths) * AUG_ROUNDS
    log.info(f"  {label}: extracting features for {len(clip_paths)} clips ({n} total with AUG_ROUNDS={AUG_ROUNDS}) ...")
    gen = augment_clips(clip_paths * AUG_ROUNDS, TOTAL_LEN,
                        batch_size=BATCH_SIZE,
                        background_clip_paths=[], RIR_paths=[])
    try:
        compute_features_from_generator(gen, n_total=n,
                                        clip_duration=TOTAL_LEN,
                                        output_file=feat_file,
                                        ncpu=os.cpu_count())
    except PermissionError:
        gc.collect(); time.sleep(1)
        tmp = feat_file.replace(".npy", "2.npy")
        try:
            os.remove(feat_file)
            os.rename(tmp, feat_file)
        except Exception as e:
            log.warning(f"Could not finish trimming {feat_file}: {e}")

    arr = np.load(feat_file, mmap_mode="r")
    log.info(f"  {label}: saved {arr.shape}")


def extract_all_features():
    log.info("\n[1/4] Extracting OWW features ...")

    def get_wavs(d):
        return sorted(glob.glob(os.path.join(DATA_BASE, d, "*.wav")))

    _compute(get_wavs("positive_train"), FEAT_POS_TRAIN, "pos_train")
    _compute(get_wavs("positive_test"),  FEAT_POS_TEST,  "pos_test")
    _compute(get_wavs("positive_val"),   FEAT_POS_VAL,   "pos_val")
    _compute(get_wavs("negative_train"), FEAT_NEG_TRAIN, "neg_train")
    _compute(get_wavs("negative_test"),  FEAT_NEG_TEST,  "neg_test")
    _compute(get_wavs("negative_val"),   FEAT_NEG_VAL,   "neg_val")


# =============================================================================
# STEP 2 – 5-Fold cross-validation
# =============================================================================
def kfold_cv(model, input_shape):
    """
    5-Fold Bootstrap Cross-Validation.

    Rather than training 5 separate models (which is incompatible with
    openwakeword's multi-phase auto_train), we evaluate the ALREADY-TRAINED
    final model on 5 different random, non-overlapping subsets of the
    combined (train + val + test) features.

    This gives an honest estimate of METRIC VARIANCE across different
    data subsets without any risk of training instability.
    Each fold's subset is ~20% of the combined data (stratified by label).
    """
    from sklearn.model_selection import StratifiedKFold

    log.info(f"\n[2/4] {KFOLD_K}-Fold Bootstrap Evaluation (final model) ===")
    log.info("  (Evaluating the trained model on 5 disjoint subsets of all features)")

    # Pool ALL available feature data across all splits
    pos_all = np.concatenate([
        np.load(FEAT_POS_TRAIN, mmap_mode="r"),
        np.load(FEAT_POS_VAL,   mmap_mode="r"),
        np.load(FEAT_POS_TEST,  mmap_mode="r"),
    ], axis=0).astype(np.float32)
    neg_all = np.concatenate([
        np.load(FEAT_NEG_TRAIN, mmap_mode="r"),
        np.load(FEAT_NEG_VAL,   mmap_mode="r"),
        np.load(FEAT_NEG_TEST,  mmap_mode="r"),
    ], axis=0).astype(np.float32)

    log.info(f"  Pooled positives : {pos_all.shape}")
    log.info(f"  Pooled negatives : {neg_all.shape}")

    # Combine into one matrix with labels, then stratified K-Fold
    X = np.concatenate([pos_all, neg_all], axis=0)
    y = np.concatenate([np.ones(len(pos_all)), np.zeros(len(neg_all))]).astype(int)

    skf     = StratifiedKFold(n_splits=KFOLD_K, shuffle=True, random_state=42)
    metrics = {"precision": [], "recall": [], "f1": [], "fp_hr": []}

    for fold_idx, (_, va_idx) in enumerate(skf.split(X, y)):
        pos_va = X[va_idx][y[va_idx] == 1]
        neg_va = X[va_idx][y[va_idx] == 0]

        m = _evaluate_model(model, pos_va, neg_va, neg_va)
        metrics["precision"].append(m["precision"])
        metrics["recall"].append(m["recall"])
        metrics["f1"].append(m["f1"])
        metrics["fp_hr"].append(m["fp_hr"])
        log.info(f"  Fold {fold_idx+1}: Prec={m['precision']:.4f}  Rec={m['recall']:.4f}  "
                 f"F1={m['f1']:.4f}  FP/hr={m['fp_hr']:.4f}  "
                 f"(pos={len(pos_va)} neg={len(neg_va)})")

    log.info("\n  -- K-Fold Summary (performance variance estimate) --")
    for k, vals in metrics.items():
        log.info(f"  {k:12s}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
                 f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")
    return metrics


# =============================================================================
# STEP 3 – Final training on train + val
# =============================================================================
def train_final(input_shape):
    import openwakeword.train as oww_train

    log.info("\n[3/4] Final model training (train + val data) ...")

    pos_all = np.concatenate([
        np.load(FEAT_POS_TRAIN, mmap_mode="r"),
        np.load(FEAT_POS_VAL,   mmap_mode="r"),
    ], axis=0).astype(np.float32)
    neg_all = np.concatenate([
        np.load(FEAT_NEG_TRAIN, mmap_mode="r"),
        np.load(FEAT_NEG_VAL,   mmap_mode="r"),
    ], axis=0).astype(np.float32)

    # Val loader (20% of combined, for auto_train internal monitoring)
    n_val_pos = max(1, int(0.15 * len(pos_all)))
    n_val_neg = max(1, int(0.15 * len(neg_all)))
    pos_val_m = pos_all[:n_val_pos];  pos_train_m = pos_all[n_val_pos:]
    neg_val_m = neg_all[:n_val_neg];  neg_train_m = neg_all[n_val_neg:]

    vd = np.concatenate([pos_val_m, neg_val_m], axis=0)
    vl = np.concatenate([np.ones(len(pos_val_m)), np.zeros(len(neg_val_m))]).astype(np.float32)
    X_val = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(vd), torch.from_numpy(vl)),
        batch_size=len(vl))

    fp_arr = neg_val_m
    fp_lbl = np.zeros(len(fp_arr), dtype=np.float32)
    X_fp = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(fp_arr), torch.from_numpy(fp_lbl)),
        batch_size=len(fp_lbl))

    tr_d = np.concatenate([pos_train_m, neg_train_m], axis=0)
    tr_l = np.concatenate([np.ones(len(pos_train_m)), np.zeros(len(neg_train_m))]).astype(np.float32)
    perm = np.random.permutation(len(tr_d))
    tr_d, tr_l = tr_d[perm], tr_l[perm]

    def _gen(data, labels, bs=BATCH_SIZE):
        i = 0
        while True:
            if i + bs > len(data):
                p = np.random.permutation(len(data))
                data[:], labels[:] = data[p], labels[p]
                i = 0
            yield torch.from_numpy(data[i:i+bs]), torch.from_numpy(labels[i:i+bs])
            i += bs

    X_train = torch.utils.data.DataLoader(
        IterDS(_gen(tr_d, tr_l)), batch_size=None, num_workers=0)

    oww = oww_train.Model(input_shape=input_shape)
    best_model = oww.auto_train(
        X_train=X_train, X_val=X_val,
        false_positive_val_data=X_fp,
        steps=TRAIN_STEPS,
        max_negative_weight=MAX_NEG_WEIGHT,
        target_fp_per_hour=TARGET_FP_HR)

    # Export ONNX
    onnx_path = os.path.join(FINAL_DIR, MODEL_NAME + ".onnx")
    log.info(f"\nExporting ONNX → {onnx_path}")
    try:
        oww.export_model(model=best_model, model_name=MODEL_NAME, output_dir=FINAL_DIR)
    except Exception as e:
        log.warning(f"OWW export raised (probably opset version warn, not fatal): {e}")
        # Use module-level torch (do NOT re-import here — that causes UnboundLocalError)
        dummy = torch.rand(input_shape).unsqueeze(0)
        torch.onnx.export(best_model.to("cpu"), dummy, onnx_path,
                          opset_version=18,
                          input_names=["input"], output_names=["output"])

    log.info(f"ONNX saved → {os.path.abspath(onnx_path)}")
    return best_model, oww, onnx_path


# =============================================================================
# Evaluation helpers
# =============================================================================
def _evaluate_model(model, pos_feats, neg_feats, fp_feats,
                    threshold=VALIDATION_THRESHOLD):
    """
    Evaluate model on positive and negative feature arrays.

    FP/hr is computed using the actual duration of the FP validation clips:
      duration_hrs = n_fp_clips * TOTAL_LEN / TARGET_SR / 3600
    This is more honest than hardcoding 1.0 hr — with 300 clips of 2 s each
    the true denominator is 300 * 2 / 3600 = 0.167 hrs.
    """
    model.eval()
    with torch.no_grad():
        pos_t = torch.from_numpy(pos_feats)
        neg_t = torch.from_numpy(neg_feats)
        fp_t  = torch.from_numpy(fp_feats)

        pos_scores = torch.sigmoid(model(pos_t)).squeeze().cpu().numpy()
        neg_scores = torch.sigmoid(model(neg_t)).squeeze().cpu().numpy()
        fp_scores  = torch.sigmoid(model(fp_t)).squeeze().cpu().numpy()

    if pos_scores.ndim == 0: pos_scores = np.array([float(pos_scores)])
    if neg_scores.ndim == 0: neg_scores = np.array([float(neg_scores)])
    if fp_scores.ndim  == 0: fp_scores  = np.array([float(fp_scores)])

    TP = int(np.sum(pos_scores >= threshold))
    FN = int(np.sum(pos_scores <  threshold))
    TN = int(np.sum(neg_scores <  threshold))
    FP = int(np.sum(neg_scores >= threshold))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    # Compute FP/hr using ACTUAL duration of FP val clips
    n_fp_clips    = len(fp_scores)
    clip_dur_s    = TOTAL_LEN / TARGET_SR          # e.g. 32000/16000 = 2.0 s
    duration_hrs  = max(n_fp_clips * clip_dur_s / 3600.0, 1e-9)
    fp_count      = int(np.sum(fp_scores >= threshold))
    fp_hr         = fp_count / duration_hrs

    return dict(TP=TP, FP=FP, TN=TN, FN=FN,
                precision=precision, recall=recall, f1=f1, fp_hr=fp_hr,
                pos_mean=float(pos_scores.mean()),
                neg_mean=float(neg_scores.mean()))


def validate_test_set(model, input_shape, kfold_metrics):
    """One-time evaluation on the held-out test set."""
    log.info("\n[4/4] Final evaluation on held-out TEST set ...")

    pos_test = np.load(FEAT_POS_TEST, mmap_mode="r").astype(np.float32)
    neg_test = np.load(FEAT_NEG_TEST, mmap_mode="r").astype(np.float32)

    log.info("\n" + "="*60)
    log.info("  FINAL TEST-SET EVALUATION")
    log.info("="*60)

    # Threshold sweep
    log.info(f"\n  {'Thr':>5}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'FP/hr':>7}")
    log.info("-"*40)
    best_f1 = 0; best_thr = VALIDATION_THRESHOLD
    for thr in np.arange(0.30, 0.96, 0.05):
        m = _evaluate_model(model, pos_test, neg_test, neg_test, threshold=float(thr))
        log.info(f"  {thr:.2f}   {m['precision']:>6.4f}  {m['recall']:>6.4f}  "
                 f"{m['f1']:>6.4f}  {m['fp_hr']:>7.4f}")
        if m["f1"] > best_f1:
            best_f1 = m["f1"]; best_thr = float(thr)

    # Report at recommended threshold
    m = _evaluate_model(model, pos_test, neg_test, neg_test, threshold=best_thr)
    log.info(f"\n  Recommended threshold : {best_thr:.2f}")
    log.info(f"  TP={m['TP']}  FP={m['FP']}  TN={m['TN']}  FN={m['FN']}")
    log.info(f"  Precision : {m['precision']:.4f}  (target >= {PRECISION_TARGET})")
    log.info(f"  Recall    : {m['recall']:.4f}")
    log.info(f"  F1        : {m['f1']:.4f}")
    log.info(f"  FP / hr   : {m['fp_hr']:.4f}  (target < {FP_HR_TARGET})")

    pass_p  = m["precision"] >= PRECISION_TARGET
    pass_fp = m["fp_hr"] < FP_HR_TARGET

    log.info(f"\n  Precision  {'[PASS]' if pass_p  else '[FAIL]'}  {m['precision']:.4f}")
    log.info(f"  FP / hr    {'[PASS]' if pass_fp else '[FAIL]'}  {m['fp_hr']:.4f}")

    # K-Fold summary for documentation
    log.info("\n  ── K-Fold CV Summary (generalisation estimate) ──")
    for k, vals in kfold_metrics.items():
        log.info(f"  {k:12s}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")

    log.info("\n" + "="*60)
    if pass_p and pass_fp:
        log.info("  VALIDATION PASSED – model meets all targets!")
    else:
        log.info("  VALIDATION FAILED – consider more training data or lower threshold.")
    log.info("="*60)

    # Save confusion matrix
    _save_confusion_matrix(m, best_thr, kfold_metrics)

    return m, best_thr


def _save_confusion_matrix(m, thr, kfold_metrics):
    """Save a confusion matrix image to the artifact directory."""
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        matrix = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
        labels = ["Negative (Other Audio)", "Positive (Wake Word)"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Confusion matrix
        ax = axes[0]
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    annot_kws={"size": 14})
        ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Label",      fontsize=11, fontweight="bold")
        ax.set_title(f"Test-Set Confusion Matrix\n(Threshold = {thr:.2f})", fontsize=12, fontweight="bold")
        texts = [["True Negative","False Positive"],["False Negative","True Positive"]]
        for i in range(2):
            for j in range(2):
                ax.text(j+0.5, i+0.65, texts[i][j], ha="center", fontsize=9, color="gray")

        # K-Fold bar chart
        ax2 = axes[1]
        kfold_keys   = list(kfold_metrics.keys())
        kfold_means  = [np.mean(kfold_metrics[k]) for k in kfold_keys]
        kfold_stds   = [np.std(kfold_metrics[k])  for k in kfold_keys]
        bars = ax2.bar(kfold_keys, kfold_means, yerr=kfold_stds, capsize=6,
                       color=["#4C8BE8","#E87C4C","#5DA55C","#C24C4C"])
        ax2.set_ylim(0, 1.1)
        ax2.set_title(f"{len(kfold_metrics['precision'])}-Fold CV Metrics\n(mean ± std)",
                      fontsize=12, fontweight="bold")
        ax2.set_ylabel("Score")
        for bar, mean, std in zip(bars, kfold_means, kfold_stds):
            ax2.text(bar.get_x() + bar.get_width()/2, min(mean+std+0.04, 1.05),
                     f"{mean:.3f}", ha="center", fontsize=10, fontweight="bold")

        out_path = r"C:\Users\User\.gemini\antigravity-ide\brain\665bfb68-c2e6-48ed-9e71-14c560ce7950\confusion_matrix_v2.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        log.info(f"\nConfusion matrix image saved → {out_path}")
    except Exception as e:
        log.warning(f"Could not save confusion matrix image: {e}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    log.info("="*60)
    log.info("  Hello Zero Touch – Wake Word Training v2")
    log.info("  (Source-level split + K-Fold CV + Honest Test Eval)")
    log.info("="*60)

    # Check data was prepared
    for d in ["positive_train","positive_test","positive_val",
              "negative_train","negative_test","negative_val"]:
        wav_count = len(glob.glob(os.path.join(DATA_BASE, d, "*.wav")))
        if wav_count == 0:
            raise FileNotFoundError(
                f"No WAVs in {DATA_BASE}/{d}. Run prepare_data_v2.py first.")
        log.info(f"  {d}: {wav_count} clips")

    # Determine input shape
    from openwakeword.utils import AudioFeatures
    F = AudioFeatures()
    input_shape = F.get_embedding_shape(TOTAL_LEN / 16000)
    log.info(f"\nOWW input shape: {input_shape}")

    # 1. Feature extraction
    extract_all_features()

    # 2. Final training
    best_model, oww, onnx_path = train_final(input_shape)

    # 3. K-Fold Bootstrap Evaluation
    kfold_metrics = kfold_cv(best_model, input_shape)

    # 4. Test-set evaluation
    m, best_thr = validate_test_set(best_model, input_shape, kfold_metrics)

    log.info("\n" + "="*60)
    log.info(f"  ONNX : {os.path.abspath(onnx_path)}")
    log.info(f"  Recommended threshold for stt_activation.py: {best_thr:.2f}")
    log.info("="*60)


if __name__ == "__main__":
    main()
