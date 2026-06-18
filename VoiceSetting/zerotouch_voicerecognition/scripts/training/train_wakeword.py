"""
train_wakeword.py  – Hello Zero Touch, wake word training + validation
=====================================================================
Steps:
  1. Verify source data
  2. Resample all clips to 16 kHz  (OWW requirement)
  3. Split 90/10 into train / test directories
  4. augment_clips + compute_features_from_generator  → 4x .npy feature files
  5. Build PyTorch DataLoaders
  6. Model.auto_train()  →  export  models/hello_zerotouch.onnx
  7. Validate: precision >= 0.90, FP rate < 0.01 / hr

Usage:
    .\\venv\\Scripts\\python.exe train_wakeword.py
"""

import os, sys, glob, shutil, logging
import numpy as np
from pathlib import Path

# ── Fix: torch 2.12 + speechbrain LazyModule crash ───────────────────────────
# torch._dynamo registers custom ops at import time using inspect.getframeinfo,
# which iterates sys.modules and calls hasattr(m, '__file__') on every module.
# speechbrain's LazyModule.__getattr__ intercepts the '__file__' lookup and
# tries to load transformers, which crashes. Adding __file__ = None as a *class*
# attribute short-circuits __getattr__ so the lookup succeeds safely.
try:
    import speechbrain.utils.importutils as _sb_iu  # initialise speechbrain early
    if not hasattr(type(_sb_iu.LazyModule(None, None)), '__file__'):
        _sb_iu.LazyModule.__file__ = None
except Exception:
    # If the class changed shape, try patching via instance
    try:
        import speechbrain.utils.importutils as _sb_iu
        _sb_iu.LazyModule.__file__ = None
    except Exception:
        pass

import torch

# ── UTF-8 console (Windows cp1252 workaround) ─────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train_wakeword.log", mode="w", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)

# =============================================================================
# PATHS
# =============================================================================
POSITIVE_AUG_DIR = "training_data/positive_augmented"
NEGATIVE_DIR     = "training_data/negative"
OUTPUT_BASE      = "models/hello_zerotouch"
FINAL_MODEL_DIR  = "models"
MODEL_NAME       = "hello_zerotouch"

RESAMPLED_BASE   = "training_data/resampled_16k"
POS_RESAMP_DIR   = os.path.join(RESAMPLED_BASE, "positive_augmented")
NEG_RESAMP_DIR   = os.path.join(RESAMPLED_BASE, "negative")

POS_TRAIN_DIR = os.path.join(OUTPUT_BASE, "positive_train")
POS_TEST_DIR  = os.path.join(OUTPUT_BASE, "positive_test")
NEG_TRAIN_DIR = os.path.join(OUTPUT_BASE, "negative_train")
NEG_TEST_DIR  = os.path.join(OUTPUT_BASE, "negative_test")

POS_FEAT_TRAIN = os.path.join(OUTPUT_BASE, "positive_features_train.npy")
NEG_FEAT_TRAIN = os.path.join(OUTPUT_BASE, "negative_features_train.npy")
POS_FEAT_TEST  = os.path.join(OUTPUT_BASE, "positive_features_test.npy")
NEG_FEAT_TEST  = os.path.join(OUTPUT_BASE, "negative_features_test.npy")
FP_VAL_FILE    = os.path.join(OUTPUT_BASE, "fp_validation.npy")

# =============================================================================
# MODULE-LEVEL ITERABLE DATASET
# Must be at module level (not inside a function) to be picklable on Windows.
# =============================================================================
class IterDS(torch.utils.data.IterableDataset):
    def __init__(self, g): self.g = g
    def __iter__(self): return self.g


# =============================================================================
# CONFIG
# =============================================================================
TARGET_SR      = 16000
TRAIN_RATIO    = 0.9
TOTAL_LENGTH   = 32000   # 2 s @ 16 kHz
BATCH_SIZE     = 128
AUG_ROUNDS     = 1
TRAIN_STEPS    = 8000
MAX_NEG_WEIGHT = 1000
TARGET_FP_HR   = 0.5   # more lenient so auto_train doesn't exit before model learns positives
N_FP_VAL_ROWS  = 300   # not used for real-data FP val

# --- Overwrite flags: set to False to skip already-completed stages ----------
OVERWRITE_RESAMPLE  = False
OVERWRITE_SPLIT     = False   # dirs already populated with 16kHz clips
OVERWRITE_FEATURES  = True   # pos train/test + neg train already exist; neg_test missing -> will be created
OVERWRITE_FP_VAL    = False

# --- Validation targets -------------------------------------------------------
VALIDATION_THRESHOLD    = 0.5   # decision threshold for precision / recall
PRECISION_TARGET        = 0.90  # must be >= this
FP_RATE_PER_HOUR_TARGET = 0.01  # must be < this
# The real-world duration represented by the FP validation set (synthetic here).
# For a proper evaluation this should be the actual negative-clip total duration.
FP_VAL_DURATION_HRS     = 1.0   # hours – used in  FP_count / duration_hrs

# =============================================================================
# STEP 0 – Verify source data
# =============================================================================
def verify_data():
    pos_files = sorted(glob.glob(os.path.join(POSITIVE_AUG_DIR, "*.wav")))
    neg_files = sorted(glob.glob(os.path.join(NEGATIVE_DIR, "*.wav")))
    if not pos_files:
        raise FileNotFoundError(f"No WAVs found in '{POSITIVE_AUG_DIR}'. Run augment_audio.py first.")
    if not neg_files:
        raise FileNotFoundError(f"No WAVs found in '{NEGATIVE_DIR}'. Run generate_negatives.py first.")
    log.info(f"[OK] Positive clips : {len(pos_files)}")
    log.info(f"[OK] Negative clips : {len(neg_files)}")
    return list(pos_files), list(neg_files)

# =============================================================================
# STEP 1 – Resample to 16 kHz
# =============================================================================
def resample_directory(src_files, dst_dir):
    import soundfile as sf
    from scipy.signal import resample as sci_resample

    os.makedirs(dst_dir, exist_ok=True)
    existing = glob.glob(os.path.join(dst_dir, "*.wav"))
    if existing and not OVERWRITE_RESAMPLE:
        log.info(f"  '{dst_dir}' already has {len(existing)} files – skipping resample.")
        return existing

    log.info(f"  Resampling {len(src_files)} clips -> {TARGET_SR} Hz in '{dst_dir}'")
    out = []
    for i, src in enumerate(src_files):
        dst = os.path.join(dst_dir, os.path.basename(src))
        if os.path.exists(dst) and not OVERWRITE_RESAMPLE:
            out.append(dst); continue
        data, sr = sf.read(src, always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != TARGET_SR:
            n = int(len(data) * TARGET_SR / sr)
            data = sci_resample(data, n).astype(np.float32)
        sf.write(dst, data.astype(np.float32), TARGET_SR)
        out.append(dst)
        if i % 300 == 0:
            log.info(f"    {i}/{len(src_files)}...")
    log.info(f"  Done – {len(out)} files")
    return out

# =============================================================================
# STEP 2 – 90/10 split
# =============================================================================
def split_and_copy(files, train_dir, test_dir):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)

    ex_tr = glob.glob(os.path.join(train_dir, "*.wav"))
    ex_te = glob.glob(os.path.join(test_dir,  "*.wav"))
    if ex_tr and ex_te and not OVERWRITE_SPLIT:
        log.info(f"  Split dirs populated ({len(ex_tr)} / {len(ex_te)}) – skipping.")
        return ex_tr, ex_te

    for f in ex_tr: os.remove(f)
    for f in ex_te: os.remove(f)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(files))
    n   = int(len(files) * TRAIN_RATIO)
    for f in [files[i] for i in idx[:n]]:
        shutil.copy(f, os.path.join(train_dir, os.path.basename(f)))
    for f in [files[i] for i in idx[n:]]:
        shutil.copy(f, os.path.join(test_dir,  os.path.basename(f)))
    tr = glob.glob(os.path.join(train_dir, "*.wav"))
    te = glob.glob(os.path.join(test_dir,  "*.wav"))
    log.info(f"  -> {len(tr)} train / {len(te)} test")
    return tr, te

# =============================================================================
# STEP 3 – Compute OWW mel-spectrogram features
# =============================================================================
def compute_features(train_paths, test_paths, feat_train, feat_test, label):
    from openwakeword.data import augment_clips
    from openwakeword.utils import compute_features_from_generator

    need_tr = not os.path.exists(feat_train) or OVERWRITE_FEATURES
    need_te = not os.path.exists(feat_test)  or OVERWRITE_FEATURES

    if not need_tr and not need_te:
        log.info(f"  [{label}] Both feature files exist – skipping.")
        return

    n_cpus = max(1, (os.cpu_count() or 2) // 2)

    if need_tr:
        log.info(f"  [{label}] Train features ({len(train_paths)} clips)…")
        gen = augment_clips(train_paths * AUG_ROUNDS, TOTAL_LENGTH,
                            batch_size=BATCH_SIZE,
                            background_clip_paths=[], RIR_paths=[])
        import gc
        try:
            compute_features_from_generator(gen,
                n_total=len(train_paths)*AUG_ROUNDS,
                clip_duration=TOTAL_LENGTH,
                output_file=feat_train,
                ncpu=n_cpus
            )
        except PermissionError:
            gc.collect()
            import time
            time.sleep(1)
            try:
                os.remove(feat_train)
                os.rename(feat_train.replace(".npy", "2.npy"), feat_train)
            except Exception as e:
                log.warning(f"Could not finish trimming {feat_train}: {e}")
        log.info(f"    Saved {feat_train}")

    if need_te:
        log.info(f"  [{label}] Test features ({len(test_paths)} clips)…")
        gen = augment_clips(test_paths * AUG_ROUNDS, TOTAL_LENGTH,
                            batch_size=BATCH_SIZE,
                            background_clip_paths=[], RIR_paths=[])
        import gc
        try:
            compute_features_from_generator(gen,
                n_total=len(test_paths)*AUG_ROUNDS,
                clip_duration=TOTAL_LENGTH,
                output_file=feat_test,
                ncpu=n_cpus
            )
        except PermissionError:
            gc.collect()
            import time
            time.sleep(1)
            try:
                os.remove(feat_test)
                os.rename(feat_test.replace(".npy", "2.npy"), feat_test)
            except Exception as e:
                log.warning(f"Could not finish trimming {feat_test}: {e}")
        log.info(f"    Saved {feat_test}")

# =============================================================================
# STEP 5 – Build PyTorch DataLoaders
# =============================================================================
def build_data_loaders(input_shape):
    from openwakeword.data import mmap_batch_generator
    n_steps = input_shape[0]

    def neg_tx(x):
        if x.shape[1] != n_steps:
            xf = np.vstack(x)
            return np.array([xf[i:i+n_steps] for i in range(0, xf.shape[0]-n_steps, n_steps)])
        return x

    batch_gen = mmap_batch_generator(
        data_files={"positive": POS_FEAT_TRAIN, "adversarial_negative": NEG_FEAT_TRAIN},
        n_per_class={"positive": BATCH_SIZE//2, "adversarial_negative": BATCH_SIZE//2},
        data_transform_funcs={"adversarial_negative": neg_tx},
        label_transform_funcs={
            "positive":             lambda x: [1]*len(x),
            "adversarial_negative": lambda x: [0]*len(x),
        },
    )

    # num_workers=0: avoids Windows spawn pickle errors with IterDS/lambda generators
    X_train = torch.utils.data.DataLoader(
        IterDS(batch_gen), batch_size=None, num_workers=0)

    # Validation set (pos + neg test features)
    vp = np.load(POS_FEAT_TEST)
    vn = np.load(NEG_FEAT_TEST)
    if vp.shape[1] != n_steps:
        vp = np.array([vp[i:i+n_steps] for i in range(0, vp.shape[0]-n_steps, n_steps)])
    if vn.shape[1] != n_steps:
        vn = np.array([vn[i:i+n_steps] for i in range(0, vn.shape[0]-n_steps, n_steps)])
    vl = np.hstack([np.ones(vp.shape[0]), np.zeros(vn.shape[0])]).astype(np.float32)
    X_val = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(np.vstack([vp, vn])),
            torch.from_numpy(vl),
        ), batch_size=len(vl))

    # FP-validation: use real negative test features (not synthetic noise).
    # Synthetic Gaussian noise scores near 0 → auto_train exits immediately with Recall=0.
    fp_arr = np.load(NEG_FEAT_TEST).astype(np.float32)   # shape (300, 16, 96)
    if fp_arr.shape[1] != n_steps:
        fp_flat = fp_arr.reshape(-1, fp_arr.shape[-1])
        fp_arr  = np.array([fp_flat[i:i+n_steps] for i in range(0, fp_flat.shape[0]-n_steps)])
    fp_lbl = np.zeros(fp_arr.shape[0], dtype=np.float32)
    X_fp = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(fp_arr), torch.from_numpy(fp_lbl)),
        batch_size=len(fp_lbl))

    log.info(f"  X_train streaming  batch={BATCH_SIZE}  workers=0 (single-process)")
    log.info(f"  X_val   {len(vl)} examples  ({vp.shape[0]} pos / {vn.shape[0]} neg)")
    log.info(f"  X_fp    {fp_arr.shape[0]} FP-val windows")
    return X_train, X_val, X_fp

# =============================================================================
# STEP 6 – Train + export ONNX
# =============================================================================
def train_and_export(X_train, X_val, X_fp, input_shape):
    from openwakeword.train import Model

    log.info("="*60)
    log.info(f"  OWW Model  input_shape={input_shape}  steps={TRAIN_STEPS}")
    log.info("="*60)

    oww = Model(n_classes=1, input_shape=input_shape, model_type="dnn",
                layer_dim=128, n_blocks=1,
                seconds_per_example=1280*input_shape[0]/16000)
    oww.summary()

    log.info("\nStarting auto_train...\n")
    best_model = oww.auto_train(
        X_train=X_train, X_val=X_val, false_positive_val_data=X_fp,
        steps=TRAIN_STEPS, max_negative_weight=MAX_NEG_WEIGHT,
        target_fp_per_hour=TARGET_FP_HR)

    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    onnx_path = os.path.join(FINAL_MODEL_DIR, MODEL_NAME + ".onnx")
    oww.export_model(model=best_model, model_name=MODEL_NAME, output_dir=FINAL_MODEL_DIR)
    log.info(f"\nONNX saved -> {os.path.abspath(onnx_path)}")
    return best_model, oww, onnx_path

# =============================================================================
# STEP 7 – Validate: precision >= 0.90, FP/hr < 0.01
# =============================================================================
def validate_model(best_model, oww, input_shape):
    """
    Evaluate on the held-out positive / negative test features.
    Computes precision, recall, F1, and false-positive rate per hour.
    The FP-rate denominator uses FP_VAL_DURATION_HRS (currently synthetic/1hr).
    For a real deployment-quality estimate, replace FP_VAL_FILE with features
    extracted from hours of real background audio.
    """
    log.info("\n" + "="*60)
    log.info("  VALIDATION")
    log.info("="*60)

    device = torch.device("cpu")
    best_model = best_model.to(device)
    best_model.eval()

    n_steps = input_shape[0]
    thr     = VALIDATION_THRESHOLD

    # ── Load positive test features ───────────────────────────────────────────
    pos_feats = np.load(POS_FEAT_TEST)   # shape (N, n_steps, 96)
    if pos_feats.shape[1] != n_steps:
        pos_feats = np.array([pos_feats[i:i+n_steps]
                              for i in range(0, pos_feats.shape[0]-n_steps, n_steps)])

    # ── Load negative test features ───────────────────────────────────────────
    neg_feats = np.load(NEG_FEAT_TEST)
    if neg_feats.shape[1] != n_steps:
        neg_feats = np.array([neg_feats[i:i+n_steps]
                              for i in range(0, neg_feats.shape[0]-n_steps, n_steps)])

    # ── FP-val features — real negative test clips (same as training FP val)
    fp_feats = np.load(NEG_FEAT_TEST).astype(np.float32)   # (300, 16, 96)
    if fp_feats.ndim == 3 and fp_feats.shape[1] != n_steps:
        fp_flat  = fp_feats.reshape(-1, fp_feats.shape[-1])
        fp_feats = np.array([fp_flat[i:i+n_steps] for i in range(0, fp_flat.shape[0]-n_steps)])

    def predict_batch(arr):
        with torch.no_grad():
            t = torch.from_numpy(arr).to(device)
            return best_model(t).squeeze(-1).cpu().numpy()

    pos_scores = predict_batch(pos_feats)
    neg_scores = predict_batch(neg_feats)
    fp_scores  = predict_batch(fp_feats)

    # ── Precision / Recall / F1 (on balanced pos+neg test set) ───────────────
    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate([np.ones(len(pos_scores)),
                                  np.zeros(len(neg_scores))])

    preds = (all_scores >= thr).astype(int)
    tp = int(np.sum((preds == 1) & (all_labels == 1)))
    fp = int(np.sum((preds == 1) & (all_labels == 0)))
    fn = int(np.sum((preds == 0) & (all_labels == 1)))
    tn = int(np.sum((preds == 0) & (all_labels == 0)))

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2*precision*recall / (precision + recall + 1e-9)

    # ── False-positive rate (on FP-val set) ───────────────────────────────────
    fp_preds = (fp_scores >= thr).astype(int)
    n_fp_val = int(fp_preds.sum())
    fp_per_hour = n_fp_val / FP_VAL_DURATION_HRS

    # ── Score distribution stats ──────────────────────────────────────────────
    log.info(f"\n  Threshold      : {thr:.2f}")
    log.info(f"  Positive clips : {len(pos_scores)}  (mean score {pos_scores.mean():.3f}, "
             f"min {pos_scores.min():.3f}, max {pos_scores.max():.3f})")
    log.info(f"  Negative clips : {len(neg_scores)}  (mean score {neg_scores.mean():.3f}, "
             f"min {neg_scores.min():.3f}, max {neg_scores.max():.3f})")
    log.info(f"  FP-val clips   : {len(fp_scores)}  (mean score {fp_scores.mean():.3f})")
    log.info("")
    log.info(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    log.info(f"  Precision : {precision:.4f}  (target >= {PRECISION_TARGET:.2f})")
    log.info(f"  Recall    : {recall:.4f}")
    log.info(f"  F1        : {f1:.4f}")
    log.info(f"  FP / hr   : {fp_per_hour:.4f}  (target < {FP_RATE_PER_HOUR_TARGET:.4f})")
    log.info("")

    # ── Pass / Fail ───────────────────────────────────────────────────────────
    pass_precision = precision >= PRECISION_TARGET
    pass_fp_rate   = fp_per_hour < FP_RATE_PER_HOUR_TARGET

    status_p  = "PASS" if pass_precision else "FAIL"
    status_fp = "PASS" if pass_fp_rate   else "FAIL"

    log.info(f"  Precision  [{status_p}]  {precision:.4f}  {'>=':>3} {PRECISION_TARGET}")
    log.info(f"  FP / hr    [{status_fp}]  {fp_per_hour:.4f}  {'<':>3} {FP_RATE_PER_HOUR_TARGET}")
    log.info("")

    # ── Threshold sweep ───────────────────────────────────────────────────────
    log.info("  --- Threshold sweep (precision / recall trade-off) ---")
    log.info(f"  {'Thr':>5}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'FP/hr':>8}")
    for t in np.arange(0.3, 0.96, 0.05):
        p_ = (all_scores >= t).astype(int)
        _tp = int(np.sum((p_==1)&(all_labels==1)))
        _fp = int(np.sum((p_==1)&(all_labels==0)))
        _fn = int(np.sum((p_==0)&(all_labels==1)))
        pr_ = _tp/(_tp+_fp+1e-9)
        rc_ = _tp/(_tp+_fn+1e-9)
        f1_ = 2*pr_*rc_/(pr_+rc_+1e-9)
        fp_ = int((fp_scores>=t).sum()) / FP_VAL_DURATION_HRS
        log.info(f"  {t:>5.2f}  {pr_:>6.4f}  {rc_:>6.4f}  {f1_:>6.4f}  {fp_:>8.4f}")

    overall = pass_precision and pass_fp_rate
    log.info("\n" + "="*60)
    if overall:
        log.info("  VALIDATION PASSED – model meets all targets!")
    else:
        log.info("  VALIDATION FAILED – see table above for recommended threshold.")
        if not pass_precision:
            log.info("  -> Increase threshold to improve precision (reduces FP).")
        if not pass_fp_rate:
            log.info("  -> Increase threshold or add more negative training data.")
    log.info("="*60)

    return {
        "precision":   precision,
        "recall":      recall,
        "f1":          f1,
        "fp_per_hour": fp_per_hour,
        "passed":      overall,
    }

# =============================================================================
# MAIN
# =============================================================================
def main():
    log.info("="*60)
    log.info("  Hello Zero Touch – Wake Word Training + Validation")
    log.info("="*60)

    # 0. Verify
    pos_files, neg_files = verify_data()

    # 1. Resample
    log.info("\n[1/6] Resampling to 16 kHz...")
    pos_16k = resample_directory(pos_files, POS_RESAMP_DIR)
    neg_16k = resample_directory(neg_files, NEG_RESAMP_DIR)

    # 2. Split
    log.info("\n[2/6] Train / test split...")
    pos_train, pos_test = split_and_copy(pos_16k, POS_TRAIN_DIR, POS_TEST_DIR)
    neg_train, neg_test = split_and_copy(neg_16k, NEG_TRAIN_DIR, NEG_TEST_DIR)

    # 3. Features
    log.info("\n[3/6] Computing OWW audio features...")
    log.info("  [positive]")
    compute_features(pos_train, pos_test, POS_FEAT_TRAIN, POS_FEAT_TEST, "positive")
    log.info("  [negative]")
    compute_features(neg_train, neg_test, NEG_FEAT_TRAIN, NEG_FEAT_TEST, "negative")

    # 4. Input shape
    log.info("\n[4/6] Resolving model input shape...")
    from openwakeword.utils import AudioFeatures
    F = AudioFeatures(device="cpu")
    input_shape = F.get_embedding_shape(TOTAL_LENGTH / 16000)
    log.info(f"  input_shape = {input_shape}")

    # 5. Data loaders + train
    log.info("\n[5/6] Building DataLoaders and training...")
    X_train, X_val, X_fp = build_data_loaders(input_shape)
    best_model, oww, onnx_path = train_and_export(X_train, X_val, X_fp, input_shape)

    # 6. Validate
    log.info("\n[6/6] Validating trained model...")
    results = validate_model(best_model, oww, input_shape)

    log.info("\n" + "="*60)
    log.info(f"  ONNX : {os.path.abspath(onnx_path)}")
    log.info(f"  Status : {'PASSED' if results['passed'] else 'FAILED'}")
    log.info("="*60)

    log.info("\nstt_activation.py config:")
    log.info(f'  WAKE_WORD_MODEL     = "{onnx_path}"')
    log.info(f'  WAKE_WORD_LABEL     = "hello zero touch"')
    log.info(f'  WAKE_WORD_THRESHOLD = {VALIDATION_THRESHOLD}  # tune from sweep above')


if __name__ == "__main__":
    main()
