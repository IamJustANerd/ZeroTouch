import os
import shutil
import glob
import yaml

# 1. Rename user's generate_samples.py to avoid import execution conflicts
if os.path.exists("generate_samples.py") and not "def generate_samples" in open("generate_samples.py").read():
    os.rename("generate_samples.py", "generate_tts_samples.py")
    with open("generate_samples.py", "w") as f:
        f.write("def generate_samples(*args, **kwargs):\n    pass\n")

# 2. Setup directories for train.py
output_base = "models/hello_zerotouch"
dirs = ["positive_train", "positive_test", "negative_train", "negative_test"]
for d in dirs:
    os.makedirs(os.path.join(output_base, d), exist_ok=True)

# 3. Copy positive clips to train/test
pos_files = glob.glob("training_data/positive_augmented/*.wav")
for i, f in enumerate(pos_files):
    if i % 10 == 0:
        shutil.copy(f, os.path.join(output_base, "positive_test", os.path.basename(f)))
    else:
        shutil.copy(f, os.path.join(output_base, "positive_train", os.path.basename(f)))

# 4. Copy negative clips to train/test
neg_files = glob.glob("training_data/negative/*.wav")
for i, f in enumerate(neg_files):
    if i % 10 == 0:
        shutil.copy(f, os.path.join(output_base, "negative_test", os.path.basename(f)))
    else:
        shutil.copy(f, os.path.join(output_base, "negative_train", os.path.basename(f)))

# 5. Create train.yaml config
config = {
    "output_dir": "models",
    "model_name": "hello_zerotouch",
    "target_phrase": ["hello zero touch"],
    "n_samples": max(len(pos_files), 100),
    "n_samples_val": max(len(pos_files) // 10, 10),
    "tts_batch_size": 10,
    "custom_negative_phrases": [],
    "rir_paths": [],
    "background_paths": [],
    "background_paths_duplication_rate": [],
    "augmentation_rounds": 1,
    "augmentation_batch_size": 128,
    "piper_sample_generator_path": "."
}

with open("train.yaml", "w") as f:
    yaml.dump(config, f)

print(f"Setup complete. Found {len(pos_files)} positive files and {len(neg_files)} negative files.")
