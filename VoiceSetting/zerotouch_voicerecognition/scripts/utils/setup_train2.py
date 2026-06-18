import os
import yaml
import numpy as np

# Create dummy false positive validation data
dummy_fp = np.random.randn(10000, 96).astype(np.float32)
np.save("dummy_fp.npy", dummy_fp)

# Load existing config
with open("train.yaml", "r") as f:
    config = yaml.safe_load(f)

config.update({
    "model_type": "dnn",
    "layer_size": 128,
    "n_blocks": 1,
    "feature_data_files": {},
    "batch_n_per_class": {
        "positive": 128,
        "adversarial_negative": 128
    },
    "false_positive_validation_data_path": "dummy_fp.npy",
    "steps": 1000,
    "max_negative_weight": 1000,
    "target_false_positives_per_hour": 0.1
})

with open("train.yaml", "w") as f:
    yaml.dump(config, f)

print("train.yaml updated and dummy_fp.npy created.")
