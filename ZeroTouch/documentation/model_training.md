# Model Training - ZeroTouch

This document describes the process of training the Static Gesture classifier using the collected landmark dataset.

## Model Architecture (MLP)

We use a **Multi-Layer Perceptron (MLP)** designed for low-latency inference on edge devices.

- **Input Layer**: 63 neurons (21 landmarks × 3 coordinates: x, y, z).
- **Hidden Layers**:
  - Dense (32, ReLU) + Dropout (0.2)
  - Dense (16, ReLU)
  - Dense (8, ReLU)
- **Output Layer**: 4 neurons (Softmax) representing our classes.

## Class Mapping

| ID    | Gesture       | Description                        |
| :---- | :------------ | :--------------------------------- |
| **0** | **Open Palm** | Idle / Awake State (The "Clutch"). |
| **1** | **Fist**      | "Grab" for manipulation.           |
| **2** | **Pinch**     | Selection/Clicking action.         |
| **3** | **Pointing**  | Standard Cursor Navigation.        |

## Training Workflow

1. **Data Loading**: Reads `gesture_data.csv` and splits it into 80% Training and 20% Testing.
2. **Optimization**: Uses the `Adam` optimizer with `sparse_categorical_crossentropy` loss.
3. **Conversion**: The final Keras model is converted to **TensorFlow Lite (.tflite)** to ensure high FPS during real-time tracking.

## How to Train

Run the training script in your virtual environment:

```powershell
python train_model.py
```

## Performance Targets

- **Accuracy**: >95% on test data.
- **Inference Time**: <5ms per frame (on CPU).
