import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import os

# --- Configuration ---
DATASET_PATH = 'gesture_data.csv'
MODEL_SAVE_PATH = 'gesture_classifier.keras'
TFLITE_SAVE_PATH = 'gesture_classifier.tflite'
CLASSES = 4 # 0: Open Palm, 1: Fist, 2: Pinch, 3: Pointing

def load_data(file_path):
    """
    Loads normalized landmark data from CSV.
    """
    x_dataset = []
    y_dataset = []
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None, None

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            # Label is the first column
            y_dataset.append(int(row[0]))
            # Landmarks are the remaining 63 values (21 * 3)
            x_dataset.append([float(val) for val in row[1:]])
            
    return np.array(x_dataset), np.array(y_dataset)

def build_model(input_shape, num_classes):
    """
    Builds a lightweight MLP for gesture classification.
    """
    model = models.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2), # Prevent overfitting
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    print("--- Loading Dataset ---")
    X, y = load_data(DATASET_PATH)
    if X is None: return

    print(f"Loaded {len(X)} samples.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("--- Training MLP Model ---")
    model = build_model(X.shape[1], CLASSES)
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=1
    )

    print("\n--- Model Evaluation ---")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc*100:.2f}%")

    # Save Keras Model
    model.save(MODEL_SAVE_PATH)
    print(f"Saved Keras model to {MODEL_SAVE_PATH}")

    # --- Convert to TFLite ---
    print("--- Converting to TFLite ---")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open(TFLITE_SAVE_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f"Saved TFLite model to {TFLITE_SAVE_PATH}")

if __name__ == "__main__":
    main()
