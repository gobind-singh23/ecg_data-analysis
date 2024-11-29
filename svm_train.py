from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
from config import get_config
from utils import *
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report

def add_noise(data, noise_factor=0.05):
    """Add random noise to the data."""
    return data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape)

def train_and_test_svm(config, X, y, Xval=None, yval=None, kernel="linear", C=1, add_noise_flag=False):
    """
    Train and test the SVM model with specific kernel and parameters.
    - kernel: Kernel type for SVM (e.g., 'linear', 'rbf', 'poly').
    - C: Regularization parameter.
    - add_noise_flag: Whether to add noise to the input data.
    """
    classes = ['N', 'V', '/', 'A', 'F', '~']  # Class labels
    
    # Expand dims to make compatible for SVM
    Xe = np.expand_dims(X, axis=2)
    if Xval is not None and yval is not None:
        Xvale = np.expand_dims(Xval, axis=2)
    elif not config.split:
        Xe, Xvale, y, yval = train_test_split(Xe, y, test_size=0.2, random_state=1)
    else:
        raise ValueError("Validation data must be provided if config.split is True.")

    # Flatten the input data for SVM (SVM expects 2D input)
    Xe_flat = Xe.reshape(Xe.shape[0], -1)
    Xvale_flat = Xvale.reshape(Xvale.shape[0], -1)

    # Add noise if required
    if add_noise_flag:
        Xe_flat = add_noise(Xe_flat)
        Xvale_flat = add_noise(Xvale_flat)

    # Initialize the SVM model
    svm = SVC(kernel=kernel, C=C, class_weight="balanced")

    # Train the SVM
    print(f"Training SVM with kernel={kernel}, C={C}, Noise={add_noise_flag}")
    svm.fit(Xe_flat, np.argmax(y, axis=-1))

    # Evaluate the model
    accuracy = svm.score(Xvale_flat, np.argmax(yval, axis=-1))
    print(f"SVM Validation Accuracy: {accuracy}")

    # Generate classification report and confusion matrix
    y_pred = svm.predict(Xvale_flat)
    print("Classification Report:")
    print(classification_report(np.argmax(yval, axis=-1), y_pred, target_names=classes))
    print("Confusion Matrix:")
    print(confusion_matrix(np.argmax(yval, axis=-1), y_pred))

    return svm, accuracy

def cross_validation_experiment(X, y, kernel="linear", C=1):
    """Perform cross-validation with SVM."""
    Xe_flat = X.reshape(X.shape[0], -1)  # Flatten input data
    y_flat = np.argmax(y, axis=-1)  # Flatten labels
    svm = SVC(kernel=kernel, C=C, class_weight="balanced")
    scores = cross_val_score(svm, Xe_flat, y_flat, cv=5)  # 5-fold cross-validation
    print(f"Cross-Validation Scores (kernel={kernel}, C={C}): {scores}")
    print(f"Average Accuracy: {np.mean(scores)}")

def main(config):
    print(f"Feature: {config.feature}")
    
    # Load data
    (X, y, Xval, yval) = loaddata(config.input_size, config.feature)

    # Experiments
    experiments = [
        {"kernel": "linear", "C": 0.1, "add_noise_flag": False},
        {"kernel": "linear", "C": 1, "add_noise_flag": False},
        {"kernel": "linear", "C": 10, "add_noise_flag": True},
        {"kernel": "rbf", "C": 1, "add_noise_flag": False},
        {"kernel": "poly", "C": 1, "add_noise_flag": False}
    ]

    for exp in experiments:
        train_and_test_svm(config, X, y, Xval, yval, **exp)

    # Cross-validation
    print("\nPerforming Cross-Validation...")
    cross_validation_experiment(X, y, kernel="linear", C=1)

if __name__ == "__main__":
    config = get_config()
    main(config)

