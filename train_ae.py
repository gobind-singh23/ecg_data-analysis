import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, Huber, CosineSimilarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from config import get_config
from utils import *
from autoencoder import AutoEncoder


def train(config, X_train, X_test, normal, anomaly):
    #data dimensions
    input_dim = X_train.shape[-1]

    early_stopping = EarlyStopping(patience=10, min_delta=1e-3, monitor="val_loss", restore_best_weights=True)
    loss_functions = {'MAE': MeanAbsoluteError(), 'MSE': MeanSquaredError(), 'Huber': Huber(), 'CosineSimilarity': CosineSimilarity()}
    best_loss = float('inf')
    best_model = None
    best_loss_function = None

    #ground truths
    y_true_normal = np.zeros(len(X_test))
    y_true_anomaly = np.ones(len(anomaly))
    y_test = np.concatenate([y_true_normal, y_true_anomaly])
    X_combined_test = np.concatenate([X_test, anomaly])
    for name, loss_function in loss_functions.items():
    # Define and compile the model
        model = AutoEncoder(input_dim, config.latent_dim)
        model.build((None, input_dim))
        # Check if the machine is using an M1/M2 processor
        if "macOS-13.5.2-arm64-arm-64bit" in platform.platform() and "arm64" in platform.machine():
            model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.01), loss=loss_function)
            print("Using legacy optimizer for M1/M2 Mac.")
        elif tf.config.list_physical_devices('GPU'):  # Check if there are any GPUs available
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=loss_function)
            print("Using standard optimizer for GPU.")
        else:
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=loss_function)
            print("Using standard optimizer for conventional CPU.")

    model.summary()
    # Train the model
    model.fit(X_train, X_train, epochs=config.epochs, batch_size=64,
              validation_split=0.1, callbacks=[early_stopping], verbose=0)

    # Calculate the threshold based on the training data
    train_errors = get_reconstruction_error(model, X_train)
    threshold = np.percentile(train_errors, 95)  # 95th percentile as threshold

    # Evaluate the model on combined validation data (normal + anomaly) using the calculated threshold
    val_errors = get_reconstruction_error(model, X_combined_test)
    y_pred_val = np.array(val_errors > threshold, dtype=int)

    accuracy = accuracy_score(y_test, y_pred_val)

    print(f"Validation Accuracy using {name}: {accuracy:.2%}")
    print(f"Validation Reconstruction Error using {name}: {np.mean(val_errors)}")
    if np.mean(val_errors) < best_loss:
        best_loss = np.mean(val_errors)
        best_model = model
        best_loss_function = name

    print(f"Best Model uses {best_loss_function} with average validation error: {best_loss}")

def main(config):
    normal_df = pd.read_csv("./arrhythmia-dataset/ptbdb_normal.csv", sep=',').drop("target", axis=1, errors="ignore")
    anomaly_df = pd.read_csv("./arrhythmia-dataset/ptbdb_abnormal.csv", sep=',').drop("target", axis=1, errors="ignore")

    normal = normal_df.to_numpy()
    anomaly = anomaly_df.to_numpy()

    X_train, X_test = train_test_split(normal, test_size=0.15, random_state=45, shuffle=True)   

    train(config, X_train, X_test, normal, anomaly)


if __name__ == "__main__":
    config = get_config()
    main(config)
