from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
from model import ECG_model
from config import get_config
from utils import *

def train(config, X, y, Xval=None, yval=None):
    classes = ['N','V','/','A','F','~']  # , 'L','R','f','j','E','a'] # added extra classes for example
    Xe = np.expand_dims(X, axis=2)
    
    # Ensure the data is not None
    if Xe is None or y is None:
        raise ValueError("Input data (Xe, y) cannot be None.")
    
    if Xval is not None and yval is not None:
        Xvale = np.expand_dims(Xval, axis=2)
        (m, n) = y.shape
        y = y.reshape((m, 1, n))
        (mvl, nvl) = yval.shape
        yval = yval.reshape((mvl, 1, nvl))
    elif not config.split:
        # Ensure data splitting works and no empty data
        from sklearn.model_selection import train_test_split
        Xe, Xvale, y, yval = train_test_split(Xe, y, test_size=0.2, random_state=1)
    else:
        raise ValueError("Validation data must be provided if config.split is True.")
    
    # Validate that Xvale, yval are not None
    if Xvale is None or yval is None:
        raise ValueError("Validation data (Xvale, yval) cannot be None.")
    
    # Load model from checkpoint or initialize new
    if config.checkpoint_path is not None:
        model = model.load_model(config.checkpoint_path)
        initial_epoch = config.resume_epoch  # put the resuming epoch
    else:
        model = ECG_model(config)
        initial_epoch = 0

    # Ensure the model is compiled and prepared
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Create directory for saving model if it doesn't exist
    mkdir_recursive('models')

    # Define callbacks
    callbacks = [
        EarlyStopping(patience=config.patience, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.01, verbose=1),
        TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True),
        ModelCheckpoint(f'models/{config.feature}-latest.keras', monitor='val_loss', save_best_only=False, verbose=1, save_freq=10)
    ]
    
    # Fit model
    model.fit(Xe, y,
              validation_data=(Xvale, yval),
              epochs=config.epochs,
              batch_size=config.batch,
              callbacks=callbacks,
              initial_epoch=initial_epoch)
    
    # Evaluate the model
    print_results(config, model, Xvale, yval, classes)

def main(config):
    print('feature:', config.feature)
    
    # Ensure the data is loaded correctly
    (X, y, Xval, yval) = loaddata(config.input_size, config.feature)
    
    # Validate that the data is not None
    if X is None or y is None:
        raise ValueError("Training data (X, y) cannot be None.")
    if Xval is None or yval is None:
        raise ValueError("Validation data (Xval, yval) cannot be None.")
    
    # Train the model
    train(config, X, y, Xval, yval)

if __name__ == "__main__":
    config = get_config()
    main(config)
