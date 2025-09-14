# src/train.py
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU
from tensorflow.keras.callbacks import ModelCheckpoint

def build_model(input_shape, gru_units=64):
    """Build a small stacked GRU model for regression."""
    model = Sequential()
    model.add(GRU(gru_units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(GRU(gru_units))
    model.add(Dropout(0.2))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(X, y, model_path="models/gru_model.h5", epochs=10, batch_size=32, verbose=1):
    """
    Train model on X,y (X shaped [samples, time_step, 1]) and save best model to model_path.
    Returns trained model (the best checkpoint is saved, and returned model is final).
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    input_shape = (X.shape[1], 1)
    model = build_model(input_shape)
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[checkpoint], verbose=verbose)
    return model
