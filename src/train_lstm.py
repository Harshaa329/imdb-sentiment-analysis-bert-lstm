"""
Train LSTM Deep Learning model for IMDb sentiment classification.

Pipeline:
1. Prepare LSTM input tensors from padded token sequences
2. Build the LSTM model architecture
3. Train with EarlyStopping + ModelCheckpoint
4. Return trained model and training history
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# -------------------------------------------------------
# 1. Prepare LSTM Input Tensors
# -------------------------------------------------------

def prepare_lstm_tensors(df, X_train, X_test, y_train, y_test):
    """
    Convert stored padded sequences (df['lstm_tokens']) into
    aligned numpy arrays for LSTM model training.

    Returns:
        X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm
    """

    # Indices ensure correct sample alignment
    train_idx = X_train.index
    test_idx = X_test.index

    # Stack padded sequences
    X_train_lstm = np.vstack(df.loc[train_idx, 'lstm_tokens'].values)
    X_test_lstm = np.vstack(df.loc[test_idx, 'lstm_tokens'].values)

    # Labels
    y_train_lstm = y_train.values
    y_test_lstm = y_test.values

    print("X_train_lstm shape:", X_train_lstm.shape)
    print("X_test_lstm shape :", X_test_lstm.shape)
    print("y_train_lstm shape:", y_train_lstm.shape)
    print("y_test_lstm shape :", y_test_lstm.shape)

    return X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm


# -------------------------------------------------------
# 2. Build the LSTM Model
# -------------------------------------------------------

def build_lstm_model(vocab_size=5000, embedding_dim=64, max_len=50, lstm_units=128):
    """
    Builds a complete LSTM architecture for text classification.

    Returns:
        compiled Keras LSTM model
    """

    model = Sequential([
        tf.keras.Input(shape=(max_len,)),
        Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        LSTM(lstm_units),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model


# -------------------------------------------------------
# 3. Train LSTM with Callbacks
# -------------------------------------------------------

def train_lstm_model(model, X_train_lstm, y_train_lstm, epochs=12, batch_size=64):
    """
    Trains the LSTM model using EarlyStopping + Checkpoint.

    Returns:
        trained model, training history
    """

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        "best_lstm_model.h5",
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        X_train_lstm,
        y_train_lstm,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, checkpoint],
        verbose=2
    )

    return model, history


# -------------------------------------------------------
# 4. Full Pipeline Wrapper
# -------------------------------------------------------

def run_lstm_training(df, X_train, X_test, y_train, y_test):
    """
    Complete training wrapper:
    - prepares tensors
    - builds model
    - trains model

    Returns:
        model, history, X_test_lstm, y_test_lstm
    """

    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = prepare_lstm_tensors(
        df, X_train, X_test, y_train, y_test
    )

    max_len = X_train_lstm.shape[1]

    model = build_lstm_model(
        vocab_size=5000,
        embedding_dim=64,
        max_len=max_len,
        lstm_units=128
    )

    model, history = train_lstm_model(
        model, X_train_lstm, y_train_lstm
    )

    return model, history, X_test_lstm, y_test_lstm
