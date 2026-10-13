"""A small TensorFlow/Keras text classifier, trained for comparison with the TF-IDF baseline."""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf


def build(train_texts, max_tokens: int = 8000, seq_len: int = 60):
    vec = tf.keras.layers.TextVectorization(max_tokens=max_tokens, output_sequence_length=seq_len)
    vec.adapt(tf.constant(list(train_texts)))
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(), dtype=tf.string),
        vec,
        tf.keras.layers.Embedding(max_tokens, 32),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-3), loss="binary_crossentropy")
    return model


def fit_predict(train_texts, y_train, test_texts, seed: int = 42, epochs: int = 12) -> np.ndarray:
    tf.keras.utils.set_random_seed(seed)
    model = build(train_texts)
    weights = {0: 1.0, 1: float((np.asarray(y_train) == 0).sum() / (np.asarray(y_train) == 1).sum())}
    model.fit(tf.constant(list(train_texts)), np.asarray(y_train), epochs=epochs, batch_size=32,
              validation_split=0.1, class_weight=weights, verbose=0)
    return model.predict(tf.constant(list(test_texts)), verbose=0).ravel()
