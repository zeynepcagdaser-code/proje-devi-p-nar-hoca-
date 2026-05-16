import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import BatchNormalization, Conv1D, Dense, Dropout, Flatten, Input, MaxPooling1D
from tensorflow.keras.models import Sequential


def scale_cnn_full_windows(x_windows):
    scaler = MinMaxScaler()
    feature_count = x_windows.shape[2]
    x_2d = x_windows.reshape(-1, feature_count)
    x_scaled = scaler.fit_transform(x_2d).reshape(x_windows.shape)
    return x_scaled, scaler


def build_cnn_model(window_size, feature_count, class_count, recipe):
    model = Sequential(
        [
            Input(shape=(window_size, feature_count)),
            Conv1D(16, kernel_size=3, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(recipe["dropout1"]),
            Conv1D(32, kernel_size=3, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(recipe["dropout2"]),
            Flatten(),
            Dense(recipe["dense_units"], activation="relu"),
            Dropout(recipe["dropout3"]),
            Dense(class_count, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=recipe["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def cnn_recipes():
    return [
        {
            "name": "stable_balanced",
            "dropout1": 0.10,
            "dropout2": 0.20,
            "dropout3": 0.25,
            "dense_units": 96,
            "learning_rate": 0.0003,
            "use_class_weight": True,
        },
        {
            "name": "high_capacity_low_lr",
            "dropout1": 0.05,
            "dropout2": 0.15,
            "dropout3": 0.20,
            "dense_units": 128,
            "learning_rate": 0.0005,
            "use_class_weight": False,
        },
        {
            "name": "balanced_plus_low_lr",
            "dropout1": 0.10,
            "dropout2": 0.20,
            "dropout3": 0.20,
            "dense_units": 96,
            "learning_rate": 0.0002,
            "use_class_weight": True,
        },
    ]

