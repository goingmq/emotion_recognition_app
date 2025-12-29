import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_lstm_model(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inp)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)

    # flatten time dimension to feed into LSTM
    x = layers.Reshape((-1, x.shape[-1]))(x)

    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(64)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inp, out)
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    return model
