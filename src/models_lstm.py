import tensorflow as tf
from tensorflow.keras import layers, models

def build_lstm_model(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)

    x = layers.LSTM(128, return_sequences=True)(inp)
    x = layers.Dropout(0.3)(x)

    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inp, out)
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model
