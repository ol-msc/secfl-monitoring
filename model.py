from keras import layers
from keras.models import Model
from keras.optimizers import Adam

cfg_sequence_len = 24
cfg_learning_rate = 0.001


def build_gru_model(
    model_name: str,
    multivariate: int,
    sequence_len: int,
    sequence_len_y: int = 1,
    learning_rate: float = cfg_learning_rate,
):
    inputs = layers.Input(shape=(sequence_len, multivariate))
    h = layers.GRU(units=16, return_sequences=True)(inputs)
    h = layers.Dropout(0.1)(h)
    h = layers.GRU(units=80, return_sequences=True)(h)
    h = layers.Dropout(0.05)(h)
    h = layers.GRU(units=40, return_sequences=True)(h)
    h = layers.Dropout(0.25)(h)
    h = layers.GRU(units=96, return_sequences=False)(h)
    h = layers.Dropout(0.2)(h)
    x = layers.Input(shape=(cfg_sequence_len, multivariate))

    outputs = layers.Dense(units=multivariate * sequence_len_y, activation="sigmoid")(h)

    model = Model(inputs=inputs, outputs=outputs, name=model_name)

    opt = Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(loss="mean_squared_error", optimizer=opt, metrics=["mse", "mae"])

    return model
