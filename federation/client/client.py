from flwr.client import NumPyClient

from config import ExpConfig
from federation.model import build_gru_model


class FlwrClient(NumPyClient):
    sequence_len = ExpConfig.SEQUENCE_LEN
    multivariate = ExpConfig.MULTIVARIATE

    def __init__(self, x_train, y_train, x_test, y_test):
        import tensorflow as tf

        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [
                        tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=1024
                        )
                    ],
                )
            except RuntimeError as e:
                print(e)

        self.model = build_gru_model("gru_fl", self.multivariate, self.sequence_len)
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_test, y_test

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=1, verbose=0)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, mse, mae = self.model.evaluate(self.x_val, self.y_val, verbose=0)
        return (
            loss,
            len(self.x_val),
            {"mean_squared_error": mse, "mean_absolute_error": mae},
        )
