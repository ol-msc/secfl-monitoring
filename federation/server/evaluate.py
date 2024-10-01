import flwr as fl

from config import ExpConfig
from federation.model import build_gru_model


def get_evaluate_fn(testset, model_name=ExpConfig.MODEL_NAME):
    """Return an evaluation function for server-side (i.e. centralized) evaluation."""
    x_test, y_test = testset

    # The `evaluate` function will be called after every round by the strategy
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: dict[str, fl.common.Scalar],
    ):

        model = build_gru_model(
            model_name, ExpConfig.MULTIVARIATE, ExpConfig.SEQUENCE_LEN
        )  # Construct the model
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, mse, mea = model.evaluate(x_test, y_test, verbose=0)
        if server_round == ExpConfig.FL_SAVE_ON_ROUND:
            print(f"[SERVER] saving model to {ExpConfig.MODEL_SAVE_PATH}")
            model.save_weights(ExpConfig.MODEL_SAVE_PATH)
        return loss, {"mean_squared_error": mse, "mean_absolute_error": mea}

    return evaluate


def get_evaluate_fn_v(testset, model_name=ExpConfig.MODEL_NAME):
    """Return an evaluation function for server-side (i.e. centralized) evaluation."""
    http_x_test, http_y_test = testset[0]
    ssl_x_test, ssl_y_test = testset[1]

    # The `evaluate` function will be called after every round by the strategy
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: dict[str, fl.common.Scalar],
    ):

        model = build_gru_model(
            model_name, ExpConfig.MULTIVARIATE, ExpConfig.SEQUENCE_LEN
        )  # Construct the model
        model.set_weights(parameters)  # Update model with the latest parameters
        http_loss, http_mse, http_mea = model.evaluate(
            http_x_test, http_y_test, verbose=0
        )
        ssl_loss, ssl_mse, ssl_mea = model.evaluate(ssl_x_test, ssl_y_test, verbose=0)

        loss = (http_loss + ssl_loss) / 2
        mse = (http_mse + ssl_mse) / 2
        mea = (http_mea + ssl_mea) / 2
        if server_round == ExpConfig.FL_SAVE_ON_ROUND:
            print(f"[SERVER] saving model to {ExpConfig.MODEL_SAVE_PATH}")
            model.save_weights(ExpConfig.MODEL_SAVE_PATH)
        return loss, {"mean_squared_error": mse, "mean_absolute_error": mea}

    return evaluate
