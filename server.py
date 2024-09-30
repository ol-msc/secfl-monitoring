import csv
import sys

import flwr as fl
import tensorflow as tf
from config import ExpConfig
from flwr.common import Context, Metrics
from flwr.server import Driver, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow
from model import build_gru_model

from data import network_data_test, network_feature_data_test

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)],
        )
    except RuntimeError as e:
        print(e)

round = 0


def write_header(writer) -> None:
    n_clients = [f"CLIENT_{i}" for i in range(ExpConfig.FL_N_CLIENTS)]
    row = ["ROUND", "METRIC"] + n_clients
    writer.writerow(row)


def write_round_metrics_to_file(metrics: tuple) -> None:
    global round
    mses, maes = metrics
    filename = ExpConfig.METRICS_SAVE_PATH

    with open(filename, mode="a+") as file:
        writer = csv.writer(file)
        if round == 0:
            write_header(writer)
        writer.writerows(
            (
                [round, "mse"] + mses,
                [round, "mae"] + maes,
            )
        )
    round += 1


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    # Multiply accuracy of each client by number of examples used

    mses_w = [num_examples * m["mean_squared_error"] for num_examples, m in metrics]
    maes_w = [num_examples * m["mean_absolute_error"] for num_examples, m in metrics]
    mses = [m["mean_squared_error"] for _, m in metrics]
    maes = [m["mean_absolute_error"] for _, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    write_round_metrics_to_file(metrics=(mses, maes))

    # Aggregate and return custom metric (weighted average)
    return {
        "aggregated_mse": sum(mses_w) / sum(examples),
        "aggregated_mea": sum(maes_w) / sum(examples),
    }


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


evaluate_fn = None
match ExpConfig.PARTITION_TYPE:
    case "noniid":
        evaluate_fn = get_evaluate_fn(network_data_test)
    case "iid":
        evaluate_fn = get_evaluate_fn(network_data_test)
    case "vertical":
        evaluate_fn = get_evaluate_fn_v(network_feature_data_test)
    case _:
        print(f"[SERVER] No Partition Type selected, exiting...")
        sys.exit()


# Define strategy
strategy = FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average,
    evaluate_fn=evaluate_fn,
)

# Flower ServerApp
app = ServerApp()


@app.main()
def main(driver: Driver, context: Context) -> None:

    # Construct the LegacyContext
    context = LegacyContext(
        state=context.state,
        config=ServerConfig(num_rounds=ExpConfig.FL_ROUNDS),
        strategy=strategy,
    )

    match ExpConfig.FL_AGGREGATION_TYPE:
        case "secure":
            print(f"[SERVER] Using Secure Aggregation Workflow")
            fit_workflow = SecAggPlusWorkflow(
                num_shares=ExpConfig.AGG_N_SHARES,
                reconstruction_threshold=ExpConfig.AGG_REC_SHARES,
                timeout=40,
            )
            workflow = DefaultWorkflow(fit_workflow)
        case "regular":
            print(f"[SERVER] Using Non-Secure Aggregation Workflow")
            workflow = DefaultWorkflow()

    # Execute
    workflow(driver, context, ExpConfig.MODEL_HISTORY_SAVE_PATH)
