import csv
import sys

import flwr as fl
import tensorflow as tf
from flwr.common import Context, Metrics
from flwr.server import Driver, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

from config import ExpConfig
from data import network_data_test, network_feature_data_test
from federation import get_evaluate_fn, get_evaluate_fn_v, weighted_average

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)],
        )
    except RuntimeError as e:
        print(e)


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
