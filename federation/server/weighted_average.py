from flwr.common import Metrics

from .utils import write_round_metrics_to_file


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
