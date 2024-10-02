import csv

from config import ExpConfig

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
