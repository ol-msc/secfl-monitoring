import os
import sys
from datetime import datetime
from pathlib import Path

from data import (
    network_data_train,
    network_feature_data_train,
    split_art_data_iid,
    split_art_data_noniid,
)

PARTITION_TYPES = ["noniid", "iid", "vertical"]
AGGREGATION_TYPES = ["regular", "secure"]


class DefaultConfig:
    PARTITION_TYPE = "noniid"
    AGGREGATION_TYPE = "regular"
    N_CLIENTS = 3


class Config:
    SAVE_DIR = Path(__file__).parent.parent.joinpath("saves")

    SEQUENCE_LEN = 24
    FL_ROUNDS = 10
    FL_SAVE_ON_ROUND = FL_ROUNDS

    def __init__(self):
        self.__set_main_settings()
        self.__set_model_settings()
        self.__set_partition_dependent_settings()

        if self.FL_AGGREGATION_TYPE == "secure":
            self.__set_secure_agg_settings()

        self.experiment_name = self.__get_experiment_name()

    def __set_main_settings(self):
        p_type = os.getenv("PARTITION_TYPE")
        self.PARTITION_TYPE = (
            p_type if p_type in PARTITION_TYPES else DefaultConfig.PARTITION_TYPE
        )

        agg_type = os.getenv("AGGREGATION_TYPE")
        self.FL_AGGREGATION_TYPE = (
            agg_type
            if agg_type in AGGREGATION_TYPES
            else DefaultConfig.AGGREGATION_TYPE
        )

        n_clients = os.getenv("N_CLIENTS")
        self.FL_N_CLIENTS = int(n_clients) if n_clients else DefaultConfig.N_CLIENTS

    def __set_secure_agg_settings(self):
        self.AGG_N_SHARES = self.FL_N_CLIENTS if self.FL_N_CLIENTS >= 3 else 3
        self.AGG_REC_SHARES = self.AGG_N_SHARES - 1

    def __set_model_settings(self):
        exp_name = self.__get_experiment_name()
        self.experiment_name = exp_name
        self.MODEL_NAME = exp_name
        self.MODEL_SAVE_PATH = self.SAVE_DIR.joinpath(f"{exp_name}.weights.h5")
        self.MODEL_HISTORY_SAVE_PATH = self.SAVE_DIR.joinpath(f"{exp_name}.history.pkl")
        self.METRICS_SAVE_PATH = self.SAVE_DIR.joinpath(f"{exp_name}.lclmetrics.csv")

    def __set_partition_dependent_settings(self):
        match self.PARTITION_TYPE:
            case "noniid":
                self.MULTIVARIATE = 2
                self.PARTITIONS = split_art_data_noniid(
                    self.FL_N_CLIENTS, train_data=network_data_train
                )
            case "iid":
                self.MULTIVARIATE = 2
                self.PARTITIONS = split_art_data_iid(
                    self.FL_N_CLIENTS, train_data=network_data_train
                )
            case "vertical":
                self.MULTIVARIATE = 1
                self.PARTITIONS = network_feature_data_train
            case _:
                self.MULTIVARIATE = ""
                print("[CONFIG] No Partition Type selected, exiting...")
                sys.exit()

    def __get_experiment_name(self):
        name = f"fl_{self.FL_AGGREGATION_TYPE}_{self.PARTITION_TYPE}_clients_{self.FL_N_CLIENTS}_rounds_{self.FL_ROUNDS}"
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        return f"{name}_{timestamp}"


ExpConfig = Config()
