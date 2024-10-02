import os

from flwr.client import Client

from config import ExpConfig

from .client import FlwrClient


def client_fn(cid: str) -> Client:
    CID = os.getenv("CID")

    x_train, y_train = ExpConfig.PARTITIONS[int(CID)]
    split_idx = int(len(x_train) * 0.9)
    x_train_cid, y_train_cid = (
        x_train[:split_idx],
        y_train[:split_idx],
    )
    x_val_cid, y_val_cid = x_train[split_idx:], y_train[split_idx:]

    one_client = FlwrClient(x_train_cid, y_train_cid, x_val_cid, y_val_cid)
    return one_client.to_client()
