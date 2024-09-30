from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import STL
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

CURREN_DIR = Path(__file__).parent
FILE_PATH = CURREN_DIR.joinpath("buffer-20210415-20210526.tsv")


CFG_DATA_SPLIT = 0.8
CFG_SEQUENCE_LEN = 24
CFG_SAMPLING_RATE = 1
CFG_STRIDE = 1
CFG_BATCH_SIZE = 1


# data is numpy array
def transform(data, epsilon=1, remove_peak=False):
    if remove_peak:
        # InterQuartile Range (IQR)
        q_min, q_max = np.percentile(data, [25, 75], axis=0)
        iqr = q_max - q_min
        iqr_min = q_min - 1.5 * iqr
        iqr_max = q_max + 1.5 * iqr
        data = np.clip(data, a_min=iqr_min, a_max=iqr_max)
    data = np.where(data < 0, epsilon, data)
    return data


# Scale all metrics but each separately: normalization or standardization
def normalize(data, scaler=None):
    if len(data.shape) == 1 and isinstance(data, np.ndarray):  # single feature
        data = data.reshape(-1, 1)
    if not scaler:
        scaler = MinMaxScaler(feature_range=(0, 1))
        norm_data = scaler.fit_transform(data)
        norm_data = np.where(norm_data <= 0, 1e-3, norm_data)
    else:
        norm_data = scaler.transform(data)
    return norm_data, scaler


def get_stl_decomposed(
    data: pd.DataFrame,
    period: int,
    resample="10min",
):
    stled = lambda x: {"trend": x.trend, "seasonal": x.seasonal, "resid": x.resid}
    features_decompose = {}

    for i, col_name in enumerate(data.columns):
        # single_feature = data[col_name].resample(resample).mean().ffill()
        try:
            res = STL(data[col_name], period=period).fit()
        except ValueError as e:
            print(f"Unable to STL {col_name} feature, e: {e}")
        else:
            features_decompose[col_name] = stled(res)
    return features_decompose


def get_stl_one(ftrs_dec, col_name, partition_type: Literal["iid", "vert"] = "iid"):
    if col_name == "http_count_uid_in" and partition_type == "iid":
        df = (
            ftrs_dec[col_name]["trend"]
            + ftrs_dec[col_name]["seasonal"]
            + ftrs_dec[col_name]["resid"]
        )
    elif col_name == "ssl_count_uid_in" or partition_type == "vert":
        df = ftrs_dec[col_name]["trend"] + ftrs_dec[col_name]["seasonal"]
    return df


def get_stl(ftrs_dec, protocols, partition_type: Literal["iid", "vert"] = "iid"):
    df = pd.DataFrame()
    for col_name in protocols:
        dc = get_stl_one(ftrs_dec, col_name, partition_type)
        df = pd.concat([df, dc], axis=1)
    df.columns = protocols
    return df


def tsg_to_dataset(tsg: TimeseriesGenerator) -> np.array:
    tsg_len = len(tsg)
    one_entry = tsg[0]  # this raises Warning about __getitem__ deprication.
    x_shape = one_entry[0][0].shape
    y_shape = one_entry[1][0].shape
    x, y = [], []
    for sequence, predicted_val in tsg:
        x.append(sequence)
        y.append(predicted_val)
    x = np.array(x).reshape((tsg_len, *x_shape))
    y = np.array(y).reshape((tsg_len, *y_shape))
    return x, y


# read data from file
data_df = pd.read_csv(FILE_PATH, sep="\t")
data_df = data_df.set_index("ts")
data_df.index = pd.to_datetime(data_df.index)
# select data
cols = ["http_count_uid_in", "ssl_count_uid_in"]
data_df = data_df[cols]

# interpolate missing data
data_df = data_df.interpolate(method="time")

# train/test split
rows = len(data_df)
train_size = int(rows * CFG_DATA_SPLIT)

data_train, data_test = data_df[:train_size], data_df[train_size:]

# convert to ndarray
data_train = data_train.values.astype("float32")
data_test = data_test.values.astype("float32")

# normalized XY train data
data_train_n = transform(data_train)
data_train, scaler = normalize(data_train_n)

# normalize XY test data
trans_test = transform(data_test)
data_test, _ = normalize(trans_test, scaler)

# dataframe
data_train_df = pd.DataFrame(
    data_train, columns=data_df.columns, index=data_df.index[:train_size]
)
data_test_df = pd.DataFrame(
    data_test, columns=data_df.columns, index=data_df.index[train_size:]
)

# This contains decomposition for each feature in the dict format
ftrs_decomposed_train = get_stl_decomposed(data_train_df, period=CFG_SEQUENCE_LEN)
ftrs_decomposed_test = get_stl_decomposed(data_test_df, period=CFG_SEQUENCE_LEN)


# Horizontal FL decomposed data
df_train = get_stl(ftrs_decomposed_train, cols, "iid")
np_train = df_train.to_numpy()

df_test = get_stl(ftrs_decomposed_test, cols, "iid")
np_test = df_test.to_numpy()


# creating time-series data
tsg_params = {
    "length": CFG_SEQUENCE_LEN,
    "sampling_rate": CFG_SAMPLING_RATE,
    "stride": CFG_STRIDE,
    "batch_size": CFG_BATCH_SIZE,
}
tsg_train = TimeseriesGenerator(np_train, np_train, **tsg_params)
tsg_test = TimeseriesGenerator(np_test, np_test, **tsg_params)

# final data
data_train_x, data_train_y = tsg_to_dataset(tsg_train)
data_test_x, data_test_y = tsg_to_dataset(tsg_test)

# this variables for export
network_data_train = (data_train_x, data_train_y)
network_data_test = (data_test_x, data_test_y)


# Vertical FL data decomposed
df_train_v = get_stl(ftrs_decomposed_train, cols, "vert")
np_train_v = df_train_v.to_numpy()

df_test_v = get_stl(ftrs_decomposed_test, cols, "vert")
np_test = df_test_v.to_numpy()

dec_train_http, dec_np_train_http = (
    df_train_v["http_count_uid_in"],
    df_train_v["http_count_uid_in"].to_numpy(),
)
dec_test_http, dec_np_test_http = (
    df_test_v["http_count_uid_in"],
    df_test_v["http_count_uid_in"].to_numpy(),
)
dec_train_ssl, dec_np_train_ssl = (
    df_train_v["ssl_count_uid_in"],
    df_train_v["ssl_count_uid_in"].to_numpy(),
)
dec_test_ssl, dec_np_test_ssl = (
    df_test_v["ssl_count_uid_in"],
    df_test_v["ssl_count_uid_in"].to_numpy(),
)

tsg_http_train = TimeseriesGenerator(dec_train_http, dec_train_http, **tsg_params)
tsg_http_test = TimeseriesGenerator(dec_test_http, dec_test_http, **tsg_params)

tsg_ssl_train = TimeseriesGenerator(dec_train_ssl, dec_train_ssl, **tsg_params)
tsg_ssl_test = TimeseriesGenerator(dec_test_ssl, dec_test_ssl, **tsg_params)

data_train_http_x, data_train_http_y = tsg_to_dataset(tsg_http_train)
data_test_http_x, data_test_http_y = tsg_to_dataset(tsg_http_test)

data_train_ssl_x, data_train_ssl_y = tsg_to_dataset(tsg_ssl_train)
data_test_ssl_x, data_test_ssl_y = tsg_to_dataset(tsg_ssl_test)

network_feature_data_train = [
    (data_train_http_x, data_train_http_y),
    (data_train_ssl_x, data_train_ssl_y),
]

network_feature_data_test = [
    (data_test_http_x, data_test_http_y),
    (data_test_ssl_x, data_test_ssl_y),
]
