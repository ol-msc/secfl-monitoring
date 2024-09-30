# -*- coding: utf-8 -*-
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Created on Mon Apr 23 12:48:52 2018

@author: giangnguyen
@author: stefan dlugolinsky
"""

import calendar
import datetime
import glob
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import string
import time
# from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import *
from math import sqrt
from os.path import basename
from random import randint
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from numpy import dot
from numpy.linalg import norm

from pandas import DataFrame
from pandas import concat

import keras

# import mods.config as cfg


# matplotlib.style.use('ggplot')
# %matplotlib inline

# import bat
# import pyarrow
# import pyspark
# def print_libs_versions():
#    print('BAT: {:s}'.format(bat.__version__))
#    print('Numpy: {:s}'.format(np.__version__))
#    print('Pandas: {:s}'.format(pd.__version__))
#    print('PyArrow: {:s}'.format(pyarrow.__version__))
#    print('PySpark: {:s}'.format(pyspark.__version__))
#    print('Scikit Learn: ', sklearn.__version__)
#    return


##### Datapool functions #####

# @giang
# time_range_excluded = [ ['20200416', '20200417'],['20200428', '20200429'] ]
# def check_excluded(day, time_range_excluded=cfg.train_time_ranges_excluded):
#     for range in time_range_excluded:
#         if range[0] <= day <= range[1]:
#             return True
#     return False


# @giang
# def create_files_from_tsv(output_dir=cfg.app_data_train,
#                           ws_choice = cfg.ws_choice,
#                           app_data_features=cfg.app_data_features,              # conn, dns, http, sip, ssh, ssl
#                           time_range_begin=cfg.train_time_range_begin,
#                           time_range_end=cfg.train_time_range_end,
#                           time_range_excluded=cfg.train_time_ranges_excluded
#                           ):

#     for protocol in sorted(next(os.walk(app_data_features))[1]):                # conn, dns, http, sip, ssh, ssl
#         print(app_data_features + protocol)
#         flist = []
#         for root, directories, filenames in sorted(os.walk(app_data_features + protocol)):
#             for fn in filenames:
#                 ffn = os.path.join(root, fn)
#                 ws  = os.path.basename(ffn).split('.')[0]
#                 day = ''.join(ffn.split('/')[-4:-1])
#                 if  (ws == ws_choice) and\
#                     (time_range_begin <= day <= time_range_end) and\
#                     (not check_excluded(day, time_range_excluded)):
#                     flist.append(ffn)
#         # print(flist)

#         filename = output_dir + ws_choice + '/' + protocol + '.tsv'
#         write_header = True
#         with open(filename, 'w') as fout:
#             for fn in flist:
#                 with open(fn) as fin:
#                     if write_header:
#                         header = fin.readline()
#                         fout.write(header)
#                         write_header = False
#                     else:
#                         next(fin)
#                     for line in fin:
#                         fout.write(line)
#         print('created data=', filename + '\n')
#     return


# @giang: merge multiple .tsv files -> pandas dataframe -> numpy array
# modified from @stevo: https://github.com/deephdc/mods/blob/5594c3d35313b4363f998bee0b65f9aadffba245/mods/models/api.py#L477
# def merge_files_on_cols(output_filename=cfg.app_data + cfg.data_filename_train,
#                         data_pool=cfg.app_data_train + cfg.ws_choice,
#                         data_select_query=cfg.data_select_query
#                         ):
#     # loading and merging data
#     keep_cols = []
#     df_data = None
#     data_files, merge_on_col = parse_data_specs(data_select_query)

#     for data_file in data_files:
#         fname = data_file['protocol']
#         data_file['protocol'] = os.path.join(data_pool, data_file['protocol'] + '.tsv')

#         # columns to be loaded: columns specified for the file as well as columns, that will be used for joins
#         d = {}
#         for col_name in data_file['cols']:
#             d[col_name[0]] = col_name[0]          # original column name
#             if len(col_name) > 1:
#                 d[col_name[0]] = col_name[1]      # rename column

#         cols = list(d.keys())
#         cols.extend(merge_on_col)

#         # load one of the data files
#         df = pd.read_csv(data_file['protocol'],
#                          usecols=cols,
#                          sep=cfg.pd_sep,
#                          skiprows=0,
#                          skipfooter=0,
#                          engine='python'
#         )

#         if cfg.fill_missing_rows_in_timeseries:
#             df = fill_missing_rows(df)

#         # rename columns
#         # add protocol prefix (conn, ssh, ...) to column names
#         df.rename(columns=lambda col: col if col in merge_on_col else fname + '_' + d[col], inplace=True)

#         # collect columns, that will be kept in the final dataset
#         keep_cols.extend([col for col in list(df) if col not in merge_on_col])

#         # convert units:
#         # from B to kB, MB, GB use _kB, MB, GB
#         df = df.fillna(0)
#         for col in df.columns:
#             if col.lower().endswith('_kb'):
#                 df[col] = df[col].div(1024).astype(int)
#             elif col.lower().endswith('_mb'):
#                 df[col] = df[col].div(1048576).astype(int)
#             elif col.lower().endswith('_gb'):
#                 df[col] = df[col].div(1073741824).astype(int)

#         if df_data is None:
#             df_data = df
#         else:
#             df_data = pd.merge(df_data, df, on=merge_on_col)

#     # select only specified columns
#     data = df_data[keep_cols]

#     # save data to file
#     data.to_csv(output_filename, sep='\t', index=False)

#     print('\n created data=', output_filename + '\n')
#     print(len(list(df_data)), list(df_data))

#     return


##### Data functions #####

# @giang
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df = df.drop(0)
    return df


# @giang: read .tsv file -> pandas dataframe
def create_df(filename):
    df = pd.read_csv(filename,
                     sep=cfg.pd_sep,
                     skiprows=0,
                     skipfooter=0,
                     engine='python'
                     # usecols=lambda col: col in cfg.pd_usecols
                     )

    # data cleaning + missing values     # Intermittent Demand Analysis (IDA) or Sparse Data Analysis (SDA)
    df = fix_missing_num_values(df)

    # in_sum_orig_bytes, in_sum_resp_bytes, out_sum_orig_bytes, out_sum_resp_bytes in MB without rename columns
    # for feature in list(df):
    #     if '_bytes' in feature:
    #         df[feature] = df[feature].div(1024*1024).astype(int)

    print('create_df', filename, '\t', len(df.columns), df.shape, '\n', list(df))
    return df


# @giang: read .tsv file -> pandas dataframe -> numpy array
def read_data(filename):
    df = create_df(filename)

    # Data: pandas dataframe to numpy array
    data = df.values.astype('float32')

    print('read_data: ', filename, '\t', data.shape[1], data.dtype, '\n', list(df))
    return data


# @giang
# def get_fullpath_model_name(dataset_name, sequence_len=cfg.sequence_len):
#     model_name = (cfg.app_models +
#                   os.path.splitext(basename(dataset_name))[0] +
#                   '-seq-' + str(sequence_len) + '.h5')
#     return model_name


# @giang: first order differential d(y)/d(t)=f(y,t)=y' for numpy array
def delta_timeseries(arr):
    return arr[1:] - arr[:-1]


# smoothed differentiable SMAPE in range <0,200>
# https://github.com/Arturus/kaggle-web-traffic/blob/master/how_it_works.md
def smape_smooth(y_true, y_pred):
    assert isinstance(y_true, np.ndarray), 'numpy array expected for y_true in smape'
    assert isinstance(y_pred, np.ndarray), 'numpy array expected for y_pred in smape'
    score = []
    for i in range(y_true.shape[1]):
        try:
            epsilon = 0.1
            sm = np.abs(y_true[:,i]) + np.abs(y_pred[:,i])
            ssm = np.maximum(sm + epsilon, 0.5 + epsilon)
            s = 100 / len(y_true[:,i]) * np.sum(2*np.abs(y_pred[:,i] - y_true[:,i]) / ssm)
            if np.isnan(s):
                s = str(s)
            score.append(s)
        except ZeroDivisionError:
            score.append(str(np.nan))
    return score

# SMAPE in range <0,200>
# @giang/@stevo: SMAPE = 100/len(A) * np.sum(2 * np.abs(F-A) / (np.abs(A) + np.abs(F))), symmetric function
def smape(y_true, y_pred):
     assert isinstance(y_true, np.ndarray), 'numpy array expected for y_true in smape'
     assert isinstance(y_pred, np.ndarray), 'numpy array expected for y_pred in smape'
     score = []
     for i in range(y_true.shape[1]):
         try:
             sm = np.abs(y_true[:,i]) + np.abs(y_pred[:,i])
             s = 100 / len(y_true[:, i]) * np.sum(2*np.abs(y_pred[:, i] - y_true[:, i]) / sm)
             if np.isnan(s):
                 s = str(s)
             score.append(s)
         except ZeroDivisionError:
             score.append(str(np.nan))
     return score


# @giang/@stevo: MAPE = np.mean(np.abs((A-F)/A)) * 100
def mape(y_true, y_pred):
    assert isinstance(y_true, np.ndarray), 'numpy array expected for y_true in mape'
    assert isinstance(y_pred, np.ndarray), 'numpy array expected for y_pred in mape'
    score = []
    for i in range(y_true.shape[1]):
        try:
            s = np.mean(np.abs((y_true[:,i] - y_pred[:,i]) / y_true[:,i])) * 100
            if np.isnan(s):
                s = str(s)
            score.append(s)
        except ZeroDivisionError:
            score.append(str(np.nan))
    return score


# @giang: RMSE for numpy array
def rmse(a, b):
    score = []
    for i in range(a.shape[1]):
        score.append(sqrt(mean_squared_error(a[:,i], b[:,i])))
    return score


# @giang: cosine similarity for two numpy arrays, <-1.0, 1.0>
def cosine(a, b):
    score = []
    for i in range(a.shape[1]):
        cos_sim = dot(a[:, i], b[:, i]) / (norm(a[:, i]) * norm(b[:, i]))
        score.append(cos_sim)
    return score


# @giang: R^2 (coefficient of determination) regression score, <-1.0, 1.0>, not a symmetric function
def r2(a, b):
    score = []
    for i in range(a.shape[1]):
        score.append(r2_score(a[:, i], b[:, i]))
    return score




##### @giang auxiliary - BEGIN - functions in this block can be removed later !!! #####

# @giang: get X from TimeseriesGenerator data
def getX(tsg_data):
    X = list()
    for x, y in tsg_data:
        X.append(x)
    return np.array(X)


# @giang: get Y from TimeseriesGenerator data
def getY(tsg_data):
    Y = list()
    for x, y in tsg_data:
        Y.append(y)
    return np.array(Y)


# @giang: get XY from TimeseriesGenerator data
def getXY(tsg_data):
    X, Y = list(), list()
    for x, y in tsg_data:
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)


# @giang data = numpy array
def plot_series(data, ylabel):
    plt.plot(data)
    plt.ylabel(ylabel)
    plt.show()


def get_one_row(i, dataset):
    row = dataset[i, :]
    return row[1:].reshape(1, -1)  # i-th row without label as [row]


def get_random_row(dataset):
    i = randint(0, dataset.shape[0])
    return get_one_row(i, dataset)


def load_dataset(dataset_name):
    dataset = np.loadtxt(cfg.app_data + dataset_name, delimiter=',')
    print(dataset_name, dataset.shape)
    return dataset


##### @giang auxiliary - END - functions in this above block can be removed later #####


##### @stevo @stevo @stevo#####


# @stevo
REGEX_TIME_INTERVAL = re.compile(
    r'((?P<years>\d)\s+years?\s+)?((?P<months>\d)\s+months?\s+)?((?P<days>\d)\s+days?\s+)?(?P<hours>\d{2}):(?P<minutes>\d{2}):(?P<seconds>\d{2})(?P<nanoseconds>\.\d+)')


# @stevo
def parseInterval(s):
    global REGEX_TIME_INTERVAL
    time_array = [['nanoseconds', 1],
                  ['seconds', 1],
                  ['minutes', 60],
                  ['hours', 3600],
                  ['days', 86400],
                  ['months', 1036800],
                  ['years', 378432000]]

    m = REGEX_TIME_INTERVAL.search(s.strip())
    seconds = float(0.0)
    for t in time_array:
        seconds += float(m.group(t[0])) * t[1] if m.group(t[0]) else 0
    return seconds


# returns metadata filename based on model filename
def get_metadata_filename(model_filename):
    return re.sub(r'\.[^.]+$', r'.json', model_filename)


# loads model and model's metadata
def load_model(filename, metadata_filename):
    try:
        model = keras.models.__load_model(filename)
        metadata = load_model_metadata(metadata_filename)
        return model, metadata
    except Exception as e:
        print(e)
        return None


# loads and returns model metadata
def load_model_metadata(metadata_filename):
    try:
        with open(metadata_filename, 'rb') as f:
            return json.load(f)
    except Exception as e:
        print(e)
    return None


# @stevo
def parse_int_or_str(val):
    val = val.strip()
    try:
        return int(val)
    except Exception:
        return str(val)


# @stevo
def compute_metrics(y_true, y_pred, model):
    result = {}

    if len(y_true) > 1 and len(y_true) == len(y_pred):

        eval_result = model.eval(y_true)

        if isinstance(y_true, pd.DataFrame):
            y_true = y_true.values

        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred.values

        err_mape = mape(y_true, y_pred)
        err_smape = smape(y_true, y_pred)
        err_r2 = r2(y_true, y_pred)
        err_rmse = rmse(y_true, y_pred)
        err_cosine = cosine(y_true, y_pred)

        result['mods_mape'] = err_mape
        result['mods_smape'] = err_smape
        result['mods_r2'] = err_r2
        result['mods_rmse'] = err_rmse
        result['mods_cosine'] = err_cosine

        i = 0
        for metric in model.model.metrics_names:
            result[metric] = eval_result[i]
            i += 1

    return result


# @stevo tsv representation of a dataframe
def df2tsv(df):
    if isinstance(df, pd.DataFrame):
        df = df.values
    ret = ''
    for row in df:
        for col in row:
            ret += str(col) + '\t'
        ret += '\n'
    return ret


# @stevo tsv representation of a tsg
def tsg2tsv(tsg):
    ret = ''
    for i in range(len(tsg)):
        x, y = tsg[i]
        ret += '%s => %s\n' % (x, y)
    return ret


# @stevo saves dataframe to a file
def save_df(df, model_name, file):
    dir = os.path.join(cfg.app_data, model_name[:-4] if model_name.lower().endswith('.zip') else model_name)
    if not os.path.isdir(dir):
        if os.path.isfile(dir):
            raise NotADirectoryError(dir)
        os.mkdir(dir)
    with open(os.path.join(dir, file), mode='w') as f:
        f.write(df2tsv(df))
        f.close()


# @stevo prints dataframe within a range of rows
def print_df(df, name, min=0, max=9):
    print('%s:\n%s' % (name, df2tsv(df[min:max])))


# @stevo prints dataframe to stdout and/or saves it to a file <<model>>/<<name>>.tsv
def dbg_df(df, model_name, df_name, print=False, save=False):
    if print:
        print_df(df, df_name)
    if save:
        save_df(df, model_name, df_name + '.tsv')


# @stevo prints TimeSeriesGenerator to stdout
def dbg_tsg(tsg, msg, debug=False):
    if debug:
        print('%s:\n%s' % (msg, tsg2tsv(tsg)))


# @stevo prints scaler to stdout
def dbg_scaler(scaler, msg, debug=False):
    if debug:
        print('%s - scaler.get_params(): %s\n\tscaler.data_min_=%s\n\tscaler.data_max_=%s\n\tscaler.data_range_=%s'
              % (
                  msg,
                  scaler.get_params(),
                  scaler.data_min_,
                  scaler.data_max_,
                  scaler.data_range_
              ))


# @stevo - parses data specification in order to support multiple data files merging
REGEX_SPLIT_SEMICOLON = re.compile(r'\s*;\s*')
REGEX_SPLIT_COMMA = re.compile(r'\s*,\s*')
REGEX_SPLIT_HASH = re.compile(r'\s*#\s*')
REGEX_SPLIT_PIPE = re.compile(r'\s*\|\s*')
REGEX_SPLIT_TILDE = re.compile(r'\s*~\s*')
def parse_data_specs(specs):

    protocols = []
    merge_on_col = []

    specs = REGEX_SPLIT_SEMICOLON.split(specs.strip())

    if specs and len(specs) > 0:

        x = REGEX_SPLIT_HASH.split(specs[-1], 1)
        if x and len(x) == 2:
            specs[-1] = x[0]
            merge_on_col = list(filter(None, REGEX_SPLIT_COMMA.split(x[1])))

        for spec in specs:
            # parse an array of file names (separated by |)
            parsed = REGEX_SPLIT_PIPE.split(spec)
            protocol = parsed[0]
            columns = parsed[1:] if len(parsed) > 1 else []
            # column rename rules
            columns = [REGEX_SPLIT_TILDE.split(col, 1) for col in columns]
            # columns.extend(merge_on_col)
            protocols.append({'protocol': protocol, 'cols': columns})

    return (protocols, merge_on_col)


# @stevo
def fix_missing_num_values(df, cols=None):
    if cols:
        for col in cols:
            df['col'] = pd.to_numeric(df['col'], errors='coerce')
            df['col'] = df['col'].replace(r'^\s*$', np.nan, regex=True, inplace=True)
            df['col'] = df['col'].replace([np.inf, -np.inf], np.nan, inplace=True)
            df['col'] = df['col'].replace(['NaN', np.nan], 0, inplace=True)
            df['col'] = df['col'].interpolate(inplace=True)
    else:
        df = df.apply(pd.to_numeric, errors='coerce')
        df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.replace(['NaN', np.nan], 0, inplace=True)
        df.interpolate(inplace=True)
    return df


# @stevo
def estimate_window_spec(df):
    tmpdf = df[['window_start', 'window_end']]
    slide_duration = tmpdf['window_start'].diff()[1:].min()
    row1 = tmpdf[:1]
    window_duration = row1.window_end[:1].iloc[0] - row1.window_start[:1].iloc[0]
    return window_duration, slide_duration


# @stevo
def fill_missing_rows(df, range_beg=None, range_end=None):
    """Fills the missing rows in the time series dataframe by estimating the slide and window duration.
    Parameters
    ----------
    df : pandas.DataFrame
        Data
    range_beg : str
        Time range begin in 'YYYY-MM-DD hh:mm:ss' format (default is None)
    range_end : str
        Time range end in 'YYYY-MM-DD hh:mm:ss' format (default is None)
    Returns
    -------
    pandas.DataFrame
        DataFrame filled with missing rows
    """
    if not ('window_start' in df.columns and 'window_end'):
        return df
    numrows = len(df.index)
    # convert cols to datetime
    df = df.apply(lambda x: pd.to_datetime(x) if x.name in ['window_start', 'window_end'] else x)
    # estimate window specification
    window_duration, slide_duration = estimate_window_spec(df)
    tz = df[:1]['window_start'].iloc[0].tzinfo
    if range_beg:
        range_beg = pd.Timestamp(range_beg, tzinfo=tz)
        if range_beg < df[:1]['window_start'].iloc[0]:
            # add the first row for the specified range to fill from
            df = df.shift()
            df.loc[0, 'window_start'] = range_beg
            df.loc[0, 'window_end'] = range_beg + window_duration
    if range_end:
        range_end = pd.Timestamp(range_end, tzinfo=tz)
        if range_end > df[-1:]['window_end'].iloc[0]:
            # add the last row for the specified range to fill to
            df = df.append(pd.Series(), ignore_index=True)
            df.loc[df.index[-1], 'window_start'] = range_end - window_duration
            df.loc[df.index[-1], 'window_end'] = range_end
    # set df index
    df = df.set_index('window_start')
    # fill missing rows using slide_duration as the frequency
    df = df.asfreq(slide_duration)
    # reset index to use window_start as a column
    df = df.reset_index(level=0)
    # compute window_end values for the newly added rows (not necessary at the moment)
    df['window_end'] = df['window_start'] + window_duration
    newnumrows = len(df.index)
    if newnumrows > numrows:
        print('filled %d missing rows (was %d)' % (newnumrows - numrows, numrows))
    return df
