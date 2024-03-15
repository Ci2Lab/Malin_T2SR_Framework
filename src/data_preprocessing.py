import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.nn.functional as T
from torch.utils.data import Dataset, random_split, DataLoader
from sklearn.model_selection import train_test_split
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
from torch.utils.data import DataLoader, TensorDataset
from math import pi, sin, cos
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.preprocessing import StandardScaler

"""
PIPELINE FOR THE DATA PREPROCESSING
"""




""""
Filling in missing data with linear interpolation in the high-resolution power concumption data.

Returns dataframe with consumption values and timestamps.
"""

def interpolate_and_fill_missing(df):

    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Set the 'timestamp' column as the DataFrame index
    df_copy.set_index('timestamp', inplace=True)

    # Resample the DataFrame with the desired frequency (2-second interval) and fill missing timestamps
    df_resampled = df_copy.resample('2S').asfreq()

    # Interpolate missing values (NaN) in the 'value' column using linear interpolation
    df_resampled['value'] = df_resampled['value'].interpolate(method='linear')

    # Reset the index to make 'timestamp' a column again
    df_resampled.reset_index(inplace=True)

    return df_resampled


    
    
"""
Downsamples the high-resolution data data into the desired low-resolution data

Returns dataframe with consumption values and timestamps.
"""    
    
def downsample_data(df, res):

    # Make a copy of the DataFrame to avoid modifying the original
    df_resampled = df.copy()
    
    # Convert the 'timestamp' column to datetime format
    df_resampled['timestamp'] = pd.to_datetime(df_resampled['timestamp'])
    
    # Set the 'timestamp' column as the DataFrame index
    df_resampled.set_index("timestamp", inplace=True)

    # Resampling resolution using the mean
    df_resampled = df_resampled.resample(res).agg({'value': 'mean'})

    # Resetting index to make it a column again
    df_resampled.reset_index(inplace=True)
    
    return df_resampled



"""
Function for mapping each high-resolution value to the desired low-resolution sequence.

Returns dataframe containing low-resolution values and its corresponding high-resolution sequences.
"""

def mapping_function(low_res_df, high_res_df, window_length):
    # Ensure the dataframes are sorted by timestamp in ascending order
    low_res_dataframe = low_res_df.sort_values(by="timestamp").reset_index(drop=True)
    high_res_dataframe = high_res_df.sort_values(by="timestamp").reset_index(drop=True)

    # Convert the timestamps to unix time for easy lookup
    low_res_dataframe["timestamp_unix"] = (
    pd.to_datetime(low_res_dataframe["timestamp"]).view('int64') // 10**9
        )
    high_res_dataframe["timestamp_unix"] = (
    pd.to_datetime(high_res_dataframe["timestamp"]).view('int64') // 10**9
        )   


    # Convert data to Numpy arrays for faster computations
    low_res_timestamps = low_res_dataframe["timestamp_unix"].values
    low_res_feature = low_res_dataframe.drop(columns=["timestamp", "timestamp_unix"]).values

    high_res_timestamps = high_res_dataframe["timestamp_unix"].values
    high_res_values = high_res_dataframe["value"].values

    # Create a lookup dictionary mapping timestamps to all low_res features
    low_res_lookup = dict(zip(low_res_timestamps, low_res_feature))

    # Initialize empty lists to store the high-resolution data (H) and low-resolution data (L)
    H_data = []
    L_data = []

    # Sliding window over the high-resolution dataframe
    for i in range(len(high_res_dataframe) - window_length + 1):
        # Extract the high-temporal-resolution data sequence in the sliding window 
        H_sequence_values = high_res_values[i : i + window_length]
        H_sequence_timestamps = high_res_timestamps[i : i + window_length]
        
        # Extract the corresponding low-temporal-resolution data point using the first timestamp in H_sequence
        L_features = low_res_lookup.get(H_sequence_timestamps[0], None)

        # Check if L_features exists and append it along with high_res data
        if L_features is not None:
            H_data.append(H_sequence_values)
            L_data.append(L_features)

    # Convert the lists to pandas Series
    H_series = pd.Series(H_data[:-2])  # Drop the last x rows to match lengths
    L_series = pd.Series(L_data[:-2])  # Drop the last x rows to match lengths

    # Create a new dataframe containing the mapped data
    mapped_df= pd.DataFrame({"H_sequence": H_series, "L_feature": L_series})

    return mapped_df




"""
Splits the dataframe into train, validation and test sets.

Returns three dataframes: train, validation and test
"""

def data_splitting(df, train_rows, validation_rows, test_rows):
    
    # Ensure the total number of rows for train and validation does not exceed the dataframe length
    assert train_rows + validation_rows+test_rows <= len(df), "Sum of train and validation rows exceeds dataframe length"

    # Use the first predetermined number of rows for train set
    train_df = df[:train_rows]
    
    # Use the next predetermined number of rows for validation set
    valid_df = df[train_rows:train_rows + validation_rows]
    
    # Use the next predetermined number of rows for test set
    test_df = df[train_rows + validation_rows:train_rows + validation_rows + test_rows]


    return train_df, valid_df, test_df



"""
Splits train, validation and test sets into feature and target.

Returns six lists: a feature list and a target list for each of the datasets.
"""

def split_feature_target (train_data, valid_data, test_data):

    # Splitting train dataset
    X_train = train_data["L_feature"].tolist()  # Convert pandas series of lists into list of lists
    y_train = train_data["H_sequence"].tolist()  # Convert pandas series of lists into list of lists

    # Splitting validation dataset
    X_valid = valid_data["L_feature"].tolist()
    y_valid = valid_data["H_sequence"].tolist()

    # Splitting test dataset
    X_test = test_data["L_feature"].tolist()
    y_test = test_data["H_sequence"].tolist()

    return  X_train, y_train, X_valid, y_valid, X_test, y_test






"""
Scaling the input and target data with StandardScaler.

Returns six lists.
"""

def scale_data (X_train, y_train, X_valid, y_valid, X_test, y_test):

    # Instantiate separate scalers for input and the target
    scaler_input = StandardScaler()
    scaler_target = StandardScaler()


    # Fit the scalers using training data
    scaler_input.fit(X_train)
    scaler_target.fit(y_train) 
    
    # Save the scaler_target for future inverse scaling
    joblib.dump(scaler_target, 'scaler_target.pkl')


    # Transform the input and targets for train, validation, and test sets

    for dataset in [X_train, X_valid, X_test]:
        for x in dataset:
            x[0] = scaler_input.transform([[x[0]]])[0][0]

    y_train = [scaler_target.transform([y])[0] for y in y_train]
    y_valid = [scaler_target.transform([y])[0] for y in y_valid]
    y_test = [scaler_target.transform([y])[0] for y in y_test]


    return X_train, y_train, X_valid, y_valid, X_test, y_test





"""
Converts pairs of inputs (X) and targets (y) into their respective dataloaders.

Returns a list of dataloaders corresponding to each (X, y) pair.
"""
def to_dataloaders(*dataset_pairs):

    dataloaders = []
    for X, y in dataset_pairs:
        X = np.array(X)
        y = np.array(y)

        X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)
        y_tensor = torch.tensor(y, dtype=torch.float32, requires_grad=True)

        dataset = TensorDataset(X_tensor, y_tensor)
        
        batch_size = 32
        dataloader = DataLoader(dataset, batch_size, shuffle=False)
        
        dataloaders.append(dataloader)
    
    return dataloaders
 

