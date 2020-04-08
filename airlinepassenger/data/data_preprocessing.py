"""
# Created by: marta
# Created on: 04.04.20
"""
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
import os.path
import logging

logger = logging.getLogger(__name__)

# Create interim data set
def initial_data_preprocessing(data_set):
    data_set.rename(columns={data_set.columns[1]: "passenger_number",
                             data_set.columns[0]: "year_month"}, inplace=True)

    data_set = data_set[:-1]
    data_set["month"] = data_set["year_month"].apply(lambda x: re.sub("\d\d\d\d-0?", "", x))
    return data_set

# Create processed data set
def interim_data_preprocessing(interim_data_set):
    interim_data_set["previous_passenger_number"] = interim_data_set["passenger_number"].shift(1)
    interim_data_set = interim_data_set[1:]
    final_data_set = interim_data_set.drop(["year_month"], axis=1)
    return final_data_set


def normalize_univariate_series(univariate_data_set):
    univariate_array = univariate_data_set.values
    univariate_array_reshaped = univariate_array.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_univariate_array = scaler.fit_transform(univariate_array_reshaped)
    return scaler, normalized_univariate_array


def normalize_multivariate_series(train_set, validation_set, test_set, features_to_normalize):
    passenger_number_normalized = []
    all_features_normalized = []

    # fit & save demand scaler in order to denormalize predictions later on
    passenger_number_scaler = MinMaxScaler(feature_range=(0, 1))
    passenger_number_train = train_set['passenger_number'].values.reshape(len(train_set), 1)
    passenger_number_scaler = passenger_number_scaler.fit(passenger_number_train)

    # normalize passenger_number
    passenger_number_val = validation_set['passenger_number'].values.reshape(len(validation_set), 1)
    passenger_number_test = test_set['passenger_number'].values.reshape(len(test_set), 1)

    passenger_number_normalized_train = passenger_number_scaler.transform(passenger_number_train)
    passenger_number_normalized.append(passenger_number_normalized_train)

    passenger_number_normalized_val = passenger_number_scaler.transform(passenger_number_val)
    passenger_number_normalized.append(passenger_number_normalized_val)

    passenger_number_normalized_test = passenger_number_scaler.transform(passenger_number_test)
    passenger_number_normalized.append(passenger_number_normalized_test)

    # fit & save a scaler for all features
    all_features_scaler = ColumnTransformer(
        remainder='passthrough',  # passthough features not listed
        transformers=[
            ('mm', MinMaxScaler(feature_range=(0, 1)), features_to_normalize)
        ])

    # reshape train set
    temp_train = train_set.drop(columns=["passenger_number"])
    train_reshaped_df = temp_train.values.reshape((len(temp_train), len(temp_train.columns)))

    # fit & save the scaler for all features
    all_features_scaler = all_features_scaler.fit(train_reshaped_df)

    # normalize train set
    normalized_train_set = all_features_scaler.transform(train_reshaped_df)
    all_features_normalized.append(normalized_train_set)

    # reshape & normalize validation set
    temp_val = validation_set.drop(columns=["passenger_number"])
    val_reshaped_df = temp_val.values.reshape((len(temp_val), len(temp_val.columns)))
    normalized_val_set = all_features_scaler.transform(val_reshaped_df)
    all_features_normalized.append(normalized_val_set)

    # reshape & normalize test set
    temp_test = test_set.drop(columns=["passenger_number"])
    test_reshaped_df = temp_test.values.reshape((len(temp_test), len(temp_test.columns)))
    normalized_test_set = all_features_scaler.transform(test_reshaped_df)
    all_features_normalized.append(normalized_test_set)

    return passenger_number_scaler, passenger_number_normalized, all_features_normalized


# Create samples for univariate approach using Moving Window Technique
def split_univariate_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix][0]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# Create samples for multivariate approach using Moving Window Technique
def split_multivariate_sequence(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1:out_end_ix, -1]  # [lookahead]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# Create input-output pairs for train, validation and test sets
def create_univariate_samples(train_set, validation_set, test_set, input_window_size, output_window_size):
    train_samples = split_univariate_sequence(train_set, input_window_size, output_window_size)
    val_samples = split_univariate_sequence(validation_set, input_window_size, output_window_size)
    test_samples = split_univariate_sequence(test_set, input_window_size, output_window_size)

    logger.info(
        f"Number of training samples: {len(train_samples[0])}, validation samples: {len(val_samples[0])}, test samples: {len(test_samples[0])}")
    return train_samples, val_samples, test_samples


# Create input-output pairs for train, validation and test sets
def create_multivariate_samples(passenger_number_normalized, all_features_normalized, input_window_size,
                                output_window_size):
    train_set = np.hstack((all_features_normalized[0], passenger_number_normalized[0]))
    train_samples = split_multivariate_sequence(train_set, input_window_size, output_window_size)

    val_set = np.hstack((all_features_normalized[1], passenger_number_normalized[1]))
    val_samples = split_multivariate_sequence(val_set, input_window_size, output_window_size)

    test_set = np.hstack((all_features_normalized[2], passenger_number_normalized[2]))
    test_samples = split_multivariate_sequence(test_set, input_window_size, output_window_size)

    logger.info(
        f"Number of training samples: {len(train_samples[0])}, validation samples: {len(val_samples[0])}, test samples: {len(test_samples[0])}")
    return train_samples, val_samples, test_samples


# Accomplish entire data preprocessing for univariate approach
def make_univariate_dataset(config, input_window_size):
    if os.path.isfile(config.interim_data):
        logger.info("Interim data set already exists.")
        interim_data_set = pd.read_csv(config.interim_data)
    else:
        raw_data_set = pd.read_csv(config.raw_data)
        interim_data_set = initial_data_preprocessing(raw_data_set)
        with open(config.interim_data, 'w') as outfile:
            interim_data_set.to_csv(outfile, header=True, index=False)
        logger.info("Initial preprocessing has been completed.")

    # Normalization
    univariate_data_set = interim_data_set["passenger_number"]
    scaler, normalized_univariate_array = normalize_univariate_series(univariate_data_set)

    # Train, Validation, Test Split
    # 2 years are allocated to each validation & test, the rest of the time steps are allocated to the training set
    train_size = len(normalized_univariate_array) - 48
    validation_size = 24
    train_set, validation_set, test_set = normalized_univariate_array[0:train_size, :], \
                                          normalized_univariate_array[train_size:train_size + validation_size, :], \
                                          normalized_univariate_array[
                                          train_size + validation_size:len(normalized_univariate_array), :]
    logger.info(f"Training set: {len(train_set)}, validation set: {len(validation_set)}, test set: {len(test_set)}")

    # Create samples using Moving Window Technique
    train_samples, val_samples, test_samples = create_univariate_samples(train_set, validation_set, test_set,
                                                                         input_window_size, 1)

    return train_samples, val_samples, test_samples


# Accomplish entire data preprocessing for multivariate approach
def make_multivariate_dataset(config, input_window_size):
    if os.path.isfile(config.processed_data):
        logger.info("Processed data set already exists.")
        processed_data_set = pd.read_csv(config.processed_data)
    else:
        # check whether interim data exists
        interim_data_set = pd.read_csv(config.interim_data)
        processed_data_set = interim_data_preprocessing(interim_data_set)
        with open(config.processed_data, 'w') as outfile:
            processed_data_set.to_csv(outfile, header=True, index=False)
        logger.info("Feature engineering has been completed.")

    # Train, Validation, Test Split
    train_size = len(processed_data_set) - 48
    validation_size = 24
    train_set, validation_set, test_set = processed_data_set[0:train_size + 1], \
                                          processed_data_set[train_size:train_size + validation_size], \
                                          processed_data_set[train_size + validation_size:]
    logger.info(f"Training set: {len(train_set)}, validation set: {len(validation_set)}, test set: {len(test_set)}")

    # Normalize Data
    features_to_normalize = [0, 1]  # apart from target variable, normalize month and previous_passenger_number
    passenger_number_scaler, passenger_number_normalized, all_features_normalized = normalize_multivariate_series(
        train_set, validation_set, test_set, features_to_normalize)

    train_samples, val_samples, test_samples = create_multivariate_samples(passenger_number_normalized, all_features_normalized,
                                                              input_window_size, 1)

    return train_samples, val_samples, test_samples