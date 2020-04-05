"""
# Created by: marta
# Created on: 04.04.20
"""
import pandas as pd
import airlinepassenger.data.data_preprocessing as mid
import os.path

if __name__ == '__main__':
    # Data preprocessing

    # Load or create data sets
    if os.path.isfile('data/interim/international-airline-passengers.csv'):
        print("Interim data set already exists.")
        interim_data_set = pd.read_csv('../data/interim/international-airline-passengers.csv')
    else:
        raw_data_set = pd.read_csv('data/raw/international-airline-passengers.csv')
        interim_data_set = mid.initial_data_preprocessing(raw_data_set)
        with open('data/interim/international-airline-passengers.csv', 'w') as outfile:
            interim_data_set.to_csv(outfile, header=True, index=False)
        print("Initial preprocessing has been completed.")

    if os.path.isfile('data/processed/international-airline-passengers.csv'):
        print("Processed data set already exists.")
        processed_data_set = pd.read_csv('../data/processed/international-airline-passengers.csv')
    else:
        interim_data_set = pd.read_csv('../data/interim/international-airline-passengers.csv')
        processed_data_set = mid.create_multivariate_data_set(interim_data_set)
        with open('data/processed/international-airline-passengers.csv', 'w') as outfile:
            processed_data_set.to_csv(outfile, header=True, index=False)
        print("Feature engineering has been completed.")

    ############################################# Univariate Approach ######################################################
    # Normalization
    univariate_data_set = interim_data_set["passenger_number"]
    scaler, normalized_univariate_array = mid.normalize_univariate_series(univariate_data_set)

    # Train, Validation, Test Split
    # 2 years are allocated to each validation & test, the rest of the time steps are allocated to the training set
    train_size = len(normalized_univariate_array) - 48
    validation_size = 24
    test_size = 24
    train_set, validation_set, test_set = normalized_univariate_array[0:train_size, :], \
                                          normalized_univariate_array[train_size:train_size + validation_size, :], \
                                          normalized_univariate_array[
                                          train_size + validation_size:len(normalized_univariate_array), :]
    print(f"Training set: {len(train_set)}, validation set: {len(validation_set)}, test set: {len(test_set)}")

    # Create samples using Moving Window Technique
    # make input_window_size a parameter
    # mid.split_univariate_sequence
    # mid.create_univariate_samples

############################################ Multivariate Approach #####################################################
