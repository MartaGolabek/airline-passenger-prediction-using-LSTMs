"""
# Created by: marta
# Created on: 04.04.20
"""
import pandas as pd
import airlinepassenger.data.data_preprocessing as mid
import os.path

if __name__ == '__main__':
    if os.path.isfile('data/interim/international-airline-passengers.csv'):
        print("Interim data set already exists.")
        interim_data_set = pd.read_csv('../data/interim/international-airline-passengers.csv')
    else:
        raw_data_set = pd.read_csv('data/raw/international-airline-passengers.csv')
        interim_data_set = mid.initial_data_preprocessing(raw_data_set)
        with open('data/interim/international-airline-passengers.csv', 'w') as outfile:
            interim_data_set.to_csv(outfile, header=True, index=False)
        print("Initial preprocessing has been completed.")

    # Univariate Approach
    univariate_df = interim_data_set["passenger_number"]


    # Multivariate Approach
