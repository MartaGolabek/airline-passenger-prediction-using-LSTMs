"""
# Created by: marta
# Created on: 04.04.20
"""
import re


def initial_data_preprocessing(data_set):
    data_set.rename(columns={data_set.columns[1]: "passenger_number",
                             data_set.columns[0]: "year_month"}, inplace=True)

    data_set = data_set[:-1]
    data_set["month"] = data_set["year_month"].apply(lambda x: re.sub("\d\d\d\d-0?", "", x))
    return data_set


def create_univariate_data_set():
    pass

def train_val_test_split():
    pass
