# airline-passenger-prediction-using-LSTMs

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General info
This project predicts a number of international airline passengers using LSTMs. It uses both univariate and multivariate approaches as well as enables hyperparameter tuning.
	
## Technologies
Project is created with:
* Conda: 4.8.3
* Python: 3.6.10
* Check versions of other libraries in environment.yml
	
## Setup
To run this project, install it locally using conda:

```
$ cd ../airline-passenger-prediction-using-LSTMs
$ conda env create -f environment.yml
$ python airlinepassenger.py --config "/home/user/airline-passenger-prediction-using-LSTMs/config/airlinepassenger.yaml --approach "multivariate" --size 6 [--max_nr_hidden_layer] [--max_nr_lstm_units] [--max_nr_dense_hidden_layer_neurons]
```

The latter script will train a model, perfrom predictions, save scores for training, validation and test samples and generate a plot.

The project is also available in jupyter notebooks. There is a notebook for each approach.