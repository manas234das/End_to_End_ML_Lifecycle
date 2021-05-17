import os
import warnings
import sys
import pickle

import pandas as pd
import numpy as np
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost.sklearn import XGBRegressor 

import mlflow
import mlflow.sklearn


# Setting up TRACKING_URI
TRACKING_URI = "http://<userid>:<password>@ec2-instance-ip:5000/"    # set to your server URI
mlflow.set_tracking_uri(TRACKING_URI)
client = mlflow.tracking.MlflowClient(TRACKING_URI)

# Setting default experiment name as "Compare_algo"
exp_name = "Compare_algo"
mlflow.set_experiment(exp_name)


# Defining the evaluation metric
def eval_metrics(actual, pred):
    
    # Compute relevant metrics
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    
    return rmse, mae, r2


# Loading the data and split
def load_data():
    
    # Loading the dataset
    boston = load_boston()

    # Create the target and features dataframe
    boston_data = pd.DataFrame(boston.data)
    boston_target = pd.DataFrame(boston.target)

    # Renaming the columns
    boston_data.columns = boston.feature_names

    # Adding the price column
    boston_data['PRICE'] = boston_target

    # Creating label and feature data frame : Label- y, Features- X
    # Labels
    y = boston_data['PRICE'].values
    # Dropping price column
    boston_data.drop(['PRICE'], axis=1, inplace=True)
    # Features
    X = boston_data.values

    #Splitting Training and Test Set
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 45)

    # Standardizing the data
    standardizer = StandardScaler()
    train_data = standardizer.fit_transform(X_train)
    test_data = standardizer.transform(X_test)

    return train_data, test_data, Y_train, Y_test


# Creating different models
def models_selection(model_name, penalty=None, alpha=None, l1_ratio=None, learning_rate=None, n_estimators=None, max_depth=None):

    if model_name == 'Linear Regression':
        model = LinearRegression()
        return model

    elif model_name == 'SGD Regression':
        model = SGDRegressor(penalty=penalty, alpha=alpha, l1_ratio=l1_ratio, learning_rate='invscaling')
        return model
    
    elif model_name == 'RF Regression':
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
        return model

    elif model_name == 'DT Regression':
        model = DecisionTreeRegressor(max_depth=max_depth)
        return model
    
    elif model_name == 'GBDT Regression':
        model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
        return model
    
    elif model_name == 'XGB Regression':
        model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
        return model


# Defining the training function
def train_fcn(exp_name=None, model_name=None, penalty=None, 
            alpha=None, l1_ratio=None, learning_rate=None, n_estimators=None, max_depth=None):
    
    """
    # Training the model with the given parameters
    
    Parameters : 
        exp_name : Choose from "Compare_algo" or "Best_algo". 
                    Compare_algo is for comparing the differents model and best algo is the best selected algo
        model_name : Choose from 'Linear Regression', 'SGD Regression', 'RF Regression', 'DT Regression', 
                    'GBDT Regression', 'XGB Regression'

        Parameters for Linear Regression : None
        Parameters for SGD Regression : penalty, alpha, l1_ratio, 
                                        learning_rate (constant, optimal, invscaling, adaptive)
        Parameters for RF Regression : n_estimators (int), max_depth (int)
        Parameters for DT Regression : max_depth (int)
        Parameters for GBDT Regression : n_estimators (int), max_depth (int), learning_rate (float)
        Parameters for XGB Regression : n_estimators (int), max_depth (int), learning_rate (float)
    """
    
    
    # Create an experiment - "Compare_algo" or "Best_algo"
    exp_name = exp_name
    mlflow.set_experiment(exp_name)

    # Getting the data
    X_train, X_test, Y_train, Y_test = load_data()

    path = "./temp_images/"
    
    # Useful for multiple runs
    with mlflow.start_run():
        
        # Execute the model - Selecting the model based on the model name
        model = models_selection(model_name, penalty, alpha, l1_ratio, 
                                learning_rate, n_estimators, max_depth)
        model.fit(X_train, Y_train)
        
        # Evaluation 
        predictions = model.predict(X_test)
        rmse, mae, r2 = eval_metrics(Y_test, predictions)
        
        # Print metrics
        print("Model trained")
        print("Metric values :")
        print(rmse, mae, r2)

	# Creating a plot which will be tracked as an artifact
	# Calculating the errors
        # delta_y = Y_test - predictions
        # sns.distplot(np.array(delta_y), label = 'ERROR (ΔY)')
        # plt.xlabel('Errors : ΔY')
        # plt.ylabel('Density')
        # plt.title('Distribution of the Errors')
        # plt.savefig(path+'/error_dist.png')
        
        # Log_parameters, Metrics and Model to MLFlow
        mlflow.log_params({"penalty":penalty, "alpha":alpha, 
                            "l1_ratio":l1_ratio, "learning_rate":learning_rate, 
                            "n_estimators":n_estimators, "max_depth":max_depth})
        mlflow.log_metrics({"rmse":rmse, "mae":mae, "r2":r2})
        # mlflow.log_artifact(path+'/error_dist.png')
        print("Saved to {}".format(mlflow.get_artifact_uri()))
        
        # Load model
        mlflow.sklearn.log_model(model, model_name)

    # returning the metric value
    return r2


# # Running for testing
# # Linear regression
# train_fcn(exp_name='Compare_algo', model_name='Linear Regression')

# # SGD regression
# train_fcn(exp_name='Compare_algo', model_name='SGD Regression', penalty='l2', alpha=0.001, l1_ratio=0.10, learning_rate='optimal')

# RF Regression
# train_fcn(exp_name='Compare_algo', model_name='RF Regression', n_estimators=120, max_depth=3)

# # DT Regression
# train_fcn(exp_name='Compare_algo', model_name='DT Regression', max_depth=5)

# # GBDT Regression
# train_fcn(exp_name='Compare_algo', model_name='GBDT Regression', n_estimators=100, max_depth=5, learning_rate=0.01)

# # XGB Regression
# train_fcn(exp_name='Compare_algo', model_name='XGB Regression', n_estimators=100, max_depth=5, learning_rate=0.01)

