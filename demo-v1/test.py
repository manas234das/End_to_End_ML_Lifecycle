# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor

from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

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

def return_test_sample():

    #Splitting Training and Test Set
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

    # Standardizing the data
    standardizer = StandardScaler()
    train_data = standardizer.fit_transform(X_train)
    test_data = standardizer.transform(X_test)

    return X_test, test_data, Y_test