# This is a sample code of linear regression written from scratch by Ozan KAYA.
# https://github.com/kayaozan

# The dataset is named as Auto MPG Data.
# "The data concerns city-cycle fuel consumption in miles per gallon,
# to be predicted in terms of 3 multivalued discrete and 5 continuous
# attributes." (Quinlan, 1993)
# The dataset is accessible via https://archive.ics.uci.edu/ml/datasets/Auto+MPG

# The objective of this code is to estimate the parameters of linear model
# and to predict the consumption in miles per gallon.

def load_and_prepare():
# Loads the data, divides it into train and test sets and returns them.

    columnNames = ['mpg', 'cyl', 'dsp', 'hp', 'wei', 'acc', 'year', 'ori', 'name']
    data = pd.read_fwf('auto-mpg.data', names = columnNames )

    # Replacing any unknown data with NaN
    data.replace('?', np.nan, inplace=True)

    # Dropping the rows that contain NaN, correcting the data type of a column
    # and dropping the column 'name' which is not needed for calculations
    data = data.dropna().astype({'hp' : 'float64'}).drop(columns= 'name').to_numpy()

    # 20% of the data to be selected for test purposes
    test_size = 0.2
    test_range = int(data.shape[0] * test_size)

    # Shuffling the data
    np.random.seed(0)
    np.random.shuffle(data)

    # Dividing the data set into training and test sets
    # Reshaping the arrays such that they work well with matrix operations
    x_test  = data[:test_range, 1:].reshape(test_range, -1)
    x_train = data[test_range:, 1:].reshape(data.shape[0]-test_range, -1)
    
    y_test  = data[:test_range, 0].reshape(-1, 1)
    y_train = data[test_range:, 0].reshape(-1, 1)

    return x_train, y_train, x_test, y_test

def normalize(x_train, x_test):
# Normalizes the feature sets.
# Uses the mean and standard deviation of the training set for both
# so that the regression model is not biased.

    trainMean = np.mean(x_train, axis=0)
    trainStd  = np.std( x_train, axis=0)
    
    x_train = (x_train - trainMean) / trainStd
    x_test  = (x_test  - trainMean) / trainStd

    return x_train, x_test

def h(x, theta):
# The hypothesis of linear regression.
# Simply the dot product of input variables and parameters.
    return x @ theta

def cost_function(x, y, theta):
# Calculates the sum of the squares of the residuals.
# Uses the formula: (1/2m)*sum((h-y)^2)
# where m is the size of samples
    cost = 0.5/y.shape[0] * (h(x, theta) - y).T @ (h(x, theta) - y)
    return np.squeeze(cost)

def gradient_descend(x, y, theta, learning_rate=0.1, epochs=100):
# Calculates the gradient descend for each parameter and adjusts them.
# Cost of each run is stored for monitoring purposes.

    J = []
    for i in range(epochs):
        theta = theta - learning_rate/y.shape[0] * x.T @ (h(x, theta) - y)
        J.append(cost_function(x, y, theta))

    return theta, J

def score(x, y, theta):
# Returns the coefficient of determination of the prediction.
# Uses the formula: 1 - (RSS/TSS)

    # Residual sum of squares
    RSS = np.squeeze((h(x, theta) - y).T @ (h(x, theta) - y))
    # Total sum of squares
    TSS = np.squeeze((y - np.mean(y)).T @ (y - np.mean(y)))
    
    return (1 - RSS/TSS)

import pandas as pd
import numpy as np
import matplotlib.pyplot as pl

x_train, y_train, x_test, y_test = load_and_prepare()

# Normalizing the feature sets
x_train, x_test = normalize(x_train, x_test)

# Adding a column of ones to the input variables for bias parameters
x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
x_test  = np.hstack((np.ones((x_test.shape[0], 1)), x_test))

# Initializing the parameters with zeros
theta = np.zeros([x_train.shape[1], 1])

# Sending the training set and parameters to start gradient descend
theta, J = gradient_descend(x_train, y_train, theta)

# Plotting the cost to ensure gradient descend has worked as intended
pl.plot(J)
pl.xlabel('number of runs by gradient descend')
pl.ylabel('J (cost)')
pl.show()

# Testing the model with the test set
accuracy = score(x_test, y_test, theta)
print('The accuracy measured with the test set:', accuracy)