# This is a sample code of linear regression performed via scikit-learn, written by Ozan KAYA.
# https://github.com/kayaozan

# The dataset is named as Auto MPG Data.
# "The data concerns city-cycle fuel consumption in miles per gallon,
# to be predicted in terms of 3 multivalued discrete and 5 continuous
# attributes." (Quinlan, 1993)
# The dataset is accessible via https://archive.ics.uci.edu/ml/datasets/Auto+MPG

# The objective of this code is to estimate the parameters of linear model
# and to predict the consumption in miles per gallon.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

columnNames = ['mpg', 'cyl', 'dsp', 'hp', 'wei', 'acc', 'year', 'ori', 'name']

# Loading the data with respected column names
data = pd.read_fwf('auto-mpg.data', names = columnNames )

# Replacing unknown data with NaN and removing the rows with any NaNs
data.replace('?', np.nan, inplace=True)
data = data.dropna().drop(columns='name')

# Splitting the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    (data.loc[:,data.columns != 'mpg']),
     data.mpg,
     test_size=0.2,
     random_state=0
     )

# Normalizing the data before processing
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Modelling the Linear Regression
linReg = LinearRegression()
linReg.fit(x_train, y_train)

# Calculating and printing the accuracy
print('The accuracy measured with test set is:\n', linReg.score(x_test, y_test))