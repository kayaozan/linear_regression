# Linear Regression

This page demostrates some sample code for linear regression. Two Python files has been shared.

`linear_regression.py` is written from scratch without any Machine Learning libraries such as `scikit-learn`.

`linear_regression_sklearn.py` uses the capabilities of `scikit-learn` tools.

## Results

### Gradient Descend

`linear_regression.py` manages to decrease the cost and it converges as seen below.

<img src="https://user-images.githubusercontent.com/22200109/210356681-c1b12a68-59d4-4f14-aa43-2aba598b3eb5.png" width="500">

### Accuracy

Using the same train and and test sets, the accuracy scores of both version:


|   | Accuracy |
| ------------- | ------------- |
| `linear_regression.py`  | 0.8324  |
| `linear_regression_sklearn.py`  | 0.8340  |
