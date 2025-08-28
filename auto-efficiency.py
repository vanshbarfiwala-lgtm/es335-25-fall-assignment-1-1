import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from base import DecisionTree 
np.random.seed(42)

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, sep=r'\s+', header=None,
                   names=["mpg","cylinders","displacement","horsepower","weight",
                          "acceleration","model year","origin","car name"])

data.replace('?', np.nan, inplace=True)
data.dropna(inplace=True)

y = data['mpg']
X = data.drop(['mpg','car name'], axis=1).astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

sk_tree = DecisionTreeRegressor(max_depth=5, random_state=42)
sk_tree.fit(X_train, y_train)
y_pred_sk = sk_tree.predict(X_test)

print("Sklearn DecisionTreeRegressor:")
print("MSE:", mean_squared_error(y_test, y_pred_sk))
print("R2:", r2_score(y_test, y_pred_sk))

X_train, X_test = X_train.reset_index(drop=True), X_test.reset_index(drop=True)
y_train, y_test = y_train.reset_index(drop=True), y_test.reset_index(drop=True)

custom_tree = DecisionTree(criterion="information_gain", max_depth=5)
custom_tree.fit(X_train, y_train)
y_pred_custom = custom_tree.predict(X_test)

print("\nCustom DecisionTree:")
print("MSE:", mean_squared_error(y_test, y_pred_custom))
print("R2:", r2_score(y_test, y_pred_custom))
