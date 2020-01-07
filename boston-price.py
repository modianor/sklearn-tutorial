# *_*coding:utf-8 *_*
#
from typing import List

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from sklearn.metrics import mean_squared_error
dataset: Bunch = load_boston()

X: List = dataset['data']
Y: List = dataset['target']
feature_names = dataset['feature_names']
DESCR = dataset['DESCR']
filename = dataset['filename']

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.20)

lr = LinearRegression()

lr.fit(x_train, y_train)

lr_predict = lr.predict(x_test)

error = mean_squared_error(y_test, lr_predict)

print(error)
