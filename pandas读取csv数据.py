# *_*coding:utf-8 *_*
#

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import Normalizer
from sklearn.utils import shuffle

names = [
    'f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7',
    'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14',
    'f15', 'f16', 'f17', 'c0']

dataset_train: DataFrame = pd.read_csv(filepath_or_buffer='data/traindata.csv', names=names)
dataset_test: DataFrame = pd.read_csv(filepath_or_buffer='data/testdata.csv', names=names)

dataset_train = np.array(dataset_train)
dataset_test = np.array(dataset_test)

dataset_train = shuffle(dataset_train)
dataset_test = shuffle(dataset_test)

x_train, y_train = dataset_train[:, :18], dataset_train[:, 18]
x_test, y_test = dataset_test[:, :18], dataset_test[:, 18]

normalizer = Normalizer()

x_train = normalizer.fit_transform(x_train)
x_test = normalizer.fit_transform(x_test)

model = RandomForestClassifier(max_depth=18, n_estimators=200, random_state=14)

model.fit(x_train, y_train)

y_pre = model.predict(x_test)

print(classification_report(y_test, y_pre))
