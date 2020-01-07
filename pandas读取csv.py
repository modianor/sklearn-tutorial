# *_*coding:utf-8 *_*
#
import pandas as pd
from numpy.core.multiarray import ndarray

dataset = pd.read_csv(filepath_or_buffer='data/online_shopping_10_cats.csv')

dataset = dataset[['cat', 'review']]

sample = dataset.sample(10)

values_true:ndarray = dataset.isnull().values == True

print(values_true.shape)

print(values_true[0])