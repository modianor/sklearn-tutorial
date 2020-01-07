# *_*coding:utf-8 *_*
#

# 生成一组0和1比例为9比1的样本，X为特征，y为对应的标签
from collections import Counter

import numpy as np
from imblearn.over_sampling import SMOTE
from keras.utils import np_utils

X = np.load('data/data_images.npy')
Y = np.load('data/data_labels.npy')
X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
Y = np.argmax(Y, axis=1)

print(Counter(Y))

smo = SMOTE(random_state=42)
X, Y = smo.fit_sample(X, Y)
print(Counter(Y))
X = X.reshape(X.shape[0], 112, 112,1)
Y = np_utils.to_categorical(Y)

np.save('augmentation_data/data_images.npy', X)
np.save('augmentation_data/data_labels.npy', Y)
