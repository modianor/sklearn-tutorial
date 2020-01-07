# *_*coding:utf-8 *_*
#
import cv2
import numpy as np
from sklearn.datasets import load_digits

dataset = load_digits()

X = dataset.data
Y = dataset.target

x: np.ndarray = X[1200]
x = x.reshape((8, 8))

cv2.namedWindow('X', cv2.WINDOW_NORMAL)
cv2.imshow('X', x)

cv2.waitKey(0)
