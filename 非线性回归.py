# *_*coding:utf-8 *_*
#
from sklearn.preprocessing import PolynomialFeatures

x = [[3], [4]]

p = PolynomialFeatures(degree=2)

transform = p.fit_transform(x)

print(transform)

