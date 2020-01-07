# *_*coding:utf-8 *_*
#
from sklearn.datasets import load_iris

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

dataset = load_iris()

X = dataset.data
Y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_pre = knn.predict(X_test)

print(classification_report(y_test, y_pre))

print(y_test)
