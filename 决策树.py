# *_*coding:utf-8 *_*
#
# import pydot
from sklearn.datasets import load_iris
from sklearn.externals.six import StringIO
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

dot_data = StringIO()
iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_pre = clf.predict(X_test)

print(classification_report(y_test, y_pre))
