# *_*coding:utf-8 *_*
#
'''
    决策树分类：决策树分类模型会找到与样本特征匹配的叶子节点然后以投票的方式进行分类。
            在样本文件中统计了小汽车的常见特征信息及小汽车的分类，使用这些数据基于决策树分类算法训练模型预测小汽车等级。
                    特征信息：   汽车价格    维修费用    车门数量    载客数    后备箱    安全性    汽车级别

    案例：基于决策树分类算法训练模型预测小汽车等级。

        1.读取文本数据，对每列进行标签编码，基于随机森林分类器进行交叉验证,模型训练.
        2.自定义测试集，使用已训练的模型对测试集进行测试，输出结果。
'''
import csv
import warnings
from _csv import reader
from typing import Tuple

import numpy as np
from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle

warnings.filterwarnings('ignore')


def load_dataset() -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    with open('data/traindata.csv', 'r') as r:
        csv_reader: reader = csv.reader(r)
        x_list = []
        y_list = []
        for row in csv_reader:
            x_list.append(row[:18])
            y_list.append(row[18:19])
        x_train = np.array(x_list)
        y_train = np.array(y_list)

        x_train, y_train = shuffle(x_train, y_train)

    with open('data/testdata.csv', 'r') as r:
        csv_reader: reader = csv.reader(r)
        x_list = []
        y_list = []
        for row in csv_reader:
            x_list.append(row[:18])
            y_list.append(row[18:19])
        x_test = np.array(x_list)
        y_test = np.array(y_list)

        x_test, y_test = shuffle(x_test, y_test)
    return (x_train, y_train, x_test, y_test)


def RandomForestClassifier(x_train, y_train, x_test, y_test):
    # 训练随机森林分类器
    model = RandomForestClassifier(max_depth=18, n_estimators=200, random_state=14)
    # 训练之前进行交叉验证
    model.fit(x_train, y_train)
    y_pre = model.predict(x_test)
    print(classification_report(y_test, y_pre))


def KnnClassifier(x_train, y_train, x_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=3)

    knn.fit(x_train, y_train)

    y_pre = knn.predict(x_test)

    print(classification_report(y_test, y_pre))


x_train, y_train, x_test, y_test = load_dataset()

KnnClassifier(x_train, y_train, x_test, y_test)
