# *_*coding:utf-8 *_*
#
# 使用sklearn的make_classification生成不平衡数据样本
from collections import Counter

from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification

# 生成一组0和1比例为9比1的样本，X为特征，y为对应的标签
X, y = make_classification(n_classes=2, class_sep=2,
                           weights=[0.9, 0.1], n_informative=3,
                           n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=1000, random_state=10)

print(X[:2, :])
# 查看所生成的样本类别分布，0和1样本比例9比1，属于类别不平衡数据
print(Counter(y))
print(type(y))
smo = SMOTE(random_state=42)
X_smo, y_smo = smo.fit_sample(X, y)

print(Counter(y_smo))
