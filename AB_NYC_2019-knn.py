# *_*coding:utf-8 *_*
#
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

'''

Note:

email: modianorserver@gmail.com

If you have any questions, 
you can contact me through the above methods



数据的初步探索

1、获得数据的基本信息
Shape: (48895, 16)
Columns: [‘id’, ‘name’, ‘host_id’, ‘host_name’, ‘neighbourhood_group’, ‘neighbourhood’, ‘latitude’, ‘longitude’, ‘room_type’, ‘price’, ‘minimum_nights’, ‘number_of_reviews’, ‘last_review’, ‘reviews_per_month’, ‘calculated_host_listings_count’, ‘availability_365’]

2、获取数据类型
<class ‘pandas.core.frame.DataFrame’>
RangeIndex: 48895 entries, 0 to 48894
Data columns (total 16 columns):
id 48895 non-null int64
name 48879 non-null object
host_id 48895 non-null int64
host_name 48874 non-null object
neighbourhood_group 48895 non-null object
neighbourhood 48895 non-null object
latitude 48895 non-null float64
longitude 48895 non-null float64
room_type 48895 non-null object
price 48895 non-null int64
minimum_nights 48895 non-null int64
number_of_reviews 48895 non-null int64
last_review 38843 non-null object
reviews_per_month 38843 non-null float64
calculated_host_listings_count 48895 non-null int64
availability_365 48895 non-null int64
dtypes: float64(3), int64(7), object(6)
'''

'''
数据初步分析

1、数据包含48895个样本, 16个特征，包括房主和房间信息、房间所在区域、地理位置、房间类型、预定要求以及评论数量等信息。
16个特征分别为：
id: listing ID
name: name of the listing
host_id: host ID
host_name: name of the host
neighbourhood_group: location（区域）
neighbourhood ： area（地区）
latitude ： latitude coordinates（维度坐标）
longitude： longitude coordinates（经度坐标）
room_type： listing space type（房间类型）
price： price in dollars（订房价格）
minimum_nights： amount of nights minimum（最少预定天数）
number_of_reviews： number of reviews（评论数量）
last_review： latest review（最新评论时间）
reviews_per_month： number of reviews per month（每月评论数量）
calculated_host_listings_count： amount of listing per host（每个房主拥有的房间数）
availability_365： number of days when listing is available for booking（可以预定的天数）

2、缺失值的分析和处理
使用pandas生成各特征缺失值数据的DataFrame
reviews_per_month和last_review缺失严重，占样本数量的20%

分析：
有缺失值的特征包括：每月评论数量、最新评论时间、房间名、房主名。
1.房间名称、房主名的缺失相对较少，考虑房间和房东的名字对数据建模的贡献不大，所以将这两列删除掉。
每月评论数量、最新评论时间的缺失值相对较多，占到整个样本数量的20%，删除缺失值所在的样本会损失很多数据。
且从理论上来讲，评论数量的多少可能和房价之间存在着不可描述的关系，所以删除特征列也不妥。那就只能填补缺失值了。
再进一步分析，如果某些房间没有评论，那么自然最新评论时间和每月评论数量都会为0。
通过统计发现每月评论数量10052，所以考虑用0填补reviews_per_month 并用整个数据集最早的评论时间填补last_review
'''

'''
Minimum nights 分析
数据呈现非常严重的偏态分布，大多数房间要求入住最少天数的值较小。因此，使用描述性统计进行进一步观察
百分之99的数据都集中在45以内。对于严重的偏态数据，可以使用numpy中的log1p()函数进行处理，其中：np.log1p(x) = log(1+x)
使用np.log1p可以一定程度上缓解数据的偏态性，所以将Minimum nights数据进行转换

Reviews per month 分析
Reviews per month 数据的偏态非常严重，即使进行对数处理，也呈现出非常严重的偏态分布。
99%的样本，月平均评论数量在7条以内。这说明大多数的房源评论数量是非常少的，而只有少数样本具有很大的月评论量。这里推测：月评论数量可能和minimum_nights 以及后面的 availability_365存在相关性，因为如果房间的最短预定时间较长，完成订单的用户数量会相对较少，而一年之内如果可预订的时间较少，也会造成月评论数量较少。这一部分推论将在后面的分析中进一步探究。

Availability 365 分析
样本的可预订天数在15天内的数量较多，其余天数的数量都较少，但是分布没有呈现间断的偏态。

calculated_host_listings_count 分析
数据也存在一定程度的偏态。考虑房东拥有的房源数量与其他特征时间的联系并不直观，所以关于这一特征，在后续统计分析中再进一步讨论。

price 分析
讨论了所以特征的分布之后，接下来对目标列，房间价格的分布进行探索。
借用np.log1p函数 可以将房间价格数据的分布转换成近似的高斯分布。

最终选择了 key features ["neighbourhood", "latitude", "longitude", "room_type", "minimum_nights", "number_of_reviews"]

ok  既然确定了 key features ， 那么就可以 process data then to train and evaluate ... 
'''

raw_data = pd.read_csv("data/AB_NYC_2019.csv")
# 选取price <= 600 的数据，排除异常值 如过大值
df = raw_data[raw_data.price <= 600].copy()

####################
# pre-process data #
####################

# reviews_per_month: replace null with 0
df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

# last_review data: convert it to numeric value
df["last_review"] = pd.to_datetime(df["last_review"], infer_datetime_format=True)
earliest_last_review = min(df["last_review"])
df["last_review"] = df["last_review"].fillna(earliest_last_review)
df["last_review"] = df["last_review"].apply(
    lambda review_date: review_date.toordinal() - earliest_last_review.toordinal())

# neighbourhood: label encoding
neighbourhood_encoder = LabelEncoder()
neighbourhood_labels = neighbourhood_encoder.fit_transform(df["neighbourhood"])
df["neighbourhood"] = neighbourhood_labels
# retain the mapping of neighbourhood and encoded values
# neighbourhood_dict = dict(zip(neighbourhood_encoder.classes_, range(len(neighbourhood_encoder.classes_))))

# room_type: label encoding
room_encoder = LabelEncoder()
room_labels = room_encoder.fit_transform(df["room_type"])
df["room_type"] = room_labels
# retain the mapping of room_type and encoded values
# room_dict = dict(zip(room_encoder.classes_, range(len(room_encoder.classes_))))

# convert feature to log(1 + feature)
df["price"] = np.log1p(df["price"])

#######################
# select key features #
#######################

x = df[["neighbourhood", "latitude", "longitude", "room_type", "minimum_nights", "number_of_reviews"]]

y = list(df["price"])

# split data set to training data and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

# normalize features in order to better train our model
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

###############################
# train model and plot result #
###############################

scores = []
k_range = range(20, 60)

print('真实的价格：{} , {}'.format(y_test,len(y_test)))
for k in k_range:
    model = KNeighborsRegressor(n_neighbors=k, weights='distance')
    model.fit(x_train_scaled, y_train)
    y_pre = model.predict(x_test_scaled)  # 预测的价格
    # 如果你需要显示 只需要打印出来即可
    print('预测的价格：{} , {}'.format(y_pre,y_pre.shape))
    scores.append(model.score(x_test_scaled, y_test))

Best_K_Index = scores.index(max(scores)) + 20
model = KNeighborsRegressor(n_neighbors=Best_K_Index + 20, weights='distance')
model.fit(x_train_scaled, y_train)
joblib.dump(model, 'model/KNN-AB-NYC-2019.m')
print('********Knn Best Score is {}********'.format(model.score(x_test_scaled, y_test)))

plt.plot(k_range, scores)
plt.title("R2_score (knn)")
plt.xlabel("k")
plt.show()
