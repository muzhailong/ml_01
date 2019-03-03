from collections import Counter

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, roc_curve, auc
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import math
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

import seaborn as sns
from mlxtend.classifier import StackingClassifier

pd.set_option("display.width", 500)
pd.set_option("display.max_columns", 200)

np.set_printoptions(threshold=np.nan)


def del_year(e):
    e = int(e)
    if e < 1920:
        e = 1920
    if e > 1990:
        e = 1990
    e -= 1920
    return int((e % 100) // 10)


def del_floor_area(e):
    e = int(e)
    if e < 200:
        e = 0
    if e > 600:
        e = 700
    return e // 200


def del_height_cm(e):
    e = int(e)
    if e < 140:
        e = 130
    if e > 190:
        e = 190
    return (e - 130) // 10


def del_weight_jin(e):
    e = int(e)
    if e < 50:
        e = 0

    if e > 200:
        e = 200
    return e // 50


def del_work_yr(e):
    e = int(e)
    if e > 50:
        e = 50
    return e // 10


def del_family_m(e):
    e = int(e)
    if e <= 0:
        e = 1
    if e >= 9:
        e = 9
    return e


def del_house(e):
    e = int(e)
    if e > 5 and e <= 10:
        e = 6
    if e > 10:
        e = 7
    return e


def del_family_income(e):
    e = int(e)
    if e < 1e4:
        return 0
    if e < 5e4:
        return 1
    if e < 1e5:
        return 2
    if e < 5e5:
        return 3
    return 4


def myFeval(preds, xgbtrain):
    label = xgbtrain.get_label()
    score = mean_squared_error(label, preds)
    return 'myFeval', score


def my_score_func(estimator, x, y):
    return 1 - mean_squared_error(estimator.predict(x), y)


df = pd.read_csv("../dataset/happiness_train_complete.csv", index_col=0)

# 抛弃调查时间
df = df.drop(["survey_time", "city", "county", "floor_area"], axis=1)
print(df.head())

# 抛弃异常的分类
drop_indexes = df[df["happiness"] < 0].index
df.drop(drop_indexes, inplace=True)
df[df < 0] = np.nan

df.fillna(df.mode().iloc[0], inplace=True)
# df.fillna(df.mean(), inplace=True)

# corr_matrix = df.corr()
# print(corr_matrix)

# plt.scatter(df["floor_area"], df["happiness"])
# sns.distplot(np.log2(df["floor_area"]))
# plt.figure()
# temp = df[(df["floor_area"] < 25) & (df["floor_area"] > 0)]
# plt.scatter(df["province"], df["happiness"], marker='.')
# plt.show()

df["birth"] = df["birth"].apply(del_year)

# df["floor_area"] = df["floor_area"].apply(del_floor_area)

df["height_cm"] = df["height_cm"].apply(del_height_cm)

df["weight_jin"] = df["weight_jin"].apply(del_weight_jin)

df["work_yr"] = df["work_yr"].apply(del_work_yr)

df["family_m"] = df["family_m"].apply(del_family_m)

df["house"] = df["house"].apply(del_house)

df["family_income"] = df["family_income"].apply(del_family_income)
df["income"] = df["income"].apply(del_family_income)
df = df.astype("int64")

data = df.values
X = data[:, 1:]
Y = data[:, 0]

x_resampled, y_resampled = SMOTE(kind="borderline1").fit_sample(X, Y)

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.25)
n_features = (train_x.shape)[1]
clf1 = RandomForestClassifier(n_estimators=500, max_features=int(math.sqrt(n_features)),
                              min_samples_split=20,
                              bootstrap=True, n_jobs=8, random_state=0)
#
# other_param = {
#     "n_estimators": [500, 700, 1000]
#     # "max_depth": [10, 15, 20],
#     # "min_samples_split": [20, 30],
#     # "min_samples_leaf": [5, 10]
# }
# gsearch = GridSearchCV(
#     estimator=clf1,
#     param_grid=other_param,
#     n_jobs=8,
#     scoring=my_score_func
# )
#
# gsearch.fit(train_x, train_y)
#
# print(gsearch.grid_scores_)
# print(gsearch.best_params_, gsearch.best_score_)
# print(gsearch.best_estimator_)


clf2 = svm.SVC(random_state=0)

clf3 = XGBClassifier(n_estimators=500, learning_rate=0.1, n_jobs=8
                     , colsample_bylevel=0.8, reg_lambda=1, min_child_weight=1, random_state=0)

clf4 = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1,
                                  min_samples_leaf=4, max_features="sqrt", random_state=0)

clf5 = GaussianNB()

clf6 = LogisticRegression(penalty='l2', C=70, multi_class='ovr', random_state=0)

# cv_params = {'n_estimators': [400, 500, 600, 700, 800], "learning_rate": [0.1, 0.15, 0.2, 0.25, 0.3]}
#
# other_params = {'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
#                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,
#                 "feval": myFeval}
#
# model = XGBClassifier(**other_params)
# optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, cv=5, verbose=1,
#                              n_jobs=8)
# optimized_GBM.fit(train_x, train_y)
# evalute_result = optimized_GBM.grid_scores_
# print('每轮迭代运行结果:{0}'.format(evalute_result))
# print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
# print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))


sclf = StackingClassifier(
    classifiers=[clf1, clf3, clf4],
    meta_classifier=clf6,
)

sclf.fit(train_x, train_y)

clf1.fit(train_x, train_y)
clf2.fit(train_x, train_y)
clf3.fit(train_x, train_y)
clf4.fit(train_x, train_y)
clf5.fit(train_x, train_y)
clf6.fit(train_x, train_y)

predict_test_y1 = clf1.predict(test_x)
predict_test_y2 = clf2.predict(test_x)
predict_test_y3 = clf3.predict(test_x)
predict_test_y4 = clf4.predict(test_x)
predict_test_y5 = clf5.predict(test_x)
predict_test_y6 = clf6.predict(test_x)
predict_test_sclf = sclf.predict(test_x)

# print(test_y[test_y != predict_test_y3])

# error_indexes = [i for i in range(len(test_y)) if test_y[i] != predict_test_y3[i]]
# error_data = df.iloc[error_indexes]
# print(error_data)

print("mean_squared_error")
print(mean_squared_error(predict_test_y1, test_y))
print(mean_squared_error(predict_test_y2, test_y))
print(mean_squared_error(predict_test_y3, test_y))
print(mean_squared_error(predict_test_y4, test_y))
print(mean_squared_error(predict_test_y5, test_y))
print(mean_squared_error(predict_test_y6, test_y))
print(mean_squared_error(predict_test_sclf, test_y))

# for cols in df:
#     sns.distplot(df[cols])
#     plt.show()

# sns.distplot(df["birth"])
plot_importance(clf3)
plt.show()
