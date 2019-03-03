import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import ADASYN
from mlxtend.classifier import StackingClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt

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
    e = np.log2(e)
    if e < 4:
        return 0
    if e < 6:
        return 1
    if e < 7:
        return 2
    if e < 8:
        return 3
    return 4


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
        e = 0
    if e >= 9:
        e = 9
    e = e // 2
    return e


def del_house(e):
    e = int(e)
    if e > 5 and e <= 10:
        e = 6
    if e > 10:
        e = 7
    return e


def del_family_income(e):
    e = np.log2(e + 1)

    if e < 10:
        return 0
    if e < 14:
        return 1
    if e < 17:
        return 2
    if e < 20:
        return 3
    return 4


def del_child_num(e):
    if e >= 4:
        e = 4
    return int(e)


def del_100_score(e):
    if e < 40:
        return 0
    if e < 60:
        return 1
    if e < 80:
        return 2
    if e < 90:
        return 3
    return 4


def del_s_income(e):
    e = np.log2(e + 1)
    if e < 8:
        return 0
    if e < 13:
        return 1
    if e < 16:
        return 2
    return 3


def myFeval(preds, xgbtrain):
    label = xgbtrain.get_label()
    score = mean_squared_error(label, preds)
    return 'myFeval', 1 - score


def my_score_func(estimator, x, y):
    return 1 - mean_squared_error(estimator.predict(x), y)


def drop_columns(df, lt):
    df = df.drop(lt, axis=1)
    return df


def df_columns_map(df):
    df["birth"] = df["birth"].apply(del_year)
    df["floor_area"] = df["floor_area"].apply(del_floor_area)
    df["height_cm"] = df["height_cm"].apply(del_height_cm)
    df["weight_jin"] = df["weight_jin"].apply(del_weight_jin)
    df["work_yr"] = df["work_yr"].apply(del_work_yr)
    df["family_m"] = df["family_m"].apply(del_family_m)
    df["house"] = df["house"].apply(del_house)
    df["family_income"] = df["family_income"].apply(del_family_income)
    df["income"] = df["income"].apply(del_family_income)
    df["son"] = df["son"].apply(del_child_num)
    df["daughter"] = df["daughter"].apply(del_child_num)
    df["minor_child"] = df["minor_child"].apply(del_child_num)
    df["s_income"] = df["s_income"].apply(del_s_income)

    sub_drop_columns = ["public_service_1", "public_service_2",
                        "public_service_3", "public_service_4",
                        "public_service_5", "public_service_6",
                        "public_service_7", "public_service_8", "public_service_9"]
    for col in sub_drop_columns:
        df[col] = df[col].apply(del_100_score)

    return df.astype("int64")


def feature_choose(X):
    return np.delete(X,
                     [74, 17, 11, 73, 94,
                      70, 72, 18, 16, 64,
                      59], axis=1)


def rf_clf(train_x, train_y):
    clf1 = RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                  min_samples_split=20,
                                  bootstrap=True, n_jobs=8)
    # param = {"n_estimators": [80], "min_samples_split": [20, 30, 40], "min_samples_leaf": [15, 20, 25],
    #          "max_depth": [12, 15, 18, 21]}
    # gsc = GridSearchCV(estimator=clf1,
    #                    param_grid=param,
    #                    n_jobs=8,
    #                    scoring=my_score_func, cv=5)
    # gsc.fit(train_x, train_y)
    # print(gsc.grid_scores_)
    # print(gsc.best_params_, gsc.best_score_)
    clf1.fit(train_x, train_y)
    return clf1


def svc_clf(train_x, train_y):
    clf2 = svm.SVC(C=10)
    clf2.fit(train_x, train_y)
    return clf2


def xgb_clf(train_x, train_y):
    clf3 = xgb.XGBClassifier(n_estimators=500,
                             learning_rate=0.1,
                             n_jobs=8,
                             objective='multi:softmax',
                             max_depth=8,
                             subsample=0.8,
                             colsample_bytree=0.8,
                             reg_lambda=1,
                             reg_alpha=0,
                             feval=myFeval,
                             maximize=True,
                             gamma=0.1,
                             min_child_weight=4
                             )
    # param = {"max_depth": [6, 8, 10]}
    # gsc = GridSearchCV(estimator=clf3,
    #                    param_grid=param,
    #                    n_jobs=8,
    #                    scoring=my_score_func)
    # gsc.fit(train_x, train_y)
    # print(gsc.grid_scores_)
    # print(gsc.best_params_, gsc.best_score_)
    clf3.fit(train_x, train_y)
    #
    # # fig = plt.figure(figsize=(10, 10))
    # # ax = fig.subplots()
    # xgb.plot_tree(clf3, num_trees=1)
    # # clf3.fit(train_x, train_y)
    # plt.show()
    return clf3


def gbdt_clf(train_x, train_y):
    clf4 = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1,
                                      min_samples_leaf=4, max_features="sqrt")
    clf4.fit(train_x, train_y)
    return clf4


def stacking_clf(train_x, train_y):
    clf1 = RandomForestClassifier(n_estimators=300,
                                  max_features="sqrt",
                                  min_samples_split=20,
                                  min_samples_leaf=15,
                                  max_depth=6,
                                  bootstrap=True, n_jobs=8)
    clf2 = svm.SVC(C=10)
    clf3 = xgb.XGBClassifier(n_estimators=300,
                             learning_rate=0.1,
                             n_jobs=8,
                             object="multi:softmax",
                             colsample_bylevel=0.8,
                             reg_lambda=1,
                             max_depth=6,
                             min_child_weight=1)

    clf4 = GradientBoostingClassifier(n_estimators=300,
                                      learning_rate=0.1,
                                      min_samples_split=20,
                                      min_samples_leaf=15,
                                      max_depth=6,
                                      max_features="sqrt")

    clf5 = LogisticRegression(penalty='l2', C=100, multi_class='ovr')

    sclf = StackingClassifier(
        classifiers=[clf1, clf3, clf4],
        meta_classifier=clf5,
    )
    sclf.fit(train_x, train_y)
    return sclf


def output(path, df):
    df.to_csv(path, sep=",", index=0)


df = pd.read_csv("../dataset/happiness_train_complete.csv", index_col=0, encoding="utf8")
test_df = pd.read_csv("../dataset/happiness_test_complete.csv", index_col=0, encoding="gbk")

drop_columns_list = ["survey_time", "city", "county", "religion",
                     "edu_other", "edu_yr", "join_party", "invest_other",
                     "property_other", "marital_1st", "s_birth",
                     "marital_now", "f_birth", "m_birth", "inc_exp"]

df = drop_columns(df, drop_columns_list)
test_df = drop_columns(test_df, drop_columns_list)
# 抛弃异常的分类
drop_indexes = df[df["happiness"] < 0].index
df.drop(drop_indexes, inplace=True)
df[df < 0] = np.nan
df.fillna(df.median(), inplace=True)

test_df[test_df < 0] = np.nan
test_df.fillna(df.median(), inplace=True)

df = df_columns_map(df)
test_df = df_columns_map(test_df)
test_index = test_df.index

data = df.values
test_data_x = test_df.values

X = data[:, 1:]
Y = data[:, 0]

# 数据清洗 异常检测
isf = IsolationForest()
isf.fit(X)
isf_res = isf.predict(X)
inner_index = (np.where(isf_res == 1))[0]
X = X[list(inner_index), :]
Y = Y[list(inner_index)]

# 特征选择
X = feature_choose(X)
test_data_x = feature_choose(test_data_x)

# 平衡数据
# x_resampled, y_resampled = SMOTE(kind="borderline1").fit_sample(X, Y)
x_resampled, y_resampled = ADASYN().fit_sample(X, Y)
# print(Counter(y_resampled))

# borderline1'``, ``'borderline2'``, ``'svm'`
train_x, test_x, train_y, test_y = train_test_split(x_resampled, y_resampled, test_size=0.25)
#
# clf1 = rf_clf(train_x, train_y)
# clf2 = svc_clf(train_x, train_y)
clf3 = xgb_clf(train_x, train_y)

digraph = xgb.to_graphviz(clf3, num_trees=1, rankdir="LR")
digraph.format = 'png'
digraph.view('./iris_xgb')
# fig = plt.gcf()
# fig.set_size_inches(150, 100)
# fig.savefig("tree.png")
# plt.show()

# xgb.to_graphviz(clf3, num_trees=0)
# plt.show()
# clf4 = gbdt_clf(train_x, train_y)
# clf5 = stacking_clf(train_x, train_y)
# predict_y = clf5.predict(test_x)
#
# plt.scatter(predict_y, test_y, marker='.')
# plt.show()
#
# sns.distplot(np.abs(predict_y - test_y))
#
# plt.show()
#
print(mean_squared_error(clf3.predict(test_x), test_y))
fig, ax = plt.subplots(figsize=(15, 15))
xgb.plot_importance(clf3, ax=ax, max_num_features=140, height=0.5)
plt.show()

#
# clf_list = [clf1, clf2, clf3, clf4, clf5]
# for clf in clf_list:
#     predict_y = clf.predict(test_x)
#     print(mean_squared_error(predict_y, test_y))

#
# clf3 = stacking_clf(train_x, train_y)
# test_data_y = clf3.predict(test_data_x)
# output("6.csv", pd.DataFrame(data={"id": test_index, "happiness": test_data_y}))
# xgb_clf(train_x, train_y)
