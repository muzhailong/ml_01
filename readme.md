# 天池新人赛【快来一起挖掘幸福感!】
<br></br>
网址：https://tianchi.aliyun.com/competition/entrance/231702/information
<hr>

## 数据清洗
1. happiness出现异常值,比较少,把出现异常的行删除
2. 缺失值 使用中值代替
3. 异常值 使用IsolationForest 删除异常数据
## 特征处理
1. 连续值处理
    1. seaborn直方图统计手动分割
    2. 根据客观判断离散化,比如年龄通过10年一代来划分
    3. 对数据进行变换(比如log2处理)在用1进行处理
2. 特征选择
    1. 通过xgboost查看每一个特征的importance,然后删除不重要的特征

## 算法选择
1. XGBoost
    <br></br>
    <pre>
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
                             min_child_weight=4)
    </pre>
    主要调的参数有:
    1. <code>n_estimators</code>    
    2. <code>max_depth</code>
    3. <code>gamma</code>
    4. <code>min_child_weight</code>
    5. <code>learning_rate</code>
    <br></br>
    可以使用类似下面的代码逐个调(因为联合调试需要很长时间,对及其性能有很大的要求)
    <pre>
    param = {"max_depth": [6, 8, 10]}
    gsc = GridSearchCV(estimator=clf3,
                        param_grid=param,
                        n_jobs=8,
                        scoring=my_score_func)
    gsc.fit(train_x, train_y)
    print(gsc.grid_scores_)
    print(gsc.best_params_, gsc.best_score_)
    </pre>
    
    本地测试结果:0.5471392587261605
    
2. RandomForestClassifier<br></br>
    待续
3. StackingClassifier<br></br>
    待续
