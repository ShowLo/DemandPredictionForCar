import scipy.io as scio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.model_selection import GridSearchCV

#解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

if __name__ == "__main__":
    
    dataPath = 'data/demand_10.mat'
    data = scio.loadmat(dataPath)
    data = data['demand']
    (timeLength, regionNum) = data.shape

    #以前featureLength个时间片作为特征，后forecastLength个时间片作为需要预测的
    #用于预测的特征变量对应的时间片长度
    featureLength = 143
    #预测目标对应的时间片长度
    forecastLength = 1

    #测试集选择最后一天的数据
    testTimeSliceLength = 144

    #训练数据
    xTrain = np.zeros((regionNum * (timeLength - testTimeSliceLength - forecastLength - featureLength + 1), featureLength + 1))
    yTrain = np.zeros((regionNum * (timeLength - testTimeSliceLength - forecastLength - featureLength + 1), forecastLength))
    trainIndex = 0
    for regionID in range(regionNum):
        for timeSlice in range(timeLength - testTimeSliceLength - forecastLength - featureLength + 1):
            #把区域id作为特征变量的一维
            xTrain[trainIndex, :] = np.append(regionID, data[timeSlice : timeSlice + featureLength, regionID])
            yTrain[trainIndex, :] = data[timeSlice + featureLength : timeSlice + featureLength + forecastLength, regionID]
            trainIndex += 1

    #测试数据
    xTest = np.zeros((regionNum * (testTimeSliceLength - forecastLength + 1), featureLength + 1))
    yTest = np.zeros((regionNum * (testTimeSliceLength - forecastLength + 1), forecastLength))
    testIndex = 0
    for regionID in range(regionNum):
        for timeSlice in range(timeLength - testTimeSliceLength - featureLength, timeLength - forecastLength - featureLength + 1):
            xTest[testIndex, :] = np.append(regionID, data[timeSlice : timeSlice + featureLength, regionID])
            yTest[testIndex, :] = data[timeSlice + featureLength : timeSlice + featureLength + forecastLength, regionID]
            testIndex += 1

    '''
    #调参过程
    cv_params = {'learning_rate': [0.09, 0.1, 0.11]}
    other_params = {
        'learning_rate' : 0.1,          #学习率
        'n_estimators' : 100,           #最佳迭代次数
        'max_depth' : 6,                #树的最大深度
        'min_child_weight' : 8,         #最小叶子节点样本权重和，用于避免过拟合
        'gamma' : 0,                    #节点分裂所需的最小损失函数下降值，值越大，算法越保守
        'subsample' : 0.7,              #控制对于每棵树，随机采样的比例，值越小，算法越保守，避免过拟合，但如果过小，可能会导致欠拟合。
        'colsample_bytree' : 0.8,       #控制每棵随机采样的列数的占比(每一列是一个特征)
        'reg_alpha' : 0.05,             #权重的L1正则化项
        'reg_lambda' : 3,               #权重的L2正则化项
        'objective' : 'reg:linear'
    }

    #使用回归树的方法进行预测
    model = XGBRegressor(**other_params)

    #交叉验证，优化参数
    optimized_XGB = GridSearchCV(estimator = model, param_grid = cv_params, scoring = 'neg_mean_absolute_error', cv = 10, verbose = 1, n_jobs = 6)
    optimized_XGB.fit(xTrain, yTrain)

    print('每轮迭代运行结果:{0}'.format(optimized_XGB.grid_scores_))
    print('参数的最佳取值:{0}'.format(optimized_XGB.best_params_))
    print('最佳模型得分:{0}'.format(optimized_XGB.best_score_))
    '''

    model = XGBRegressor(
        learning_rate = 0.1,        #学习率
        n_estimators = 100,         #最佳迭代次数
        max_depth = 9,              #树的最大深度
        min_child_weight = 8,       #最小叶子节点样本权重和，用于避免过拟合
        gamma = 0,                  #节点分裂所需的最小损失函数下降值，值越大，算法越保守
        subsample = 0.7,            #控制对于每棵树，随机采样的比例，值越小，算法越保守，避免过拟合，但如果过小，可能会导致欠拟合。
        colsample_bytree = 0.8,     #控制每棵随机采样的列数的占比(每一列是一个特征)
        reg_alpha = 0.05,           #权重的L1正则化项
        reg_lambda = 2,             #权重的L2正则化项
        objective = 'reg:linear'
    )
    watchlist = [(xTrain, yTrain), (xTest, yTest)]
    model.fit(xTrain, yTrain, eval_set = watchlist, eval_metric = 'mae', early_stopping_rounds = 10)

    result = model.predict(xTest, ntree_limit = model.best_iteration)
    result = result.round()
    mse = MAE(yTest, result)
    print(mse)

    #plt.rcParams['font.sas-serig'] = ['SimHei']     #用来正常显示中文标签
    #plt.rcParams['axes.unicode_minus'] = False      #用来正常显示负号

    plt.plot(yTest[0 : 144], label = '真实值')
    plt.plot(result[0 : 144], color = 'red', label = '预测值')
    plt.xlabel('时间')
    plt.ylabel('需求量')
    plt.title('区域1第21天的需求量真实值与预测值(取整)')
    plt.legend(loc = 'upper right')
    plt.show()
    mse1 = MAE(yTest[0 : 144], result[0 : 144])
    print(mse1)
    from sklearn.metrics import mean_squared_error
    print('rmse1:' + str(mean_squared_error(yTest[0 : 144], result[0 : 144])))

    plt.plot(yTest[288 : 432], label = '真实值')
    plt.plot(result[288 : 432], color = 'red', label = '预测值')
    plt.xlabel('时间')
    plt.ylabel('需求量')
    plt.title('区域3第21天的需求量真实值与预测值(取整)')
    plt.legend(loc = 'upper right')
    plt.show()
    mse2 = MAE(yTest[288 : 432], result[288 : 432])
    print(mse2)
    print('rmse2:' + str(mean_squared_error(yTest[288 : 432], result[288 : 432])))

    # plt.plot(yTest[-288 : -144])
    # plt.plot(result[-288 : -144], 'r-')
    # plt.show()
    # mse2 = MAE(yTest[-288 : -144], result[-288 : -144])
    # print(mse2)

    plt.plot(yTest, label = '真实值')
    plt.plot(result, color = 'red', label = '预测值')
    plt.xlabel('时间(每144的时间长度为一个区域)')
    plt.ylabel('需求量')
    plt.title('所有区域第21天的需求量真实值与预测值(取整)')
    plt.legend(loc = 'upper right')
    plt.show()

    # plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
    # plt.show()