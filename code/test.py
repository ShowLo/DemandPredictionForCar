import matplotlib.pyplot as plt
from xgboost.sklearn import XGBRegressor
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

x = np.random.uniform(0, 2 * np.pi, (1000, 1))
y = 10 * np.sin(x) + np.random.normal(0, 1, (1000, 1))

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=0)


model2 = XGBRegressor(
    learn_rate = 0.1,
    max_depth = 2,
    min_child_weight = 3,
    gamma = 0,
    subsample = 0.8,
    colsample_bytree = 0.7,
    reg_alpha = 1,
    objective = 'reg:linear',
    n_estimators = 100
)
watchlist2 = [(xTrain,yTrain),(xTest,yTest)]
model2.fit(xTrain,yTrain,eval_set=watchlist2,early_stopping_rounds=10)

result2 = model2.predict(xTest,ntree_limit=model2.best_iteration)
mse2 = mean_squared_error(yTest,result2)

print(mse2)
plt.scatter(xTest, yTest, marker = '.')
plt.scatter(xTest, result2, marker = 'x')
plt.show()