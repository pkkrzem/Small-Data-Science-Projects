import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor

# Loading and examining the dataset
rental_info_df = pd.read_csv("rental_info.csv")
print(rental_info_df.info())
print(rental_info_df.head())

#Creating a column of target variable
rental_info_df['rental_length']=pd.to_datetime(rental_info_df["return_date"])-pd.to_datetime(rental_info_df["rental_date"])
rental_info_df['rental_length_days']=rental_info_df['rental_length'].dt.days

#Creating columns of dummy variables
rental_info_df["deleted_scenes"] =  np.where(rental_info_df["special_features"].str.contains("Deleted Scenes"), 1,0)
rental_info_df["behind_the_scenes"] =  np.where(rental_info_df["special_features"].str.contains("Behind the Scenes"), 1,0)

#asigning target variable
y=rental_info_df['rental_length_days']

#asigning features variables
columns_to_drop = ["special_features", "rental_length", "rental_length_days", "rental_date", "return_date"]
X=rental_info_df.drop(columns_to_drop, axis=1)

#splitting the data
X_train, X_test, y_train, y_test=train_test_split(X, y, stratify=y, test_size=0.2, random_state=9)

#creating the Lasso model
lasso_reg=Lasso(alpha=0.3, random_state=9)

#training the model and acessing its coefficients
names=X.columns
lasso_coef=lasso_reg.fit(X_train, y_train).coef_
plt.bar(names, lasso_coef)
plt.xticks(rotation=45)
plt.show()

#performing feature selection by choosing columns with positive coefficients
X_lasso_train, X_lasso_test = X_train.iloc[:, lasso_coef > 0], X_test.iloc[:, lasso_coef > 0]

# Running Linear Regression model on lasso chosen regression
lin_reg = LinearRegression()
lin_reg = lin_reg.fit(X_lasso_train, y_train)
y_test_pred = lin_reg.predict(X_lasso_test)
mse_lin_reg_lasso = mean_squared_error(y_test, y_test_pred)
print(mse_lin_reg_lasso)

#creating various Regression models
forest_reg = RandomForestRegressor(n_estimators=100, random_state=9)
tree_reg=DecisionTreeRegressor(random_state=9)
svm_reg = LinearSVR(max_iter=100, tol=20, dual=True, random_state=9)
extratrees_reg = ExtraTreesRegressor(n_estimators=100, random_state=9)
mlp_reg = MLPRegressor(random_state=9)
regressors=[forest_reg, tree_reg, svm_reg, extratrees_reg, mlp_reg]

#looping through the models-trainning
for reg in regressors:
    print("Training the", reg)
    reg.fit(X_train, y_train)

#looping through the models-calculating the mse
for reg in regressors:
    y_test_pred = reg.predict(X_test)
    mse_reg = mean_squared_error(y_test, y_test_pred)
    print(reg," mean squared error:", mse_reg)

#parameters for chosen regressors
params = {'n_estimators': np.arange(1,101,1),
          'max_depth':np.arange(1,11,1)}

#hyperparameter tunnig for random forest regressor
print("Training the", forest_reg)
rand_search_CV = RandomizedSearchCV(forest_reg, 
                                 param_distributions=params, 
                                 cv=5, 
                                 random_state=9)
rand_search_CV.fit(X_train, y_train)

print("Best params", rand_search_CV.best_params_)
hyper_params_forest=rand_search_CV.best_params_

forest_reg=RandomForestRegressor(n_estimators=hyper_params_forest["n_estimators"], 
                           max_depth=hyper_params_forest["max_depth"], 
                           random_state=9)

#creating random forest regressor with the best params
print("Training the", forest_reg,"with best params")
forest_reg.fit(X_train, y_train)
y_test_pred_forest = forest_reg.predict(X_test)
mse_reg_forest = mean_squared_error(y_test, y_test_pred_forest)
print(forest_reg," mean squared error with best params:", mse_reg_forest)

#hyperparameter tunnig for extra trees regressor
print("Training the", extratrees_reg)
rand_search_CV = RandomizedSearchCV(extratrees_reg, 
                                 param_distributions=params, 
                                 cv=5, 
                                 random_state=9)
rand_search_CV.fit(X_train, y_train)

print("Best params", rand_search_CV.best_params_)
hyper_params_extratrees=rand_search_CV.best_params_

#creating extra trees regressor with the best params
extratrees_reg=ExtraTreesRegressor(n_estimators=hyper_params_extratrees["n_estimators"], 
                           max_depth=hyper_params_extratrees["max_depth"], 
                           random_state=9)

print("Training the", extratrees_reg,"with best params")
extratrees_reg.fit(X_train, y_train)
y_test_pred_extratrees = extratrees_reg.predict(X_test)
mse_reg_extratrees = mean_squared_error(y_test, y_test_pred_extratrees)
print(extratrees_reg," mean squared error with best params:", mse_reg_extratrees)

#the model with lowest MSE:
best_model = forest_reg
best_mse = mse_reg_forest