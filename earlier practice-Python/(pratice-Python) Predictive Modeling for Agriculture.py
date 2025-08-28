#Importing all required libraries 
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

#Loading the dataset
crops = pd.read_csv("soil_measures.csv")

#Checking for missing values
print(crops.isna().sum())
#there are no missing values

#checking unique values for crop types
print(crops['crop'].value_counts())

#splitting the data
y=crops['crop']
X=crops.drop('crop', axis=1)
X_train, X_test, y_train, y_test=train_test_split(X, y, stratify=y, random_state=42)

#creating dictionary for storing features performance 
features_dictionary={}

#instanciting Logistic Regressor
log_reg=LogisticRegression(multi_class='multinomial')

#creating a for loop
for feature in list(X_train.columns):
    log_reg.fit(X_train[[feature]], y_train)
    y_pred=log_reg.predict(X_test[[feature]])
    feature_performance=metrics.f1_score(y_test, y_pred, average='weighted')
    features_dictionary[feature] = feature_performance
    print(f"F1-score for {feature}: {feature_performance}")

#choosing the best predictive feature
best_predictive_feature={(max(features_dictionary, key=features_dictionary.get)):max(features_dictionary.values())}
print(best_predictive_feature)