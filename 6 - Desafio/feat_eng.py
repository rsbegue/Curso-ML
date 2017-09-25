# Kaggle: Titanic competition

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Dropping unecessary features
train = train.drop(['PassengerId','Ticket','Cabin','Name','Embarked'], axis=1)
PassengerId = test['PassengerId']
test = test.drop(['Name','Ticket','Cabin','Name','Embarked','PassengerId'], axis=1)

# Filling missing values in the training set, with the most occurency
test['Fare'].fillna(test['Fare'].median(), inplace = True)
train['Age'].fillna(train['Age'].mean(), inplace = True)
test['Age'].fillna(test['Age'].mean(), inplace = True)

# Getting childs 
def get_person(passenger):
    age, sex = passenger
    return 'child' if age < 16 else sex

train['Person'] = train[['Age','Sex']].apply(get_person, axis=1)
test['Person'] = test[['Age','Sex']].apply(get_person, axis=1)

train.drop(['Sex'],axis = 1, inplace = True)
test.drop(['Sex'],axis = 1, inplace = True)

# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers
person_dummies_titanic  = pd.get_dummies(train['Person'])
person_dummies_titanic.columns = ['Child','Female','Male']
person_dummies_titanic.drop(['Male'], axis=1, inplace=True)

person_dummies_test  = pd.get_dummies(test['Person'])
person_dummies_test.columns = ['Child','Female','Male']
person_dummies_test.drop(['Male'], axis=1, inplace=True)

train = train.join(person_dummies_titanic)
test = test.join(person_dummies_test)

train.drop(['Person'], axis = 1, inplace = True)
test.drop(['Person'], axis = 1, inplace = True)

# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers
sns.factorplot('Pclass','Survived',order=[1,2,3], data=train,size=5)
pclass_dummies = pd.get_dummies(train['Pclass'])
pclass_dummies.columns = ['class1','class2','class3']
pclass_dummies.drop(['class3'], axis=1, inplace = True)

pclass_dummies_train = pd.get_dummies(train['Pclass'])
pclass_dummies_train.columns = ['class1','class2','class3']
pclass_dummies_train.drop(['class3'], axis=1, inplace = True)

pclass_dummies_test = pd.get_dummies(test['Pclass'])
pclass_dummies_test.columns = ['class1','class2','class3']
pclass_dummies_test.drop(['class3'], axis=1, inplace = True)

train.drop(['Pclass'], axis=1, inplace = True)
test.drop(['Pclass'], axis=1, inplace = True)

# Handling family members
train['Family'] = train['SibSp'] + train['Parch']
train['Family'].loc[train['Family'] > 0] = 1
train['Family'].loc[train['Family'] == 0] = 0

test['Family'] = test['SibSp'] + test['Parch']
test['Family'].loc[test['Family'] > 0] = 1
test['Family'].loc[test['Family'] == 0] = 0

train.drop(['SibSp','Parch'],axis=1, inplace=True)
test.drop(['SibSp','Parch'],axis=1, inplace=True)

# Getting the training data and target variable
y_train = train['Survived']
X_train = train.drop(['Survived'], axis=1)
X_test = test

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Importing libraries for classification
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Creating Classifiers
clf_et = ExtraTreesClassifier(**et_params).fit(X_train, y_train)
clf_ada = AdaBoostClassifier(**ada_params).fit(X_train, y_train)
clf_gb = GradientBoostingClassifier(**gb_params).fit(X_train, y_train)
clf_svm = SVC(kernel='linear', random_state=42).fit(X_train, y_train)
clf_RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42).fit(X_train, y_train)
clf_LR = LogisticRegression(random_state = 42).fit(X_train, y_train)
clf_knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski').fit(X_train, y_train)

# Predicting for training set
y_pred_train_et = clf_et.predict(X_train)
y_pred_train_ada = clf_ada.predict(X_train)
y_pred_train_gb = clf_gb.predict(X_train)
y_pred_train_svm = clf_svm.predict(X_train)
y_pred_train_RF = clf_RF.predict(X_train)
y_pred_train_LR = clf_LR.predict(X_train)
y_pred_train_knn = clf_knn.predict(X_train)

# Predicting for test set
y_pred_test_et = clf_et.predict(X_test)
y_pred_test_ada = clf_ada.predict(X_test)
y_pred_test_gb = clf_gb.predict(X_test)
y_pred_test_svm = clf_svm.predict(X_test)
y_pred_test_RF = clf_RF.predict(X_test)
y_pred_test_LR = clf_LR.predict(X_test)
y_pred_test_knn = clf_knn.predict(X_test)

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = clf_svm,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
y_pred_gs_svm_test = grid_search.predict(X_test)
y_pred_gs_svm_train = grid_search.predict(X_train)

#Using XGBoost to ensemble previous predictions
base_predictions_train = pd.DataFrame({'ET': y_pred_train_et,
                                 'ADA': y_pred_train_ada,
                                 'GB': y_pred_train_gb,
                                 'SVM': y_pred_train_svm,
                                 'RF': y_pred_train_RF,
                                 'LR': y_pred_train_LR,
                                 'KNN':y_pred_train_knn,
                                 'GS': y_pred_gs_svm_train})

base_predictions_test = pd.DataFrame({'ET': y_pred_test_et,
                                 'ADA': y_pred_test_ada,
                                 'GB': y_pred_test_gb,
                                 'SVM': y_pred_test_svm,
                                 'RF': y_pred_test_RF,
                                 'LR': y_pred_test_LR,
                                 'KNN':y_pred_test_knn,
                                 'GS': y_pred_gs_svm_test})

    
from xgboost import XGBClassifier
xgb = XGBClassifier(
    #learning_rate = 0.02,
    n_estimators= 2000,
    max_depth= 4,
    min_child_weight= 2,
    #gamma=1,
    gamma=0.9,                        
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread= -1,
    scale_pos_weight=1).fit(base_predictions_train, y_train)

final_prediction = xgb.predict(base_predictions_test)
results_xgboost = pd.DataFrame({"PassengerId": PassengerId, "Survived": final_prediction}).to_csv("../results/results_xgboost.csv", index=None)










def create_dummies(train):
	""" Function that create dummies for categorical variables
	Input: Dataframe
	Output: Dataframe with dummy variables
	"""
	dummies_total = pd.DataFrame()
	for col in train.columns:
		if train[col].dtype == 'object': # checking if col type is an object
			count = train[col].nunique() # couting the number of unique incidences
			if count > 5:
				train.drop([col], axis = 1, inplace = True) # dropping columns with high number of dummies
			else:
				dummies = pd.get_dummies(train[col]) # creating dummies
				dummies.drop(dummies.columns[1], axis=1, inplace = True) # avoiding the dummy variable trap
				train.drop([col], axis=1, inplace = True) # dropping the original column of dummies
				dummies_total = pd.concat([dummies_total, dummies], axis=1)
	train = pd.concat([train, dummies_total], axis=1)
	return train
train = create_dummies(train)