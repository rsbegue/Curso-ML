# Kaggle: Titanic competition

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
train = pd.read_csv('../raw_data/train.csv')
test = pd.read_csv('../raw_data/test.csv')
X_train = train.iloc[:, [2,3,4,5,6,7,9]].values
y_train = train.iloc[:, 1].values
X_test = test.iloc[:, [1,2,3,4,5,6,8]].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 1)
imputer = imputer.fit(X_train[:, 3])
X_train[:, 3] = imputer.transform(X_train[:, 3])
X_test[:,[3,6]] = imputer.fit(X_test[:, [3,6]]).transform(X_test[:, [3,6]])

# Getting the titles from names in training set
titles_train = []
for name_train in X_train[:, 1]:
    st = name_train.split(",")[1].split(".")[0]
    titles_train.append(st)
X_train[:,1] = titles_train

# Getting the titles from names in test set
titles_test = []
for name_test in X_test[:, 1]:
    st = name_test.split(",")[1].split(".")[0]
    titles_test.append(st)
X_test[:,1] = titles_test

X_train = pd.DataFrame(X_train).replace('Mmle','Miss').replace('Ms','Miss').replace('Mme','Miss').replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
X_test = pd.DataFrame(X_test).replace('Mmle','Miss').replace('Ms','Miss').replace('Mme','Miss').replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')


for dataset in X_train[1]:
    dataset[1] = dataset[1].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset[1] = dataset[1].replace('Mlle', 'Miss')
    dataset[1] = dataset[1].replace('Ms', 'Miss')
    dataset[1] = dataset[1].replace('Mme', 'Mrs')
   
    
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_train[:, 1] = labelencoder_X.fit_transform(X_train[:, 1])
X_train[:, 2] = labelencoder_X.fit_transform(X_train[:, 2])
X_train = OneHotEncoder(categorical_features=[1]).fit_transform(X_train).toarray()
X_train = pd.DataFrame(X_train)

X_test[:, 1] = labelencoder_X.transform(X_test[:, 1])
X_test[:, 2] = labelencoder_X.transform(X_test[:, 2])
X_test = OneHotEncoder(categorical_features=[1]).transform(X_test).toarray()
X_test = pd.DataFrame(X_test)

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

# Creating Classifiers
clf_svm = SVC(kernel='linear', random_state=42).fit(X_train, y_train)
clf_RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42).fit(X_train, y_train)
clf_LR = LogisticRegression(random_state = 42).fit(X_train, y_train)
clf_knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski').fit(X_train, y_train)

clf_svm.score(X_train, y_train)
# Predicting for test set
y_pred_svm = clf_svm.predict(X_test)
y_pred_RF = clf_RF.predict(X_test)
y_pred_LR = clf_LR.predict(X_test)
y_pred_knn = clf_knn.predict(X_test)

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
y_pred_gs_svm = grid_search.predict(X_test)

results_svm = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": y_pred_svm}).to_csv("../results/results_svm.csv", index=None)
results_RF = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": y_pred_RF}).to_csv("../results/results_RF.csv", index=None)
results_LR = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": y_pred_LR}).to_csv("../results/results_LR.csv", index=None)
results_knn = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": y_pred_knn}).to_csv("../results/results_knn.csv", index=None)
results_gs_svm = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": y_pred_gs_svm}).to_csv("../results/results_gs_svm.csv", index=None)