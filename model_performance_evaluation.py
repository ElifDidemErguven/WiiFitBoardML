import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

# Load your preprocessed and standardized data
X = pd.read_csv("./Standardized Csv Files/features.csv")
Y = pd.read_csv("./Standardized Csv Files/labels.csv").squeeze()

X = np.asarray(X)
Y =np.asarray(Y)

#Comparing the models with default hyperparameter values using Cross-Validation

#List of Models
#Check if you still have to give max iter to Logistic Reg, because we are Standardizing the data!!!
#models = [LogisticRegression(max_iter=5000), XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5), SVC(kernel="linear"), SVC(kernel="rbf"), SVC(kernel="poly", degree=2), SVC(kernel="poly", degree=3), SVC(kernel="poly", degree=4),
#SVC(kernel="poly", degree=5),SVC(kernel="sigmoid"), KNeighborsClassifier(),RandomForestClassifier(random_state= 42), DecisionTreeClassifier(random_state=0),GaussianNB()]
models = [XGBClassifier(n_estimators=8, learning_rate=0.12, max_depth=2), GaussianNB()]

def compare_models_cross_validation():
    for model in models:
        cv_score = cross_val_score(model, X, Y, cv=5) # 5 folds for the cross-validation
        mean_accuracy = sum(cv_score)/len(cv_score)
        mean_accuracy = mean_accuracy*100
        mean_accuracy = round(mean_accuracy, 2) # take only two decimal after the comma

        if hasattr(model, 'kernel'):
            print(f"Cross Validation accuracies for the {model.__class__.__name__} with {model.kernel} kernel and degree {model.degree if model.kernel == 'poly' else ''} = {cv_score}")
            print(f"Accuracy score of the {model.__class__.__name__} with {model.kernel} kernel and degree {model.degree if model.kernel == 'poly' else ''} = {mean_accuracy} %")
        else:
            print(f"Cross Validation accuracies for the {model.__class__.__name__} = {cv_score}")
            print(f"Accuracy score of the {model.__class__.__name__} = {mean_accuracy} %")
        print("-------------------------------------------------")



compare_models_cross_validation()
