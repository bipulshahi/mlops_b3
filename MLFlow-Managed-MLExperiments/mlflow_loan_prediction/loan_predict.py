import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import mlflow
import os

#load dataset
dataset = pd.read_csv('loanTrain.csv')
numerical_cols = dataset.select_dtypes(include='number').columns.to_list()    #list of numerical column names
categorical_cols = dataset.select_dtypes(exclude='number').columns.to_list()  #list of non numerical column names
categorical_cols.remove('Loan_ID')
categorical_cols.remove('Loan_Status')

#Fill missing values in categorical using mode
for col in categorical_cols:
    dataset[col].fillna(dataset[col].mode()[0], inplace=True)


#Fill missing values in numerical using mean
for col in numerical_cols:
    dataset[col].fillna(dataset[col].median(), inplace=True)

# Take care of outliers
dataset[numerical_cols] = dataset[numerical_cols].apply(lambda x: x.clip(*x.quantile([0.05,0.95])))


#Log transformation
dataset['LoanAmount'] = np.log(dataset['LoanAmount'])
dataset['ApplicantIncome'] = dataset['ApplicantIncome'] + dataset['CoapplicantIncome']
dataset['ApplicantIncome'] = np.log(dataset['ApplicantIncome'])


#Drop Co-applicant income
dataset = dataset.drop(columns = ['CoapplicantIncome'])


#Label encoding of non numerical values
for col in categorical_cols:
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col])

#Encode the target column
dataset['Loan_Status'] = le.fit_transform(dataset['Loan_Status'])


#Train test split
X = dataset.drop(['Loan_ID','Loan_Status'] , axis = 'columns')
y = dataset['Loan_Status']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=5)

#Random forest
rf_model = RandomForestClassifier()
param_grid_forest = {
    'n_estimators' : [50,100,200],
    'max_depth' : [10,20,30],
    'criterion': ['gini','entropy'],
    'max_leaf_nodes':[50,75]
}

grid_forest = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid_forest,
    cv=5,
    scoring='accuracy'
)

model_forest = grid_forest.fit(X_train,y_train)

#Logistic regression
lr_model = LogisticRegression()
param_grid_log = {
    'C':[100,10,1.0,0.1,0.01],
    'penalty':['l1','l2'],
    'solver':['liblinear']
}

grid_log = GridSearchCV(
    estimator=lr_model,
    param_grid=param_grid_log,
    cv=5,
    scoring='accuracy'
)
model_log = grid_log.fit(X_train,y_train)


#Decision tree
dt_model = DecisionTreeClassifier()

param_grid_tree = {
    'max_depth' : [10,20,30],
    'criterion': ['gini','entropy']
}

grid_tree = GridSearchCV(
    estimator=dt_model,
    param_grid=param_grid_tree,
    cv=5,
    scoring='accuracy'
)
model_tree = grid_tree.fit(X_train,y_train)


mlflow.set_tracking_uri("http://127.0.0.1:5000")

#Model evaluation metrics
def eval_metrics(actual,pred):
    accuracy = metrics.accuracy_score(actual,pred)
    f1 = metrics.f1_score(actual,pred)
    fpr,tpr,_ = metrics.roc_curve(actual,pred)
    auc = metrics.auc(fpr,tpr)

    plt.figure(figsize=(8,8))
    plt.plot(fpr,tpr,color='blue',label='ROC curve area =%0.2f'%auc)

    plt.xlabel('False Positive Rate', size=14)
    plt.ylabel('True Positive Rate', size=14)
    plt.legend()
    #Save plot
    os.makedirs("plots",exist_ok=True)
    plt.savefig("plots/ROC_curve.png")
    plt.close()
    return (accuracy,f1,auc)




#mlflow logging
def mlflow_logging(model,X,y,name):
    mlflow.set_experiment("ML_loan_exp")
    with mlflow.start_run(run_name=name):
        pred = model.predict(X)
        #metirics
        (accuracy,f1,auc) = eval_metrics(y,pred)
        #Log best parameters from grid search
        mlflow.log_params(model.best_params_)
        #Log the metrics
        mlflow.log_metric("Mean CV score", model.best_score_)
        mlflow.log_metric("Accuracy",accuracy)
        mlflow.log_metric("F1 Score",f1)
        mlflow.log_metric("AUC",auc)

        #Logging artifacts
        mlflow.log_artifact("plots/ROC_curve.png")
        mlflow.sklearn.log_model(model,name)

        mlflow.end_run()



mlflow_logging(model_tree,X_test,y_test,"DecisionTreeClassifier")
mlflow_logging(model_log,X_test,y_test,"LogisticRegression")
mlflow_logging(model_forest,X_test,y_test,"RandomForestClassifier")