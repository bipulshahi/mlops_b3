import os
import mlflow
import argparse
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")

def load_data():
    URL='https://github.com/bipulshahi/Dataset/raw/refs/heads/main/winequality-red.csv'
    try:
        df = pd.read_csv(URL,sep=';')
        return df
    except Exception as e:
        raise e

def eval_function(actual,pred):
    rmse = (mean_squared_error(actual,pred))**0.5
    mae = mean_absolute_error(actual,pred)
    r2 = r2_score(actual,pred)
    return rmse,mae,r2


def main(alpha,l1_ratio):
    df = load_data()
    TARGET = "quality"
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=5)

    mlflow.set_experiment("ML_Exp")
    with mlflow.start_run():
        mlflow.set_tag("version","1.0.0")
        mlflow.log_param("alpha",alpha)
        mlflow.log_param("l1_ratio",l1_ratio)

        model = ElasticNet(alpha=alpha , l1_ratio=l1_ratio)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        rmse,mae,r2 = eval_function(y_test,y_pred)

        mlflow.log_metric("rmse",rmse)
        mlflow.log_metric("mae",mae)
        mlflow.log_metric("r2",r2)
        mlflow.sklearn.log_model(model,"trained_model")   #model, foldername


if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--alpha","-a",type=float,default=0.2)
    args.add_argument("--l1_ratio","-l1",type=float,default=0.3)
    parsed_args = args.parse_args()
    main(parsed_args.alpha,parsed_args.l1_ratio)