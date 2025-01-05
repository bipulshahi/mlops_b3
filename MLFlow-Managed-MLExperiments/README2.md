# MLFLOW - It's an open source platform to manage the ML lifecycle, including experimentation, reproducibility, deployment and a central model registry.

* MLflow Tracking - Records and query experiments: code, data, config, and results.
* Mlflow Projects - Package data science code in a format to reproduce runs on any platform
* Mlflow models - Deploy ml models in diverse serving environments.
* Mlflow Registry - Store, annotate, discover, and mange models in a central repository.

# Feature
* It offers data scientists the flexibilty to conduct numerous experiments before moving a model to production.
* It records crucial model evaluation metrics like RMSE and AUC, while also maintaining a log of hyperparameters employed during model development.

## Install miniconda
'''
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o miniconda.exe
start /wait "" .\miniconda.exe /S
del miniconda.exe
'''

## virtual environment
Create virual environment with Conda

'''python
* conda create -n mlflow-venv python=3.10
'''

'''
* conda activate mlflow-venv
'''

'''
* pip install mlflow
'''

'''
* mlflow ui
'''