from Myloanapp.config import config
import numpy as np

#Numerical imputation - mean
class MeanImputer():
    def __init__(self,variables=None):
        self.variables = variables

    def fit(self,X,y=None):
        self.mean_dict = {}
        for var in self.variables:
            self.mean_dict[var] = X[var].mean()
        return self

    def transform(self,X):
        for var in self.variables:
            X[var] = X[var].fillna(self.mean_dict[var])
        return X


#Categorical imputation - mode
class ModeImputer():
    def __init__(self,variables=None):
        self.variables = variables

    def fit(self,X,y=None):
        self.mode_dict = {}
        for var in self.variables:
            self.mode_dict[var] = X[var].mode()[0]
        return self

    def transform(self,X):
        for var in self.variables:
            X[var] = X[var].fillna(self.mode_dict[var])
        return X

#Drop columns
class DropColumns():
    def __init__(self,variables_to_drop=None):
        self.variables_to_drop = variables_to_drop

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        X = X.drop(columns = self.variables_to_drop)
        return X

#Domain specific preprocessing
class DomainProcessing():
    def __init__(self,variables_to_modify=None,variables_to_add=None):
        self.variables_to_modify = variables_to_modify
        self.variables_to_add = variables_to_add

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        for feature in self.variables_to_modify:
            X[feature] = X[feature] + X[self.variables_to_add]
        return X


#Custom label encoder
class CustomLabelEncoder():
    def __init__(self,variables=None):
        self.variables = variables

    def fit(self,X,y=None):
        self.label_dict = {}
        for val in self.variables:
            t = X[val].value_counts().sort_values(ascending=True).index
            self.label_dict[val] = {cat : i for i,cat in enumerate(t)}
        return self

    def transform(self,X):
        for val in self.variables:
            X[val] = X[val].map(self.label_dict[val])
        return X


#Log transformation
class LogTransform():
    def __init__(self,variables=None):
        self.variables = variables

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        for val in self.variables:
            X[val] = np.log(X[val])
        return X


