from sklearn.pipeline import Pipeline
from Myloanapp.config import config
import Myloanapp.processing.preprocessing as pp
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import numpy as np


#pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])

classification_pipeline = Pipeline([
    ('Mean Imputation' , pp.MeanImputer(config.NUM_FEATURES)),
    ('Mode Imputation' , pp.ModeImputer(config.CAT_FEATURES)),
    ('DomainProcessing' , pp.DomainProcessing(variables_to_modify=config.FEATURES_TO_MODIFY,variables_to_add=config.FEATURES_TO_ADD)),
    ('Drop Features', pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
    ('LabelEncoder' , pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
    ('LogTransform' , pp.LogTransform(variables=config.LOG_FEATURES)),
    ('MinMaxScale' , MinMaxScaler()),
    ('LogiscticClassifier' , LogisticRegression())
])