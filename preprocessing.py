from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

from constants import NO_TRANSFORMATION_CONST, ROBUST_SCALER_TRANSFORMATION_CONST, POWER_TRANSFORMATION_CONST, QUANTILE_TRANSFORMATION_CONST


def no_transformation(X):
    return X

def robust_transformation(X):
    scaler = RobustScaler()
    data_scaled = scaler.fit_transform(X)
    return data_scaled

def power_transformation(X):
    scaler = PowerTransformer(method='yeo-johnson', standardize=True)
    data_scaled = scaler.fit_transform(X)
    return data_scaled

def quantile_transformer(X):
    scaler = QuantileTransformer(random_state=0, subsample=10000, output_distribution='uniform',n_quantiles=1000)
    data_scaled = scaler.fit_transform(X)
    return data_scaled

def preprocess_data_by_method_name(X, method_name):

    if method_name==NO_TRANSFORMATION_CONST:
        return no_transformation(X)

    if method_name==ROBUST_SCALER_TRANSFORMATION_CONST:
        return robust_transformation(X)

    if method_name==POWER_TRANSFORMATION_CONST:
        return power_transformation(X)

    if method_name==QUANTILE_TRANSFORMATION_CONST:
        return quantile_transformer(X)

    return None
