from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import numpy as np
import json

from constants import IF_CONST, LOF_CONST, SGD_OC_SVM_CONST, ELL_ENV_CONST


def create_IF_model(random_state_val=0):
  if_model = IsolationForest(n_estimators=20, contamination=10**-12, random_state=random_state_val, n_jobs = -1)
  return IF_CONST, if_model

def create_LOF_model():
  lof_model = LocalOutlierFactor(contamination=10**-12, novelty = True, n_jobs = -1, n_neighbors = 20)
  return LOF_CONST, lof_model

def create_SGD_OC_SVM_model(random_state_val=0):
  model = SGDOneClassSVM( nu=10**-12, random_state=random_state_val)
  return SGD_OC_SVM_CONST, model

def create_Elliptic_envelope_model(random_state_val=0):
  ell_env_model = EllipticEnvelope(contamination=10**-12, random_state=random_state_val, support_fraction=1.0, store_precision=False)
  return ELL_ENV_CONST, ell_env_model


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
