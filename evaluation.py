from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

from constants import IF_CONST, LOF_CONST, SGD_OC_SVM_CONST, ELL_ENV_CONST


def train_evaluate_model(x_train, x_test, y_train, y_test, model_name, model_obj):
    """
    Train a classifier and evaluate it.

    Parameters
    ----------
    x_train : array-like
        Training features
    x_test : array-like
        Testing features
    y_train : array-like
        Training labels
    y_test : array-like
        Testing labels
    positive_label : int/str
        Label considered as the positive class for metrics

    Returns
    -------
    dict
        Dictionary containing accuracy, recall, precision, and f1-score
    """

    # Create and train classifier
    model_obj.fit(x_train)


    # Predictions
    y_pred = model_obj.predict(x_test)
    y_pred_scores = model_obj.decision_function(x_test)

    # here change labels based on model format; 1 for anomaly, 0 for normal ones
    if model_name in [IF_CONST, LOF_CONST, SGD_OC_SVM_CONST, ELL_ENV_CONST]:
      y_pred[y_pred == 1] = 0  # replace normals (1) with 0
      y_pred[y_pred == -1] = 1  # replace anomalies (-1) with 1

    return y_pred, y_pred_scores


def compute_auroc_score(y_true, y_scores):
  f32 = np.finfo(np.float32)
  y_scores = np.nan_to_num(y_scores, nan=f32.max)
  return roc_auc_score(y_true, y_scores)

def compute_aupr_score(y_true, y_scores, pos_label_value):
  f32 = np.finfo(np.float32)
  y_scores = np.nan_to_num(y_scores, nan=f32.max)
  return average_precision_score(y_true, y_scores, pos_label = pos_label_value)
