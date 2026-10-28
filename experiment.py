import numpy as np

from constants import IF_CONST, LOF_CONST, SGD_OC_SVM_CONST, ELL_ENV_CONST, AUROC_CONST, AUPR_CONST, MINORITY_CLASS_LABEL, MAJORITY_CLASS_LABEL
from models import create_IF_model, create_LOF_model, create_SGD_OC_SVM_model, create_Elliptic_envelope_model
from preprocessing import preprocess_data_by_method_name
from see_method import training_at_dataset_level, testing_at_dataset_level
from evaluation import train_evaluate_model, compute_auroc_score, compute_aupr_score


# X data and y labels; shuffle seed for deterministic shuffle; training_percentage param refers to the percentage of normal samples
# (not all, i.e., anomalous + normal) used in the training; e.g. a dataset contains 80% normals and 20% anomalies;
# if training_percentage = 0.5 then from the 80% of normal samples, half of them, i.e., 40% are used for training, rest for testing along with anomalies
# !!! this method uses np.copy, thus it does not produces side effects
def split_data_for_train_test(X_data, y_data, shuffle_seed = 0, training_percentage = 0.9):
  X = np.copy(X_data)
  y = np.copy(y_data)

  # consider 1 for anomaly, 0 for normal
  x_anomaly = np.array([X[i] for i in range(len(y)) if y[i] == 1])
  y_anomaly = np.array([y_val for y_val in y if y_val == 1])

  x_normal = np.array([X[i] for i in range(len(y)) if y[i] == 0])
  y_normal = np.array([y_val for y_val in y if y_val == 0])

  rng1 = np.random.default_rng(seed=shuffle_seed)
  rng2 = np.random.default_rng(seed=shuffle_seed)

  rng1.shuffle(x_normal)
  rng2.shuffle(y_normal)

  no_of_all_normal_samples = int(len(y_normal) * training_percentage)
  x_normal_training = x_normal[:no_of_all_normal_samples]
  x_normal_testing = x_normal[no_of_all_normal_samples:]
  y_normal_testing = y_normal[no_of_all_normal_samples:]

  x_all_testing = np.vstack([x_normal_testing, x_anomaly])
  y_all_testing = np.concatenate([y_normal_testing, y_anomaly])

  # return x train, y train, x test, y test
  return x_normal_training, None, x_all_testing, y_all_testing

def get_model_by_name(model_name, random_state_val):
  if model_name == IF_CONST:
    return create_IF_model(random_state_val)

  if model_name == LOF_CONST:
    return create_LOF_model()

  if model_name == SGD_OC_SVM_CONST:
    return create_SGD_OC_SVM_model(random_state_val)

  if model_name == ELL_ENV_CONST:
    return create_Elliptic_envelope_model(random_state_val)

  return None, None


def compute_for_exp_tuned_data_model(X, y, model_name, no_of_iterations, exponents_val):
  res_dict = dict()

  minority_collected_auroc = []
  minority_collected_aupr = []

  majority_collected_auroc = []
  majority_collected_aupr = []

  for i in range(no_of_iterations):
    model_name_val2, model_obj = get_model_by_name(model_name, random_state_val = i)
    x_train, y_train, x_test, y_test = split_data_for_train_test(X, y, shuffle_seed = i, training_percentage = 0.8)
    x_train, training_output_col_i = training_at_dataset_level(x_train,intermediate_range=(0,1), exponents_values_v = exponents_val)
    x_test = testing_at_dataset_level(x_test, training_output_col_i, intermediate_range=(0,1), exponents_values_v = exponents_val)
    y_pred, y_scores = train_evaluate_model(x_train, x_test, y_train, y_test, model_name, model_obj)

    for positive_label in [MINORITY_CLASS_LABEL, MAJORITY_CLASS_LABEL]:
      auroc_value = compute_auroc_score(y_test, y_scores)
      aupr_value = compute_aupr_score(y_test, y_scores, pos_label_value = positive_label)

      if positive_label == MINORITY_CLASS_LABEL:
        minority_collected_auroc.append(auroc_value)
        minority_collected_aupr.append(aupr_value)
      else:
        majority_collected_auroc.append(auroc_value)
        majority_collected_aupr.append(aupr_value)

  res_dict[MINORITY_CLASS_LABEL] = dict()
  res_dict[MINORITY_CLASS_LABEL][AUROC_CONST] = (np.mean(minority_collected_auroc), np.std(minority_collected_auroc))
  res_dict[MINORITY_CLASS_LABEL][AUPR_CONST] = (np.mean(minority_collected_aupr), np.std(minority_collected_aupr))

  res_dict[MAJORITY_CLASS_LABEL] = dict()
  res_dict[MAJORITY_CLASS_LABEL][AUROC_CONST] = (np.mean(majority_collected_auroc), np.std(majority_collected_auroc))
  res_dict[MAJORITY_CLASS_LABEL][AUPR_CONST] = (np.mean(majority_collected_aupr), np.std(majority_collected_aupr))

  return res_dict

def compute_for_preprocessing_data_model(X, y, model_name, no_of_iterations, preprocessing_method_name):
  X_copy = np.copy(X)
  y_copy = np.copy(y)
  res_dict = dict()

  minority_collected_auroc = []
  minority_collected_aupr = []

  majority_collected_auroc = []
  majority_collected_aupr = []

  for i in range(no_of_iterations):
    X_copy = preprocess_data_by_method_name(X_copy, preprocessing_method_name)
    model_name_val2, model_obj = get_model_by_name(model_name, random_state_val = i)
    x_train, y_train, x_test, y_test = split_data_for_train_test(X_copy, y, shuffle_seed = i, training_percentage = 0.8)
    y_pred, y_scores = train_evaluate_model(x_train, x_test, y_train, y_test, model_name, model_obj)

    for positive_label in [MINORITY_CLASS_LABEL, MAJORITY_CLASS_LABEL]:
      auroc_value = compute_auroc_score(y_test, y_scores)
      aupr_value = compute_aupr_score(y_test, y_scores, pos_label_value = positive_label)

      if positive_label == MINORITY_CLASS_LABEL:
        minority_collected_auroc.append(auroc_value)
        minority_collected_aupr.append(aupr_value)
      else:
        majority_collected_auroc.append(auroc_value)
        majority_collected_aupr.append(aupr_value)

  res_dict[MINORITY_CLASS_LABEL] = dict()
  res_dict[MINORITY_CLASS_LABEL][AUROC_CONST] = (np.mean(minority_collected_auroc), np.std(minority_collected_auroc))
  res_dict[MINORITY_CLASS_LABEL][AUPR_CONST] = (np.mean(minority_collected_aupr), np.std(minority_collected_aupr))

  res_dict[MAJORITY_CLASS_LABEL] = dict()
  res_dict[MAJORITY_CLASS_LABEL][AUROC_CONST] = (np.mean(majority_collected_auroc), np.std(majority_collected_auroc))
  res_dict[MAJORITY_CLASS_LABEL][AUPR_CONST] = (np.mean(majority_collected_aupr), np.std(majority_collected_aupr))

  return res_dict
