# EXPERIMENT SETUP

import warnings
import json
import numpy as np

from constants import EYE_DATASET, PHISING_DATASET, IONOSPHERE_DATASET, SHUTTLE_CONST, SPAMBASE_CONST, CARDIO_CONST, BANKNOTE_CONST, MAGIC_CONST, OCCUPANCY_CONST, SENSORLESS_DRIVE_CONST, ELECTRIC_GRID_CONST, MALWARE_DATASET, CAMPAIGN_CONST
from constants import IF_CONST, LOF_CONST, SGD_OC_SVM_CONST, ELL_ENV_CONST
from constants import NO_TRANSFORMATION_CONST, ROBUST_SCALER_TRANSFORMATION_CONST, POWER_TRANSFORMATION_CONST, QUANTILE_TRANSFORMATION_CONST
from models import NpEncoder
from datasets import get_dataset_by_name
from experiment import compute_for_preprocessing_data_model, compute_for_exp_tuned_data_model

warnings.filterwarnings("ignore")

experiment_res_dict = dict()
used_datasets =  [EYE_DATASET, PHISING_DATASET, IONOSPHERE_DATASET, SHUTTLE_CONST, SPAMBASE_CONST, CARDIO_CONST, BANKNOTE_CONST, MAGIC_CONST,
                  OCCUPANCY_CONST, SENSORLESS_DRIVE_CONST, ELECTRIC_GRID_CONST, MALWARE_DATASET, CAMPAIGN_CONST]


used_models = [IF_CONST, LOF_CONST, SGD_OC_SVM_CONST, ELL_ENV_CONST]
no_of_iters = 10
#used_exponents_collection = [ [2, 3], [2,5], [3, 5], [4,5], [5, 5], [5,7], [7, 5], [8,3], [8,5], [9,5]]
used_exponents_collection = [ [3,5] ]

used_preprocessing_methods = [NO_TRANSFORMATION_CONST, ROBUST_SCALER_TRANSFORMATION_CONST, POWER_TRANSFORMATION_CONST, QUANTILE_TRANSFORMATION_CONST]

for dataset_name_val in used_datasets:
  experiment_res_dict[dataset_name_val] = dict()
  # import data
  X, y = get_dataset_by_name(dataset_name = dataset_name_val)
  X = X.astype(np.float64)
  for model_name_val in used_models:
    no_of_iters = 10
    for use_exp in [True, False]:
      if use_exp is False:
        for pp_method_name in used_preprocessing_methods:
          res_dict_default_model = compute_for_preprocessing_data_model(X, y, model_name_val, no_of_iters, pp_method_name)
          exp_key = str(pp_method_name)+"-"+str(model_name_val)
          experiment_res_dict[dataset_name_val][exp_key] = res_dict_default_model
          print("DONE: ", dataset_name_val," - ",exp_key)

      else:
        for exp_val in used_exponents_collection:
          res_dict_exp_model = compute_for_exp_tuned_data_model(X, y, model_name_val, no_of_iters, exp_val)
          key_val = model_name_val + "_" + str(exp_val)
          experiment_res_dict[dataset_name_val][key_val] = res_dict_exp_model
          print("DONE: ", dataset_name_val," - ",key_val)


json_dict_name = 'output.json'

with open(json_dict_name, 'w') as json_file:
    json.dump(experiment_res_dict, json_file, indent=2, cls=NpEncoder)
