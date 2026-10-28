import pandas as pd

from constants import SENSORLESS_DRIVE_CONST, OCCUPANCY_CONST, MAGIC_CONST, BANKNOTE_CONST, CARDIO_CONST, SHUTTLE_CONST, SPAMBASE_CONST, ELECTRIC_GRID_CONST, EYE_DATASET, PHISING_DATASET, IONOSPHERE_DATASET, MALWARE_DATASET, CAMPAIGN_CONST


def import_campaign_dataset():
  # https://github.com/mala-lab/ADBenchmarks-anomaly-detection-datasets?tab=readme-ov-file
  # dim 41k x 62

  df = pd.read_csv("datasets/bank-additional-full_normalised_campaign.csv")
  # id col removed
  column_name = "class"
  freq = df[column_name].value_counts()
  df_except_target = df.drop(columns=column_name)
  X = df_except_target.to_numpy()
  y = df[column_name].to_numpy()

  return X, y


def import_cardio_dataset():
  df = pd.read_csv("datasets/Cardiotocography.csv")

  column_name = "NSP"  # replace with your target col

  freq = df[column_name].value_counts()

  print('has na', df.isna().values.any())

  df_except_target = df.drop(columns=column_name)
  X = df_except_target.to_numpy()
  y = df[column_name].to_numpy()

  return X, y

def import_shuttle_dataset():
  df = pd.read_csv("datasets/statlog_shuttle.csv")

  column_name = "class"  # replace with your target col

  freq = df[column_name].value_counts()

  print('has na', df.isna().values.any())

  df_except_target = df.drop(columns=column_name)
  X = df_except_target.to_numpy()
  y = df[column_name].to_numpy()

  return X, y

def import_banknote_dataset():
  df = pd.read_csv("datasets/data_banknote_authentication.csv")
  # id col removed
  column_name = "class"

  df_except_target = df.drop(columns=column_name)
  X = df_except_target.to_numpy()
  y = df[column_name].to_numpy()

  return X, y

def import_magic_dataset():
  df = pd.read_csv("datasets/MAGIC_gamma_telescope.csv")
  column_name = "class"

  df_except_target = df.drop(columns=column_name)
  X = df_except_target.to_numpy()
  y = df[column_name].to_numpy()

  return X, y

def import_occupancy_dataset():
  df = pd.read_csv("datasets/occupancy.csv")
  column_name = "class"

  df_except_target = df.drop(columns=column_name)
  X = df_except_target.to_numpy()
  y = df[column_name].to_numpy()

  return X, y

def import_sensorless_drive_dataset():
  df = pd.read_csv("datasets/sensorless-drive-diagnosis.csv")
  column_name = "class"

  df_except_target = df.drop(columns=column_name)
  X = df_except_target.to_numpy()
  y = df[column_name].to_numpy()

  return X, y

def import_electric_grid_dataset():
  # https://www.kaggle.com/datasets/sowlarn/ucis-electrical-grid-stability-simulated-data
  # dim 10000 × 14

  df = pd.read_csv("datasets/electrical_grid-processed.csv")
  # id col removed
  column_name = "stabf"
  freq = df[column_name].value_counts()

  df_except_target = df.drop(columns=column_name)
  X = df_except_target.to_numpy()
  y = df[column_name].to_numpy()

  return X, y

def import_spambase_dataset():
  # https://archive.ics.uci.edu/dataset/94/spambase
  # dim 4601 × 57

  df = pd.read_csv("datasets/spambase.csv")
  # id col removed
  column_name = "Class"
  freq = df[column_name].value_counts()

  df_except_target = df.drop(columns=column_name)
  X = df_except_target.to_numpy()
  y = df[column_name].to_numpy()

  return X, y

def import_eye_dataset():
    # https://www.kaggle.com/datasets/robikscube/eye-state-classification-eeg-dataset
    df = pd.read_csv("datasets/EEG_Eye_State_Classification.csv")
    # id col removed
    print(df.isna().any().any())
    column_name = "eyeDetection"
    freq = df[column_name].value_counts()
    print(freq)

    df_except_target = df.drop(columns=column_name)
    X = df_except_target.to_numpy()
    y = df[column_name].to_numpy()

    return X, y

def import_phishing_detection():
  # https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset
  df = pd.read_csv("datasets/dataset_phishing-processed.csv")

  column_name = "status"
  freq = df[column_name].value_counts()
  print('has na', df.isna().values.any())
  df_except_target = df.drop(columns=column_name)
  X = df_except_target.to_numpy()
  y = df[column_name].to_numpy()



  return X, y

def import_ionosphere():
  # https://www.kaggle.com/datasets/jamieleech/ionosphere
  df = pd.read_csv("datasets/ionosphere-processed.csv")
  column_name = "class"

  df_except_target = df.drop(columns=column_name)
  X = df_except_target.to_numpy()
  y = df[column_name].to_numpy()

  return X, y


def import_malware_dataset():
  # https://www.kaggle.com/datasets/nsaravana/malware-detection/data

  df = pd.read_csv("datasets/Malware-dataset-fin.csv")
  column_name = "classification"

  freq = df[column_name].value_counts()
  print('has na', df.isna().values.any())
  print(freq)

  df_except_target = df.drop(columns=column_name)
  X = df_except_target.to_numpy()
  y = df[column_name].to_numpy()

  return X, y


def get_dataset_by_name(dataset_name):

  if dataset_name == SENSORLESS_DRIVE_CONST:
    return import_sensorless_drive_dataset()

  if dataset_name == OCCUPANCY_CONST:
    return import_occupancy_dataset()

  if dataset_name == MAGIC_CONST:
    return import_magic_dataset()

  if dataset_name == BANKNOTE_CONST:
    return import_banknote_dataset()

  if dataset_name ==  CARDIO_CONST:
    return import_cardio_dataset()

  if dataset_name == SHUTTLE_CONST:
    return import_shuttle_dataset()

  if dataset_name == SPAMBASE_CONST:
    return import_spambase_dataset()

  if dataset_name == ELECTRIC_GRID_CONST:
    return import_electric_grid_dataset()

  if dataset_name == EYE_DATASET:
    return import_eye_dataset()

  if dataset_name == PHISING_DATASET:
    return import_phishing_detection()

  if dataset_name == IONOSPHERE_DATASET:
    return import_ionosphere()

  if dataset_name == MALWARE_DATASET:
    return import_malware_dataset()

  if dataset_name == CAMPAIGN_CONST:
    return import_campaign_dataset()

