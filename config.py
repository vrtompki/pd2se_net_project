import os
import torch

# ================================== Directory configuration information ===============================================
# DATA_DIR = "C:\\Users\\Vincent\\Documents\\Big Data in Agriculture\\Datasets\\Liang\\pd2se_net_project\\"
DATA_DIR = os.getcwd()
TRAIN_DIR = "\\ai_challenger_pdr2018_trainingset_20181023\\AgriculturalDisease_trainingset\\"
TRAIN_ANN_FILENAME = "AgriculturalDisease_train_annotations.json"
TRAIN_IMG_FOLDER = "images\\"
TRAIN_IMG_PATH = os.path.join(DATA_DIR + TRAIN_DIR, TRAIN_IMG_FOLDER)
TRAIN_ANN_PATH = os.path.join(DATA_DIR + TRAIN_DIR, TRAIN_ANN_FILENAME)

VAL_DIR = "\\ai_challenger_pdr2018_validationset_20181023\\AgriculturalDisease_validationset\\"
VAL_ANN_FILENAME = "AgriculturalDisease_validation_annotations.json"
VAL_IMG_FOLDER = "images\\"
VAL_IMG_PATH = os.path.join(DATA_DIR + VAL_DIR, VAL_IMG_FOLDER)
VAL_ANN_PATH = os.path.join(DATA_DIR + VAL_DIR, VAL_ANN_FILENAME)

OUTPUT_DIR = os.path.join(DATA_DIR, "processed_data")
MODEL_DIR = DATA_DIR + "\\models\\"
# ====================================== Image configuration information ===============================================
IMG_HEIGHT = 256
IMG_WIDTH = 256
NUM_CHANNELS = 3

# ====================================== Model configuration information ===============================================
# ---------------------------------------------- Label configuration ---------------------------------------------------
MAP_61to45 = {'0': {'Plant ID': 1, 'Disease ID': 1, 'Severity ID': 1},
              '1': {'Plant ID': 1, 'Disease ID': 2, 'Severity ID': 2},
              '2': {'Plant ID': 1, 'Disease ID': 2, 'Severity ID': 3},
              '4': {'Plant ID': 1, 'Disease ID': 3, 'Severity ID': 4},
              '5': {'Plant ID': 1, 'Disease ID': 3, 'Severity ID': 5},
              '6': {'Plant ID': 2, 'Disease ID': 4, 'Severity ID': 6},
              '7': {'Plant ID': 2, 'Disease ID': 5, 'Severity ID': 7},
              '8': {'Plant ID': 2, 'Disease ID': 5, 'Severity ID': 8},
              '9': {'Plant ID': 3, 'Disease ID': 6, 'Severity ID': 9},
              '10': {'Plant ID': 3, 'Disease ID': 7, 'Severity ID': 10},
              '11': {'Plant ID': 3, 'Disease ID': 7, 'Severity ID': 11},
              '12': {'Plant ID': 3, 'Disease ID': 8, 'Severity ID': 12},
              '13': {'Plant ID': 3, 'Disease ID': 8, 'Severity ID': 13},
              '14': {'Plant ID': 3, 'Disease ID': 9, 'Severity ID': 14},
              '15': {'Plant ID': 3, 'Disease ID': 9, 'Severity ID': 15},
              '17': {'Plant ID': 4, 'Disease ID': 10, 'Severity ID': 16},
              '18': {'Plant ID': 4, 'Disease ID': 11, 'Severity ID': 17},
              '19': {'Plant ID': 4, 'Disease ID': 11, 'Severity ID': 18},
              '20': {'Plant ID': 4, 'Disease ID': 12, 'Severity ID': 19},
              '21': {'Plant ID': 4, 'Disease ID': 12, 'Severity ID': 20},
              '22': {'Plant ID': 4, 'Disease ID': 13, 'Severity ID': 21},
              '23': {'Plant ID': 4, 'Disease ID': 13, 'Severity ID': 22},
              '27': {'Plant ID': 5, 'Disease ID': 14, 'Severity ID': 23},
              '28': {'Plant ID': 5, 'Disease ID': 15, 'Severity ID': 24},
              '29': {'Plant ID': 5, 'Disease ID': 15, 'Severity ID': 25},
              '30': {'Plant ID': 6, 'Disease ID': 16, 'Severity ID': 26},
              '31': {'Plant ID': 6, 'Disease ID': 17, 'Severity ID': 27},
              '32': {'Plant ID': 6, 'Disease ID': 17, 'Severity ID': 28},
              '33': {'Plant ID': 7, 'Disease ID': 18, 'Severity ID': 29},
              '34': {'Plant ID': 7, 'Disease ID': 19, 'Severity ID': 30},
              '35': {'Plant ID': 7, 'Disease ID': 19, 'Severity ID': 31},
              '36': {'Plant ID': 7, 'Disease ID': 20, 'Severity ID': 32},
              '37': {'Plant ID': 7, 'Disease ID': 20, 'Severity ID': 33},
              '38': {'Plant ID': 8, 'Disease ID': 21, 'Severity ID': 34},
              '39': {'Plant ID': 8, 'Disease ID': 22, 'Severity ID': 35},
              '40': {'Plant ID': 8, 'Disease ID': 22, 'Severity ID': 36},
              '41': {'Plant ID': 9, 'Disease ID': 23, 'Severity ID': 37},
              '42': {'Plant ID': 9, 'Disease ID': 24, 'Severity ID': 38},
              '43': {'Plant ID': 9, 'Disease ID': 24, 'Severity ID': 39},
              '50': {'Plant ID': 9, 'Disease ID': 25, 'Severity ID': 40},
              '51': {'Plant ID': 9, 'Disease ID': 25, 'Severity ID': 41},
              '54': {'Plant ID': 9, 'Disease ID': 26, 'Severity ID': 42},
              '55': {'Plant ID': 9, 'Disease ID': 26, 'Severity ID': 43},
              '58': {'Plant ID': 9, 'Disease ID': 27, 'Severity ID': 44},
              '59': {'Plant ID': 9, 'Disease ID': 27, 'Severity ID': 45},
                 }

# -------------------------------------------- Performance configuration -----------------------------------------------
torch.cuda.current_device()
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")


# -------------------------------------------- Network configuration ---------------------------------------------------
BATCH_SIZE = 25
NUM_EPOCHS = 1000
NUM_CLASSES_1 = 9
NUM_CLASSES_2 = 27
NUM_CLASSES_3 = 45
PATIENCE = 50
FREEZE_LAYERS = 2
DROPOUT = 0.5
VERBOSE = 1