# config.py
import os

import os

# Go one level UP from booed_ensamble/ to PSIS/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "Data")

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALID_DIR = os.path.join(DATASET_DIR, "valid")
TEST_DIR = os.path.join(DATASET_DIR, "test")



ENSEMBLE_SIZE = 3  # Try 3–5 to allow the ensemble to learn errors
IMAGE_SIZE = (224, 224)  # Match single model
BATCH_SIZE = 32          # Larger batches stabilize gradients
SAMPLE_SIZE = 150        # Good balance: not too small, not full data
EPOCHS = 10              # Give each model a chance to learn
INIT_LR = 1e-4
WEIGHT_DECAY = 1e-4



MODEL_SAVE_DIR = "./saved_models"
