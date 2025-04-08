# main.py
from ensamble import run_ensemble
from vote import weighted_vote
from config import *
from dataset_utils import create_test_generator
from sklearn.metrics import classification_report
from config import TRAIN_DIR

if __name__ == "__main__":

    print("[DEBUG] Using TRAIN_DIR:", TRAIN_DIR)

    models, weights = run_ensemble()
    test_gen = create_test_generator(TEST_DIR)

    y_true = test_gen.classes
    y_pred = weighted_vote(models, weights, test_gen)

    print(classification_report(y_true, y_pred))
