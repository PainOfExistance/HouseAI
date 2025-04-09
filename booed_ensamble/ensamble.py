# ensemble.py
from dataset_utils import *
from bc_model import train_and_evaluate
from config import *

def run_ensemble():
    all_filepaths, all_labels = load_all_filepaths_and_labels(TRAIN_DIR)
    val_gen = create_validation_generator(VALID_DIR)

    models = []
    model_weights = []
    errors = []

    for i in range(ENSEMBLE_SIZE):
        print(f"\n🔄 [BC-{i + 1}] Preparing data...")

        #X, y = bootstrap_sample_with_errors(all_filepaths, all_labels, errors)
        X = all_filepaths
        y = all_labels
        train_gen = get_generator_from_paths(X, y, IMAGE_SIZE, BATCH_SIZE, augment=True)

        model_path = f"{MODEL_SAVE_DIR}/model_{i}.keras"

        print(f"🚀 [BC-{i + 1}] Training started...")
        model_file, acc, errors = train_and_evaluate(train_gen, val_gen, EPOCHS, model_path, y)


        print(f"✅ [BC-{i + 1}] Training complete! Accuracy on its training subset: {acc:.4f}")
        print(f"📦 [BC-{i + 1}] Misclassified samples passed to next stage: {len(errors)}")

        models.append(model_file)
        model_weights.append(acc)

    return models, model_weights
