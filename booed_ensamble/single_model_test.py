from dataset_utils import load_all_filepaths_and_labels, get_generator_from_paths, create_validation_generator
from bc_model import train_and_evaluate
from config import *


if __name__ == "__main__":
    X, y = load_all_filepaths_and_labels(TRAIN_DIR)
    train_gen = get_generator_from_paths(X, y, IMAGE_SIZE, BATCH_SIZE, augment=True)
    val_gen = create_validation_generator(VALID_DIR)

    model_path = "./saved_models/single_model"
    model_file, acc, errors = train_and_evaluate(train_gen, val_gen, EPOCHS, model_path, y)

    print(f"[✅] Single model accuracy: {acc:.4f}")
