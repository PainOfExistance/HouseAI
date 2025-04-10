import numpy as np
import tensorflow as tf
from evaluator import ModelEvaluator
from model import CancerClassifier
from preprocessing import MedicalDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from trainer import ModelTrainer

# Configuration
config = {
    'image_size': (224, 224),
    'batch_size': 32,
    'epochs': 20,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'dataset_dir': "../Data",
    'augmentation_factor': 2
}

# Initialize data generators
def create_generators():
    train_datagen = MedicalDataGenerator(
        preprocessing_function=lambda x: x/255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.01,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],
        fill_mode='constant',
        cval=0
    )

    val_test_datagen = MedicalDataGenerator(
        preprocessing_function=lambda x: x/255.0
    )

    train_gen = train_datagen.flow_from_directory(
        f"{config['dataset_dir']}/train",
        target_size=config['image_size'],
        batch_size=config['batch_size'],
        class_mode="categorical",
        shuffle=True
    )

    val_gen = val_test_datagen.flow_from_directory(
        f"{config['dataset_dir']}/valid",
        target_size=config['image_size'],
        batch_size=config['batch_size'],
        class_mode="categorical",
        shuffle=False
    )

    test_gen = val_test_datagen.flow_from_directory(
        f"{config['dataset_dir']}/test",
        target_size=config['image_size'],
        batch_size=config['batch_size'],
        class_mode="categorical",
        shuffle=False
    )

    # Compute class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weights = dict(enumerate(class_weights))

    return train_gen, val_gen, test_gen, class_weights

# Main execution
if __name__ == "__main__":
    # Initialize data
    train_gen, val_gen, test_gen, class_weights = create_generators()

    # Build and train model
    model = CancerClassifier.build_model(
        input_shape=(*config['image_size'], 3),
        weight_decay=config['weight_decay']
    )

    trainer = ModelTrainer(
        model=model,
        train_gen=train_gen,
        val_gen=val_gen,
        class_weights=class_weights
    )
    trainer.compile_model(lr=config['lr'])
    history = trainer.train(epochs=config['epochs'])

    # Evaluate
    model = tf.keras.models.load_model('best_model.keras')
    ModelEvaluator.generate_report(model, test_gen)