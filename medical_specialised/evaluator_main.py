from main import create_generators
from model import CancerClassifier  # Or your model builder file
from evaluator import ModelEvaluator

if __name__ == "__main__":
    _, _, test_gen, _ = create_generators()

    # Rebuild architecture
    model = CancerClassifier.build_model()

    # Load weights only
    model.load_weights("models/medical_specialised_weights.h5")  # <-- This must be a weights-only file

    # Evaluate
    ModelEvaluator.generate_medical_report(model, test_gen)
