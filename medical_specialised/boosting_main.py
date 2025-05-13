# main_boosting.py

import json
from datetime import datetime

from evaluator import ModelEvaluator
from main import create_generators
from boosting import train_boosting_ensemble

if __name__ == "__main__":
    # Load data generators
    train_gen, val_gen, test_gen, class_weights = create_generators()
    class_names = list(test_gen.class_indices.keys())

    # Train boosting ensemble
    model_paths = train_boosting_ensemble(
        train_gen=train_gen,
        val_gen=val_gen,
        class_weights=class_weights,
        class_names=class_names,
        num_rounds=3,  # You can increase if needed
        save_dir="boost_models"
    )

    # Final clinical evaluation on test set
    print("\n=== FINAL ENSEMBLE EVALUATION ON TEST SET ===")
    from boosting import predict_with_ensemble
    import numpy as np

    x_test, y_test = [], []
    for i in range(len(test_gen)):
        x, y = test_gen[i]
        x_test.append(x)
        y_test.append(y)
    x_test = np.vstack(x_test)
    y_test = np.vstack(y_test)

    ensemble_preds = predict_with_ensemble(model_paths, x_test)
    ModelEvaluator.generate_medical_report(
        model=type("BoostedEnsemble", (), {"predict": lambda _, x: predict_with_ensemble(model_paths, x)})(),
        test_gen=test_gen
    )

    # Save deployment-ready metadata
    deployment_pkg = {
        'ensemble': {
            'model_paths': model_paths,
            'trained_with_boosting': True,
            'last_validated': datetime.now().isoformat()
        },
        'class_mapping': {v: k for k, v in test_gen.class_indices.items()}
    }
    with open('deployment_package_boosted.json', 'w') as f:
        json.dump(deployment_pkg, f)

    print("\n✅ Boosted ensemble pipeline complete. Ready for deployment.")
