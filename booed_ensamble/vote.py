# vote.py
import numpy as np
from tensorflow.keras.models import load_model

def weighted_vote(models, weights, test_generator):
    predictions = []

    for i, model_path in enumerate(models):
        model = load_model(model_path)
        y_probs = model.predict(test_generator)
        predictions.append(y_probs * weights[i])

    total = np.sum(predictions, axis=0)
    final_preds = np.argmax(total, axis=1)
    return final_preds
