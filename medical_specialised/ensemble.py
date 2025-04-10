import numpy as np
import tensorflow as tf
from sklearn.metrics import recall_score
from tensorflow.keras.models import load_model


class MedicalEnsemble:
    def __init__(self, model_paths=None):
        self.models = []
        self.recall_weights = []  # Weight by recall (critical for cancer detection)
        
        if model_paths:
            for path in model_paths:
                self.add_model(path)

    def add_model(self, model_path):
        """Load a pre-trained model with safety checks"""
        try:
            model = load_model(model_path)
            self.models.append(model)
            print(f"✅ Loaded model: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load {model_path}: {str(e)}")
            raise RuntimeError("Model loading failed - verify model integrity")

    def _calculate_recall_weights(self, X_val, y_val):
        """Weight models by their class-wise recall (prioritize sensitivity)"""
        self.recall_weights = []
        for model in self.models:
            y_pred = model.predict(X_val, verbose=0)
            recalls = []
            for class_idx in range(y_val.shape[1]):
                recalls.append(recall_score(
                    y_val.argmax(axis=1) == class_idx,
                    y_pred.argmax(axis=1) == class_idx,
                    zero_division=0
                ))
            self.recall_weights.append(np.mean(recalls))  # Average recall across classes

    def predict_proba(self, X):
        """Weighted probability prediction with uncertainty estimation"""
        if not self.models:
            raise RuntimeError("No models in ensemble")
            
        all_preds = []
        for model, weight in zip(self.models, self.recall_weights):
            pred = model.predict(X, verbose=0) * weight
            all_preds.append(pred)
        
        mean_pred = np.mean(all_preds, axis=0)
        std_pred = np.std(all_preds, axis=0)  # Measure of uncertainty
        
        # Safety thresholding
        mean_pred[mean_pred < 0.05] = 0  # Reject very low confidence predictions
        return mean_pred, std_pred

    def predict_with_safety(self, X, uncertainty_threshold=0.3):
        """Conservative prediction with uncertainty checks"""
        mean_pred, std_pred = self.predict_proba(X)
        
        # Flag uncertain predictions
        uncertain = std_pred.mean(axis=1) > uncertainty_threshold
        final_pred = mean_pred.argmax(axis=1)
        final_pred[uncertain] = -1  # Mark for clinician review
        
        return final_pred, uncertain