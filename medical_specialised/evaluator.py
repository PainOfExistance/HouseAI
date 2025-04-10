import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from tqdm import tqdm


class ModelEvaluator:
    @staticmethod
    def generate_medical_report(model, test_gen):
        print("\n[STATUS] Generating medical evaluation report...")

        # Get true and predicted values
        y_true = test_gen.classes
        y_pred_probs = model.predict(test_gen, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        class_names = list(test_gen.class_indices.keys())

        # 1. Classification Report
        print("\n=== Detailed Classification Report ===")
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

        # 2. Confusion Matrix
        plt.figure(figsize=(10,8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        plt.close()

        # 3. ROC Curves (One-vs-Rest)
        plt.figure(figsize=(10,8))
        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_pred_probs[:, i])
            auc_score = roc_auc_score((y_true == i).astype(int), y_pred_probs[:, i])
            plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc_score:.3f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves by Class')
        plt.legend()
        plt.savefig('roc_curves.png')
        plt.close()

        # 4. Save Misclassified Examples
        misclassified = np.where(y_pred != y_true)[0]
        os.makedirs('misclassified', exist_ok=True)

        for idx in tqdm(misclassified[:20], desc="Saving misclassified examples"):
            batch_idx = idx // 32
            img_idx = idx % 32

            batch = test_gen[batch_idx]
            img = batch[0][img_idx]
            true_class = class_names[y_true[idx]]
            pred_class = class_names[y_pred[idx]]

            plt.imshow(img)
            plt.title(f"True: {true_class}\nPred: {pred_class}")
            plt.savefig(f'misclassified/{idx}_{true_class}_as_{pred_class}.png')
            plt.close()


    @staticmethod
    def save_misclassified(test_gen, y_true, y_pred, class_names, n=20):
        misclassified = np.where(y_pred != y_true)[0]
        os.makedirs('misclassified', exist_ok=True)
        
        for idx in tqdm(misclassified[:n], desc="Saving misclassified examples"):
            batch_idx = idx // test_gen.batch_size
            img_idx = idx % test_gen.batch_size
            
            batch = test_gen[batch_idx]
            img = batch[0][img_idx]
            true_class = class_names[y_true[idx]]
            pred_class = class_names[y_pred[idx]]
            
            plt.imshow(img)
            plt.title(f"True: {true_class}\nPred: {pred_class}")
            plt.savefig(f'misclassified/{idx}_{true_class}_as_{pred_class}.png')
            plt.close()