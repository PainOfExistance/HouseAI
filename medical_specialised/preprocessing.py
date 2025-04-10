import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class MedicalImagePreprocessor:
    @staticmethod
    def medical_preprocess(image):
        """CT-specific preprocessing pipeline"""
        # CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        
        # Gamma correction
        gamma = 0.8
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        image = cv2.LUT(image, table)
        return image

class MedicalDataGenerator(ImageDataGenerator):
    def __call__(self, *args, **kwargs):
        batches = super().__call__(*args, **kwargs)
        for batch_x, batch_y in batches:
            processed = np.array([MedicalImagePreprocessor.medical_preprocess(img) for img in batch_x])
            yield (processed, batch_y)