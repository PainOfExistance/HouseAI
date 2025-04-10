from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, TensorBoard)
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.optimizers import AdamW


class ModelTrainer:
    def __init__(self, model, train_gen, val_gen, class_weights):
        self.model = model
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.class_weights = class_weights
        
    def compile_model(self, lr=1e-4, weight_decay=1e-4):
        optimizer = AdamW(learning_rate=lr, weight_decay=weight_decay)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', AUC(name='auc'), Precision(name='precision'), Recall(name='recall')]
        )
    
    def get_callbacks(self):
        return [
            EarlyStopping(monitor='val_auc', patience=10, mode='max'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
            ModelCheckpoint('best_model.keras', monitor='val_auc', save_best_only=True, mode='max'),
            TensorBoard(log_dir='./logs')
        ]
    
    def train(self, epochs=20, augmentation_factor=2):
        history = self.model.fit(
            self.train_gen,
            steps_per_epoch=self.train_gen.samples // self.train_gen.batch_size * augmentation_factor,
            validation_data=self.val_gen,
            validation_steps=self.val_gen.samples // self.val_gen.batch_size,
            epochs=epochs,
            callbacks=self.get_callbacks(),
            class_weight=self.class_weights
        )
        return history