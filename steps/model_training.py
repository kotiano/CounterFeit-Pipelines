from abc import ABC, abstractmethod
from zenml import step
import tensorflow as tf
import numpy as np
import mlflow
import logging
from dotenv import load_dotenv
import os

load_dotenv()
MODEL_PATH = os.path.join(os.getenv("MODEL_PATH", "models/autoencoder"), ".keras")
TARGET_SIZE = tuple(map(int, os.getenv("TARGET_SIZE", "224,224").split(",")))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))

class BaseModel(ABC):
    """Abstract base class for model implementations."""
    
    @abstractmethod
    def __init__(self, model_path: str, target_size: tuple):
        """Initialize the model with path and target size."""
        pass
    
    @abstractmethod
    def train_model(self, cleaned_images: list[np.ndarray]) -> tf.keras.Model:
        """Train the model on the provided images."""
        pass

class Model(BaseModel):
    def __init__(self, model_path: str, target_size: tuple):
        self.model_path = model_path
        self.target_size = target_size
        self.batch_size = BATCH_SIZE

    def train_model(self, cleaned_images: list[np.ndarray]) -> tf.keras.Model:
        """Train the autoencoder model."""
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=self.target_size + (3,)),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                                     kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', 
                                     kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', 
                                     kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                tf.keras.layers.UpSampling2D((2, 2)),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                                     kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                tf.keras.layers.UpSampling2D((2, 2)),
                tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
            ])

            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

            x_train = np.array(cleaned_images)
            if x_train.shape[1:] != self.target_size + (3,):
                raise ValueError(f"Input images shape {x_train.shape[1:]} does not match target size {self.target_size + (3,)}")

            train_dataset = tf.data.Dataset.from_tensor_slices((x_train, x_train)).batch(self.batch_size).shuffle(buffer_size=len(x_train)).repeat()
            val_dataset = tf.data.Dataset.from_tensor_slices((x_train, x_train)).batch(self.batch_size).repeat()

            train_samples = int(len(x_train) * 0.8)  
            val_samples = len(x_train) - train_samples
            steps_per_epoch = max(1, train_samples // self.batch_size)
            validation_steps = max(1, val_samples // self.batch_size)

            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.model_path,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    verbose=1,
                    restore_best_weights=True
                )
            ]

            history = model.fit(
                train_dataset,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_dataset,
                validation_steps=validation_steps,
                epochs=100,
                callbacks=callbacks,
                verbose=2
            )

            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            model.save(self.model_path)

            mlflow.log_param("epochs", 100)
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("learning_rate", 1e-4)
            mlflow.log_param("validation_split", 0.2)
            mlflow.log_metric("training_loss", history.history['loss'][-1])
            mlflow.log_metric("validation_loss", history.history['val_loss'][-1])
            mlflow.log_metric("training_mae", history.history['mae'][-1])
            mlflow.log_metric("validation_mae", history.history['val_mae'][-1])

            return model

        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            raise

@step(experiment_tracker="mlflow_tracker")
def train_model_step(cleaned_images: list[np.ndarray]) -> tf.keras.Model:
    """ZenML step for training the autoencoder model."""
    try:
        logging.info("Starting model training")
        model_trainer = Model(model_path=MODEL_PATH, target_size=TARGET_SIZE)
        trained_model = model_trainer.train_model(cleaned_images)
        logging.info("Model training completed successfully")
        return trained_model
    except Exception as e:
        logging.error(f"Error in training step: {str(e)}")
        raise