from abc import ABC, abstractmethod
from zenml import step
import cv2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import mlflow
import logging
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

load_dotenv()
TARGET_SIZE = tuple(map(int, os.getenv("TARGET_SIZE", "224,224").split(",")))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
COLOR_MODE = 'rgb'

class BasePreprocessData(ABC):
    
    @abstractmethod
    def __init__(self, image_paths: list[Path]):
        """Initialize with a list of image file paths."""
        pass
    
    @abstractmethod
    def preprocess(self) -> list[np.ndarray]:
        """Preprocess images and return a list of preprocessed image arrays."""
        pass

class PreprocessData(BasePreprocessData):
    def __init__(self, image_paths: list[Path]):
        self.image_paths = image_paths
        self.target_size = TARGET_SIZE
        self.batch_size = BATCH_SIZE

    def custom_preprocessing(self, image):
        """Apply contrast adjustment and normalization."""
        image = tf.image.adjust_contrast(image, contrast_factor=1.2)
        image = image / 255.0
        return image

    def random_noise(self, image):
        """Add random noise to the image."""
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.02, dtype=tf.float32)
        image = image + noise
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image

    def random_cutout(self, image):
        """Apply random cutout to the image."""
        h, w = self.target_size
        patch_size = tf.random.uniform([], minval=int(h * 0.1), maxval=int(h * 0.2), dtype=tf.int32)
        y = tf.random.uniform([], minval=0, maxval=h - patch_size, dtype=tf.int32)
        x = tf.random.uniform([], minval=0, maxval=w - patch_size, dtype=tf.int32)
        y_indices, x_indices = tf.meshgrid(
            tf.range(y, y + patch_size, dtype=tf.int32),
            tf.range(x, x + patch_size, dtype=tf.int32),
            indexing='ij'
        )
        indices = tf.stack([tf.reshape(y_indices, [-1]), tf.reshape(x_indices, [-1])], axis=1)
        updates = tf.zeros([patch_size * patch_size, image.shape[-1]], dtype=tf.float32)
        mask = tf.ones_like(image)
        mask = tf.tensor_scatter_nd_update(mask, indices, updates)
        return image * mask

    def advanced_preprocessing(self, image):
        """Combine preprocessing steps with random augmentations."""
        image = self.custom_preprocessing(image)
        if tf.random.uniform([]) < 0.4:
            image = self.random_noise(image)
        if tf.random.uniform([]) < 0.4:
            image = self.random_cutout(image)
        return image

    def preprocess(self) -> list[np.ndarray]:
        """Clean and preprocess images with augmentation."""
        try:
            logging.info(f"Starting image preprocessing for {len(self.image_paths)} images")
            
            temp_dir = os.path.join(os.getcwd(), 'temp_images', 'normal_class')
            os.makedirs(temp_dir, exist_ok=True)

            for i, path in enumerate(self.image_paths):
                if not path.is_file():
                    logging.warning(f"Skipping invalid file path: {path}")
                    continue
                img = cv2.imread(str(path))
                if img is None:
                    logging.warning(f"Failed to read image: {path}")
                    continue
                img = cv2.resize(img, self.target_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(temp_dir, f'image_{i}.jpg'), img)

            aug_datagen = ImageDataGenerator(
                preprocessing_function=self.advanced_preprocessing,
                rotation_range=30,
                width_shift_range=0.3,
                height_shift_range=0.3,
                zoom_range=[0.7, 1.3],
                brightness_range=[0.7, 1.3],
                shear_range=15,
                horizontal_flip=True,
                fill_mode='reflect'
            )

            generator = aug_datagen.flow_from_directory(
                os.path.dirname(temp_dir),
                target_size=self.target_size,
                batch_size=self.batch_size,
                class_mode='input',
                color_mode=COLOR_MODE,
                shuffle=True
            )

            cleaned_images = []
            total_images = min(len(self.image_paths), generator.samples)
            for _ in range(total_images // self.batch_size + 1):
                batch = next(generator)
                cleaned_images.extend(batch[0])

            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)
            os.rmdir(os.path.dirname(temp_dir))

            if not cleaned_images:
                raise ValueError("No valid images were preprocessed")

            cleaned_images = np.array(cleaned_images)
            logging.info(f"Preprocessed {len(cleaned_images)} images")

            mlflow.log_param("target_size", self.target_size)
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_metric("num_images_preprocessed", len(cleaned_images))
            return cleaned_images.tolist()

        except Exception as e:
            logging.error(f"Error preprocessing images: {str(e)}")
            raise

@step(experiment_tracker="mlflow_tracker")
def preprocess_data_step(image_paths: list[Path]) -> list[np.ndarray]:
    """ZenML step for cleaning and preprocessing images."""
    try:
        logging.info("Starting preprocessing step")
        preprocessor = PreprocessData(image_paths)
        preprocessed_data = preprocessor.preprocess()
        logging.info(f"Successfully preprocessed {len(preprocessed_data)} images")
        return preprocessed_data
    except Exception as e:
        logging.error(f"Error in preprocessing step: {str(e)}")
        raise