from abc import ABC, abstractmethod
from zenml import step
import cv2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import mlflow
import logging
import os

load_dotenv()
TARGET_SIZE = tuple(map(int, os.getenv("TARGET_SIZE", "224,224").split(",")))

class BasePreprocessData(ABC):
    """Abstract base class for data preprocessing implementations."""
    
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

    def preprocess(self) -> list[np.ndarray]:
        """Clean and preprocess images."""
        try:
            cleaned_images = []
            logging.info(f"Starting image preprocessing for {len(self.image_paths)} images")
            for path in self.image_paths:
                if not path.is_file():
                    logging.warning(f"Skipping invalid file path: {path}")
                    continue
                img = cv2.imread(str(path))
                if img is None:
                    logging.warning(f"Failed to read image: {path}")
                    continue
                img = cv2.resize(img, TARGET_SIZE)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype('float32') / 255.0
                cleaned_images.append(img)
                logging.info(f"Preprocessed image: {path}")
            
            if not cleaned_images:
                raise ValueError("No valid images were preprocessed")
            
            mlflow.log_param("target_size", TARGET_SIZE)
            mlflow.log_metric("num_images_preprocessed", len(cleaned_images))
            return cleaned_images
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