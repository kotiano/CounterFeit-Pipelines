from zenml import pipeline
from pathlib import Path
import tensorflow as tf
import numpy as np
import logging

from steps.data_ingestion import ingest_data_step
from steps.data_preprocessing import preprocess_data_step
from steps.model_training import train_model_step

@pipeline
def model_training_pipeline(image_dir: str) -> tf.keras.Model:
    """Pipeline for ingesting, preprocessing, and training an autoencoder model from a directory of images."""
    try:
        logging.info(f"Starting model training pipeline with image directory: {image_dir}")
        
        dir_path = Path(image_dir)
        
        if not dir_path.is_dir():
            raise ValueError(f"Directory does not exist or is not a directory: {image_dir}")
        
        supported_extensions = {'.png', '.jpg', '.jpeg'}
        image_paths = [
            file for file in dir_path.iterdir()
            if file.is_file() and file.suffix.lower() in supported_extensions
        ]
        
        if not image_paths:
            raise ValueError(f"No valid image files found in directory: {image_dir}")
        
        logging.info(f"Found {len(image_paths)} image files in {image_dir}")
        
        ingested_paths = ingest_data_step(image_paths)
        
        preprocessed_data = preprocess_data_step(ingested_paths)
        
        trained_model = train_model_step(preprocessed_data)
        
        logging.info("Pipeline completed successfully")
        return trained_model
    except Exception as e:
        logging.error(f"Error in pipeline execution: {str(e)}")
        raise