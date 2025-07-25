from abc import ABC, abstractmethod
from zenml import step
import os
import cv2
from pathlib import Path
import uuid
from dotenv import load_dotenv
import mlflow
import logging

load_dotenv()
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "Uploads/")

class BaseIngestData(ABC):
    """Abstract base class for data ingestion implementations."""
    
    @abstractmethod
    def __init__(self, image_files: list[Path]):
        """Initialize with a list of image file paths."""
        pass
    
    @abstractmethod
    def get_data(self) -> list[Path]:
        """Ingest the data and return the list of ingested file paths."""
        pass

class IngestData(BaseIngestData):
    def __init__(self, image_files: list[Path]):
        self.image_files = image_files

    def get_data(self) -> list[Path]:
        """Ingest raw image data and save to upload folder."""
        try:
            ingested_paths = []
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            for file in self.image_files:
                if file.is_file() and file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    dest = Path(UPLOAD_FOLDER) / f"{uuid.uuid4().hex}_{file.name}"
                    img = cv2.imread(str(file))
                    if img is None:
                        logging.warning(f"Failed to read image: {file}")
                        continue
                    cv2.imwrite(str(dest), img)
                    ingested_paths.append(dest)
                    logging.info(f"Ingested: {dest}")
                else:
                    logging.warning(f"Skipping invalid file: {file}")
            
            # Log the number of ingested files as a metric
            mlflow.log_metric("num_files_ingested", len(ingested_paths))
            
            # Save ingested file paths to a text file and log as an artifact
            if ingested_paths:
                artifact_file = "ingested_files.txt"
                with open(artifact_file, "w") as f:
                    for path in ingested_paths:
                        f.write(f"{path}\n")
                mlflow.log_artifact(artifact_file)
                os.remove(artifact_file)  # Clean up temporary file
            
            return ingested_paths
        except Exception as e:
            logging.error(f"Error ingesting data: {str(e)}")
            raise

@step(experiment_tracker="mlflow_tracker")
def ingest_data_step(image_files: list[Path]) -> list[Path]:
    """ZenML step for ingesting raw image data."""
    try:
        logging.info("Starting data ingestion")
        data_ingestor = IngestData(image_files)
        ingested_paths = data_ingestor.get_data()
        logging.info(f"Successfully ingested {len(ingested_paths)} files")
        return ingested_paths
    except Exception as e:
        logging.error(f"Error in ingestion step: {str(e)}")
        raise