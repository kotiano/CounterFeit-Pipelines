from zenml import step
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
import tensorflow as tf
import numpy as np
from datetime import date, datetime
from database import SessionLocal
from models.db_models import ScanResult
from dotenv import load_dotenv
import os
import mlflow
import uuid
import logging

load_dotenv()
#LATENT_DIM = int(os.getenv("LATENT_DIM", 256))
#SIGMA = float(os.getenv("SIGMA", 0.0003))
#AUTHENTICITY_THRESHOLD = float(os.getenv("AUTHENTICITY_THRESHOLD", 0.4))
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "Uploads/")
KEEP_IMAGES = os.getenv("KEEP_IMAGES", "true").lower() == "true"

class DeployModel:

    def __init__(self, latent_dim: str, sigma: str, threshold: float,
                 model: tf.keras.Model, image: np,ndarray, data: dict):

        self.latent_dim= latent_dim
        self.sigma= sigma
        self.threshold= threshold
        self.model=model

    def deploy_prediction(self) -> dict:
        """Run prediction and save to database."""
        pseudo_negative = tf.random.normal([1, self.latent_dim], mean=0.0, stddev=self.sigma)
        test_probs, _ = self.model.predict([np.expand_dims(self.image, axis=0), pseudo_negative], verbose=0)
        score = float(test_probs[0][0])
        is_authentic = score >= self.threshold
        batch_no = f"{self.data['brand'][:3].upper()}-{datetime.now().year}-{uuid.uuid4().hex[:4]}"
        mlflow.log_metric("confidence", score)
        with SessionLocal() as db:
            scan = ScanResult(
                brand=self.data['brand'],
                batch_no=batch_no,
                date=date.today(),
                confidence=score,
                is_authentic=is_authentic,
                latitude=float(self.data['latitude']) if self.data.get('latitude') != "Unknown" else None,
                longitude=float(self.data['longitude']) if self.data.get('longitude') != "Unknown" else None,
                image_url=f"/Uploads/{uuid.uuid4().hex}_{os.path.basename(self.data['image_path'])}" if KEEP_IMAGES else None,
                timestamp=datetime.now().isoformat()
            )
            db.add(scan)
            db.commit()
            db.refresh(scan)
        return {
            "id": scan.id,
            "is_authentic": is_authentic,
            "brand": self.data['brand'],
            "batch_no": batch_no,
            "date": date.today().strftime("%Y-%m-%d"),
            "confidence": f"{score:.2%}",
            "latitude": str(scan.latitude) if scan.latitude else "Unknown",
            "longitude": str(scan.longitude) if scan.longitude else "Unknown",
            "image_url": scan.image_url or "",
            "message": "Authentic" if is_authentic else "Counterfeit detected"
        }
        
            

@step(experiment_tracker="mlflow_tracker")

def deployment(latent_dim: str, sigma: str, threshold: float,
                 model: tf.keras.Model, image: np,ndarray, data: dict
):
    try:
        logging.info(f'Deploying Model')
        deploy=DeployModel
        deploy=deploy.deploy_prediction()
        return deploy
    except Exception as e:
        logging.error(f'An error occured while deploying model {e}')
        raise e