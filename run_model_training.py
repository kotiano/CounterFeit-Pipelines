from pipelines.model_training_pipeline import model_training_pipeline

if __name__ == "__main__":
    image_dir = "/home/jude/test/img"  
    model = model_training_pipeline(image_dir)