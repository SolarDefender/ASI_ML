import pandas as pd
from google.cloud import aiplatform
from google.cloud import storage
import joblib
 
def train_models(X_train, Y_train, X_dev, Y_dev, parameters):
    # Save training data locally
    X_train.to_csv('X_train.csv', index=False)
    Y_train.to_csv('Y_train.csv', index=False)
 
    # Upload training data to Google Cloud Storage
    bucket_name = 'solar_defender_data'
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
 
    blob = bucket.blob('X_train.csv')
    blob.upload_from_filename('X_train.csv')
    blob = bucket.blob('Y_train.csv')
    blob.upload_from_filename('Y_train.csv')
    print("aaaaaaaa")
 
    # Initialize AI Platform
    project_id = 'SolarDefender'
    region = 'us-central1'
    aiplatform.init(project=project_id, location=region, staging_bucket=bucket_name)
    print("bbbbbbb")
 
    # Define the custom training job
    job = aiplatform.CustomJob.from_local_script(
        display_name='kedro-training-job',
        script_path='src/ASI_ML/pipelines/data_science/trainer.py',
        container_uri = 'us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.0-23:latest',
        args=['--bucket-name', bucket_name],
        requirements=[
            'pandas',
            'joblib',
            'scikit-learn',
            'google-cloud-storage'
        ],
        replica_count=1,
        machine_type='n1-standard-4',
    )
 
    # Run the training job
    job.run(sync=True)
 
    # Download the trained model from Cloud Storage
    blob = bucket.blob('models/model.joblib')
    blob.download_to_filename('model.joblib')
 
    # Load the model
    model = joblib.load('model.joblib')
 
    return model