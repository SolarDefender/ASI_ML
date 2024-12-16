import argparse
import pandas as pd
import joblib
from google.cloud import storage
from sklearn.ensemble import RandomForestClassifier
 
def main(bucket_name):
    # Download training data from GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
 
    blob = bucket.blob('X_train.csv')
    blob.download_to_filename('X_train.csv')
    blob = bucket.blob('Y_train.csv')
    blob.download_to_filename('Y_train.csv')
 
    # Load data
    X_train = pd.read_csv('X_train.csv')
    Y_train = pd.read_csv('Y_train.csv')
 
    # Train the model
    model = RandomForestClassifier()
    model.fit(X_train, Y_train.values.ravel())
 
    # Save the model locally
    joblib.dump(model, 'model.joblib')
 
    # Upload the model to GCS
    blob = bucket.blob('models/model.joblib')
    blob.upload_from_filename('model.joblib')
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket-name', type=str, required=True)
    args = parser.parse_args()
    main(args.bucket_name)