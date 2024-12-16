import json
from google.cloud import storage
from kedro.framework.session import KedroSession
from pathlib import Path


def export_parameters_to_gcs(parameters, gcs_path="gs://solar_defender_training/parameters.json"):
    client = storage.Client()
    bucket_name, file_path = gcs_path.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)

    # Konwersja parametrów Kedro na JSON
    with blob.open("w") as f:
        json.dump(parameters, f)

project_path = Path.cwd()
with KedroSession.create("solar_defender", project_path) as session:
    context = session.load_context()
    params = context.params  # Pobieranie parametrów
    export_parameters_to_gcs(params, "gs://solar_defender_training/parameters.json")
