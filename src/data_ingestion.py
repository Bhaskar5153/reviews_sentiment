import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import PROJECT_ID, LOCATION, SERVIVE_ACCOUNT_PATH

from google.cloud import storage
import tempfile
import pandas as pd



# Use only the bucket name, no gs:// prefix
bucket_name = "machine_learning_datasets"
blob_name = "raw_data/cleaned_reviews.csv"

storage_client = storage.Client(project=PROJECT_ID)

PROCESSED_FOLDER = "processed_data"


def load_raw_data_from_gcs(file_name):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    local_file_path = os.path.join(tempfile.gettempdir(), file_name)
    blob.download_to_filename(local_file_path)
    df = pd.read_csv(local_file_path)
    return df


def save_processed_data_to_gcs(df: pd.DataFrame, file_name: str) -> None:
    """Upload a processed DataFrame to GCS processed_data/ folder."""
    bucket     = storage_client.bucket(bucket_name)
    blob       = bucket.blob(f"{PROCESSED_FOLDER}/{file_name}")
    local_path = os.path.join(tempfile.gettempdir(), file_name)
    df.to_csv(local_path, index=False)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {len(df):,} rows  ->  gs://{bucket_name}/{PROCESSED_FOLDER}/{file_name}")


def load_processed_data_from_gcs(file_name: str) -> pd.DataFrame:
    """Download processed data from GCS processed_data/ folder."""
    bucket     = storage_client.bucket(bucket_name)
    blob       = bucket.blob(f"{PROCESSED_FOLDER}/{file_name}")
    local_path = os.path.join(tempfile.gettempdir(), file_name)
    blob.download_to_filename(local_path)
    return pd.read_csv(local_path)
