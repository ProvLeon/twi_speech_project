import os
import logging
import pandas as pd
import requests
import boto3
from botocore.exceptions import ClientError
from pymongo import MongoClient
from dotenv import load_dotenv
from typing import Optional, Tuple

# --- Configuration ---
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define the expected path to the backend .env file relative to the 'training_engine' root
# This allows the script to be run from the 'twi_speech/training_engine' directory
BACKEND_ENV_PATH = os.path.join(os.path.dirname(__file__), '.env')
LOCAL_AUDIO_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'e_commerce_audio')

# --- Database Connection ---

def load_backend_env(env_path: str) -> None:
    """
    Loads environment variables from the backend's .env file.

    Args:
        env_path (str): The full path to the .env file.
    """
    if not os.path.exists(env_path):
        logging.warning(
            f"Environment file not found at {env_path}. "
            "Ensure this script is run from a context where the path is correct, "
            "or that environment variables are set manually."
        )
        return

    load_dotenv(dotenv_path=env_path)
    logging.info(f"Loaded environment variables from {env_path}")

def get_mongo_client() -> Optional[MongoClient]:
    """
    Establishes a connection to the MongoDB database using credentials
    from environment variables.

    Returns:
        A MongoClient instance if connection is successful, otherwise None.
    """
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        logging.error("MONGODB_URI not found in environment variables.")
        return None

    try:
        logging.info("Connecting to MongoDB...")
        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=10000)
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster')
        logging.info("MongoDB connection successful.")
        return client
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        return None

# --- S3/R2 Connection ---

def get_s3_client() -> Optional[boto3.client]:
    """Initializes a boto3 S3 client for Cloudflare R2."""
    try:
        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        access_key_id = os.getenv("CLOUDFLARE_ACCESS_KEY_ID")
        secret_access_key = os.getenv("CLOUDFLARE_SECRET_ACCESS_KEY")

        if not all([account_id, access_key_id, secret_access_key]):
            logging.error("R2/S3 credentials not found in environment variables.")
            return None

        s3_client = boto3.client(
            service_name='s3',
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name="auto"
        )
        logging.info("Successfully initialized S3 client for R2.")
        return s3_client
    except Exception as e:
        logging.error(f"Failed to initialize S3 client: {e}")
        return None

def generate_presigned_url(s3_client, bucket_name: str, object_key: str, expiration: int = 3600) -> Optional[str]:
    """Generates a pre-signed URL to download an S3 object."""
    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': object_key},
            ExpiresIn=expiration
        )
        return url
    except ClientError as e:
        logging.error(f"Failed to generate pre-signed URL for {object_key}: {e}")
        return None


# --- Data Fetching and Preparation ---

def fetch_transcribed_data(client: MongoClient) -> pd.DataFrame:
    """
    Fetches transcribed recording metadata from MongoDB.
    It specifically looks for records that have been transcribed.

    Args:
        client: An active MongoClient instance.

    Returns:
        A pandas DataFrame containing the fetched data, or an empty DataFrame on error.
    """
    db_name = os.getenv("MONGO_DB_NAME")
    if not db_name:
        logging.error("MONGO_DB_NAME not found in environment variables.")
        return pd.DataFrame()

    db = client[db_name]
    recordings_collection = db["audio_recordings"]

    # Query for documents that are transcribed and have a valid transcription
    query = {
        "transcription_status": "transcribed",
        "transcription": {"$ne": None, "$exists": True}
    }

    try:
        logging.info(f"Fetching transcribed data from collection '{recordings_collection.name}'...")
        cursor = recordings_collection.find(query)
        data = list(cursor)
        logging.info(f"Successfully fetched {len(data)} transcribed records.")
        return pd.DataFrame(data)
    except Exception as e:
        logging.error(f"An error occurred while fetching data from MongoDB: {e}")
        return pd.DataFrame()

def fetch_data_for_training(client: MongoClient) -> pd.DataFrame:
    """
    Fetches all recordings that have a non-empty prompt_text (for command recognition training).
    """
    db_name = os.getenv("MONGO_DB_NAME")
    if not db_name:
        logging.error("MONGO_DB_NAME not found in environment variables.")
        return pd.DataFrame()

    db = client[db_name]
    recordings_collection = db["audio_recordings"]

    query = {
        "prompt_text": {"$exists": True, "$ne": None, "$ne": ""}
    }

    try:
        logging.info(f"Fetching training data from collection '{recordings_collection.name}'...")
        cursor = recordings_collection.find(query)
        data = list(cursor)
        logging.info(f"Successfully fetched {len(data)} records with prompt_text.")
        return pd.DataFrame(data)
    except Exception as e:
        logging.error(f"An error occurred while fetching data from MongoDB: {e}")
        return pd.DataFrame()


def download_audio_files(df: pd.DataFrame, download_dir: str, s3_client, bucket_name: str) -> pd.DataFrame:
    """
    Downloads audio files using pre-signed URLs and saves them locally.
    Adds a 'local_path' column to the DataFrame.

    Args:
        df: DataFrame with an 'object_key' column.
        download_dir: The directory where audio files will be saved.
        s3_client: An active boto3 S3 client.
        bucket_name: The name of the R2 bucket.

    Returns:
        The DataFrame with the added 'local_path' column.
    """
    if df.empty:
        logging.warning("DataFrame is empty. No files to download.")
        return df

    os.makedirs(download_dir, exist_ok=True)
    logging.info(f"Audio files will be saved to: {download_dir}")

    local_paths = []
    total_files = len(df)

    for index, row in df.iterrows():
        object_key = row.get('object_key')

        if not object_key or not isinstance(object_key, str):
            logging.warning(f"Skipping row {index} due to missing or invalid 'object_key'.")
            local_paths.append(None)
            continue

        # Create local path from object key to ensure uniqueness
        try:
            # Sanitize object_key to be a valid local path segment if needed
            filename = os.path.basename(object_key)
            local_path = os.path.join(download_dir, filename)
        except Exception as e:
            logging.error(f"Could not create local path for object key {object_key}: {e}")
            local_paths.append(None)
            continue

        # Check if file already exists to avoid re-downloading
        if os.path.exists(local_path):
            logging.info(f"[{index + 1}/{total_files}] File already exists: {local_path}")
            local_paths.append(local_path)
            continue

        # Generate pre-signed URL and download
        download_url = generate_presigned_url(s3_client, bucket_name, object_key)
        if not download_url:
            logging.warning(f"Could not generate download URL for {object_key}. Skipping.")
            local_paths.append(None)
            continue

        try:
            logging.info(f"[{index + 1}/{total_files}] Downloading: {object_key}")
            response = requests.get(download_url, stream=True, timeout=30)
            response.raise_for_status()

            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            local_paths.append(local_path)
            logging.info(f" -> Saved to: {local_path}")

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download from pre-signed URL for {object_key}: {e}")
            local_paths.append(None)

    df['local_path'] = local_paths
    return df.dropna(subset=['local_path'])

# --- Main Orchestration ---

def load_and_prepare_data() -> Optional[pd.DataFrame]:
    """
    Orchestrates the entire data loading and preparation process.

    1. Loads environment variables.
    2. Connects to the database and S3.
    3. Fetches transcribed metadata.
    4. Downloads the corresponding audio files using pre-signed URLs.

    Returns:
        A pandas DataFrame with metadata and local audio file paths, or None on failure.
    """
    logging.info("--- Starting Data Loading and Preparation ---")

    # 1. Load Environment Variables
    load_backend_env(BACKEND_ENV_PATH)

    # 2. Connect to Services
    mongo_client = get_mongo_client()
    s3_client = get_s3_client()
    bucket_name = os.getenv("R2_BUCKET_NAME")

    if not mongo_client or not s3_client or not bucket_name:
        logging.error("Failed to connect to MongoDB or S3/R2. Aborting.")
        if mongo_client:
            mongo_client.close()
        return None

    # 3. Fetch Data
    metadata_df = fetch_data_for_training(mongo_client)
    if metadata_df.empty:
        logging.warning("No transcribed data found in the database. Exiting.")
        mongo_client.close()
        return None

    # 4. Download Audio Files
    prepared_df = download_audio_files(metadata_df, LOCAL_AUDIO_DIR, s3_client, bucket_name)

    # --- Save metadata CSV for all successfully downloaded files ---
    if not prepared_df.empty:
        metadata_save_path = os.path.join(LOCAL_AUDIO_DIR, "metadata.csv")
        # Select columns to save (add more if needed)
        columns_to_save = ["local_path", "prompt_text", "prompt_id", "participant_code"]
        # Only keep columns that exist in the DataFrame
        columns_to_save = [col for col in columns_to_save if col in prepared_df.columns]
        prepared_df[columns_to_save].to_csv(metadata_save_path, index=False)
        logging.info(f"Saved metadata for training to {metadata_save_path}")

    logging.info(f"--- Data Preparation Complete ---")
    logging.info(f"Successfully processed and downloaded {len(prepared_df)} audio files.")

    mongo_client.close()
    logging.info("MongoDB connection closed.")

    return prepared_df

# --- Example Usage ---

if __name__ == '__main__':
    """
    When run as a script, this will execute the data loading process
    and print the head of the final DataFrame.
    """
    prepared_data = load_and_prepare_data()

    if prepared_data is not None and not prepared_data.empty:
        print("\n--- Sample of Prepared Data ---")
        print(prepared_data.head())
        print("\nColumns:", prepared_data.columns.tolist())
        print(f"\nTotal records processed: {len(prepared_data)}")
    else:
        print("\nNo data was prepared. Check logs for errors.")
