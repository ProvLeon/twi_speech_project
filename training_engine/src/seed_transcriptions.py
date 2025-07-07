import os
import logging
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from typing import Optional, Dict

# --- Configuration ---
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define paths relative to this script's location
# Path to the backend .env file to get DB credentials
BACKEND_ENV_PATH = os.path.join(os.path.dirname(__file__), '.env')
# Path to the CSV file containing the prompt transcriptions
PROMPTS_CSV_PATH = os.path.join(os.path.dirname(__file__), 'twi_prompts.csv')

# --- Utility Functions ---

def load_backend_env(env_path: str) -> None:
    """Loads environment variables from the backend's .env file."""
    if not os.path.exists(env_path):
        logging.error(f"Environment file not found at {env_path}. Cannot connect to the database.")
        raise FileNotFoundError(f"Required .env file not found at {env_path}")
    load_dotenv(dotenv_path=env_path)
    logging.info(f"Loaded environment variables from {env_path}")

def get_mongo_client() -> Optional[MongoClient]:
    """Establishes a connection to MongoDB."""
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        logging.error("MONGODB_URI not found in environment variables.")
        return None
    try:
        logging.info("Connecting to MongoDB...")
        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=10000)
        client.admin.command('ismaster')
        logging.info("MongoDB connection successful.")
        return client
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        return None

def load_prompt_map(csv_path: str) -> Dict[str, str]:
    """
    Loads the twi_prompts.csv file and creates a mapping from
    'prompt_id' to its 'meaning' (transcription).
    """
    if not os.path.exists(csv_path):
        logging.error(f"Prompts CSV file not found at: {csv_path}")
        raise FileNotFoundError(f"Prompts CSV file not found at: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
        # Ensure required columns exist
        if 'id' not in df.columns or 'meaning' not in df.columns:
            logging.error("CSV must contain 'id' and 'meaning' columns.")
            return {}

        # Create the dictionary mapping and handle potential duplicates
        # Keep the first meaning found for any duplicate prompt_id
        prompt_map = df.drop_duplicates(subset=['id']).set_index('id')['meaning'].to_dict()
        logging.info(f"Loaded {len(prompt_map)} unique prompt-to-meaning mappings from {csv_path}.")
        return prompt_map
    except Exception as e:
        logging.error(f"Failed to read or process prompts CSV file: {e}")
        return {}

# --- Main Seeding Logic ---

def seed_database_with_transcriptions():
    """
    Main function to update MongoDB records with transcriptions from the CSV file.
    """
    logging.info("--- Starting Transcription Seeding Process ---")
    updated_count = 0
    not_found_count = 0

    try:
        # 1. Load environment variables and connect to DB
        load_backend_env(BACKEND_ENV_PATH)
        client = get_mongo_client()
        if not client:
            return

        # 2. Load the prompt-to-transcription map
        prompt_map = load_prompt_map(PROMPTS_CSV_PATH)
        if not prompt_map:
            logging.warning("Prompt map is empty. Cannot proceed with seeding.")
            return

        # 3. Get the database and collection
        db_name = os.getenv("MONGO_DB_NAME")
        db = client[db_name]
        recordings_collection = db["audio_recordings"]

        # 4. Find all recordings that need transcription
        # Records that are 'pending' or don't have the status field yet
        records_to_update = list(recordings_collection.find({
            "$or": [
                {"transcription_status": "pending"},
                {"transcription_status": {"$exists": False}}
            ]
        }))

        if not records_to_update:
            logging.info("No recordings found with 'pending' status. Database is likely up-to-date.")
            return

        logging.info(f"Found {len(records_to_update)} recordings to process.")

        # 5. Iterate and update
        for record in records_to_update:
            prompt_id = record.get('prompt_id')
            record_id = record.get('_id')

            if not prompt_id:
                logging.warning(f"Record with ID {record_id} is missing a 'prompt_id'. Skipping.")
                continue

            transcription = prompt_map.get(prompt_id)

            if transcription:
                # Update the document in MongoDB
                update_result = recordings_collection.update_one(
                    {"_id": record_id},
                    {
                        "$set": {
                            "transcription": transcription,
                            "transcription_status": "transcribed"
                        }
                    }
                )
                if update_result.modified_count > 0:
                    logging.info(f"Updated record {record_id} (Prompt: {prompt_id}) -> '{transcription}'")
                    updated_count += 1
            else:
                logging.warning(f"No transcription found in CSV for Prompt ID: {prompt_id} (Record ID: {record_id}).")
                not_found_count += 1

    except Exception as e:
        logging.error(f"An unexpected error occurred during the seeding process: {e}", exc_info=True)
    finally:
        if 'client' in locals() and client:
            client.close()
            logging.info("MongoDB connection closed.")

    logging.info("--- Transcription Seeding Process Finished ---")
    logging.info(f"Successfully updated {updated_count} records.")
    if not_found_count > 0:
        logging.warning(f"Could not find transcriptions for {not_found_count} records.")

if __name__ == '__main__':
    # To run this script, navigate to the `training_engine` directory and use:
    # python -m src.e_commerce_command_recognition.seed_transcriptions
    seed_database_with_transcriptions()
