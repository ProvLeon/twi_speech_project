# app/r2.py
import os
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from .config import settings
import logging
from fastapi import UploadFile
import io
import uuid
from typing import Dict, List, Tuple # <-- Import Tuple for type hinting

logger = logging.getLogger(__name__)

# Initialize S3 client for R2
s3_client = boto3.client(
    service_name='s3',
    endpoint_url=settings.r2_endpoint_url,
    aws_access_key_id=settings.CLOUDFLARE_ACCESS_KEY_ID,
    aws_secret_access_key=settings.CLOUDFLARE_SECRET_ACCESS_KEY,
    region_name='auto',
    config=Config(signature_version='s3v4')
)

logger.info(f"Initialized S3 client for R2 endpoint: {settings.r2_endpoint_url}")

def generate_r2_object_key(participant_code: str, prompt_id: str, original_filename: str) -> str:
    """Generates a unique and structured key (path) for the object in R2."""
    file_extension = os.path.splitext(original_filename)[1].lower()
    # Using .m4a as the target based on frontend recorder settings
    safe_extension = ".m4a" if file_extension in ['.m4a', '.mp4'] else file_extension
    if safe_extension not in ['.wav', '.mp3', '.ogg', '.flac', '.m4a']:
        safe_extension = ".m4a" # Default to m4a if still unrecognized
    unique_id = uuid.uuid4()
    # Structure consistent with previous examples
    return f"recordings/{participant_code}/{prompt_id}_{unique_id}{safe_extension}"

def get_r2_public_url(object_key: str) -> str:
    """Constructs the expected public URL for an object in R2."""
    # Ensure your R2 bucket has public access enabled or use a custom domain
    # The bucket name needs to be part of the path for the default public URL
    # Example: https://pub-XXXXXXXXXXXXXXXX.r2.dev/your-bucket-name/your-object-key
    # Check your R2 settings for the correct public URL format.
    account_hash = settings.CLOUDFLARE_ACCOUNT_ID.split('.')[0] # Extract hash if full endpoint is used
    public_host = f"pub-{account_hash}.r2.dev" # Construct the public host

    # *** IMPORTANT: Check if your Cloudflare R2 Public URL includes the bucket name ***
    # If it does (common default):
    return f"https://{public_host}/{settings.R2_BUCKET_NAME}/{object_key}"
    # If you have a custom domain mapped DIRECTLY to the bucket root (less common):
    # return f"https://your-custom-domain.com/{object_key}"
    # If you've configured R2 to serve from the root of pub-....r2.dev (unlikely):
    # return f"https://{public_host}/{object_key}"


async def upload_file_to_r2(
    file: UploadFile,
    participant_code: str,
    prompt_id: str
) -> Tuple[str, str]: # <-- CHANGE 1: Update return type hint to Tuple[str, str]
    """
    Uploads an audio file to Cloudflare R2.

    Args:
        file: The UploadFile object from FastAPI.
        participant_code: Identifier for the speaker.
        prompt_id: Identifier for the specific prompt/recording task.

    Returns:
        A tuple containing:
            - The public URL of the uploaded file in R2.
            - The object key used for the upload.

    Raises:
        ValueError: If file content cannot be read or is empty.
        ClientError: If the upload to R2 fails.
        Exception: For other unexpected errors.
    """
    object_key = None # Initialize object_key
    try:
        contents = await file.read()
        if not contents:
            logger.error("Upload aborted: Received empty file.")
            raise ValueError("Received empty file content.")

        file_stream = io.BytesIO(contents)
        object_key = generate_r2_object_key(participant_code, prompt_id, file.filename) # Generate key

        logger.info(f"Uploading file to R2. Bucket: {settings.R2_BUCKET_NAME}, Key: {object_key}")

        # Determine content type, default if necessary
        content_type = file.content_type or 'application/octet-stream' # Use a generic default if unknown
        if 'm4a' in object_key and content_type == 'application/octet-stream':
            content_type = 'audio/mp4' # Be more specific for .m4a if possible
        logger.debug(f"Using ContentType: {content_type} for upload.")

        s3_client.upload_fileobj(
            Fileobj=file_stream,
            Bucket=settings.R2_BUCKET_NAME,
            Key=object_key,
            ExtraArgs={'ContentType': content_type}
        )

        logger.info(f"Successfully uploaded {object_key} to R2.")

        file_url = get_r2_public_url(object_key) # Generate the URL

        return file_url, object_key # <-- CHANGE 2: Return both url and key

    except ClientError as e:
        logger.error(f"Failed to upload file to R2 (Object Key: {object_key}): {e}")
        raise # Re-raise the specific BotoCore error
    except ValueError as e: # Catch empty file error specifically
        logger.error(f"File processing error during R2 upload (Object Key: {object_key}): {e}")
        raise # Re-raise ValueError
    except Exception as e:
        logger.error(f"An unexpected error occurred during R2 upload (Object Key: {object_key}): {e}")
        raise # Re-raise generic exception
    finally:
        # Ensure file pointer is closed
        if 'file_stream' in locals() and hasattr(file_stream, 'close'):
            file_stream.close()
        # Close the FastAPI UploadFile stream
        await file.close()

async def delete_file_from_r2(object_key: str) -> bool:
    """
    Deletes a single object from the R2 bucket.

    Args:
        object_key: The key of the object to delete.

    Returns:
        True if deletion was successful or object didn't exist, False otherwise.
    """
    if not object_key or object_key == "unknown_key":
         logger.warning(f"Attempted to delete invalid object key: {object_key}")
         return True # Treat invalid key as "nothing to delete"

    logger.info(f"Attempting to delete object from R2: {object_key}")
    try:
        # Boto3 runs sync code, FastAPI handles it in threadpool
        s3_client.delete_object(
            Bucket=settings.R2_BUCKET_NAME,
            Key=object_key
        )
        logger.info(f"Successfully submitted delete request for R2 object: {object_key}")
        # Note: delete_object doesn't raise error if key doesn't exist, it succeeds silently.
        # For stricter checking, you might head_object first, but that adds latency.
        return True
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        logger.error(f"Failed to delete object {object_key} from R2 ({error_code}): {e}")
        return False
    except Exception as e:
         logger.error(f"Unexpected error deleting object {object_key} from R2: {e}")
         return False

# --- Optional: Function for Batch Deletion (More efficient for many files) ---
async def delete_multiple_files_from_r2(object_keys: List[str]) -> Dict[str, bool]:
    """Deletes multiple objects from R2 using batch operation."""
    if not object_keys:
        return {}

    objects_to_delete = [{'Key': key} for key in object_keys if key and key != "unknown_key"]
    if not objects_to_delete:
        logger.info("No valid object keys provided for batch deletion.")
        return {key: True for key in object_keys} # Indicate success as nothing needed deleting

    results = {}
    try:
        logger.info(f"Attempting to batch delete {len(objects_to_delete)} objects from R2...")
        response = s3_client.delete_objects(
            Bucket=settings.R2_BUCKET_NAME,
            Delete={'Objects': objects_to_delete, 'Quiet': False} # Quiet=False returns results
        )
        # Process successful deletions
        deleted_keys = {d['Key'] for d in response.get('Deleted', [])}
        for obj in objects_to_delete:
            results[obj['Key']] = obj['Key'] in deleted_keys

        # Process errors
        errors = response.get('Errors', [])
        for error in errors:
            key = error.get('Key')
            if key:
                results[key] = False
                logger.error(f"Failed to delete {key} during batch operation: {error.get('Code')} - {error.get('Message')}")

        logger.info(f"Batch delete finished. Success: {len(deleted_keys)}, Errors: {len(errors)}")
        # Ensure all original keys have a status
        for key in object_keys:
            if key not in results:
                 results[key] = True # Assume keys not in objects_to_delete were invalid/handled

        return results

    except ClientError as e:
        logger.error(f"Failed during R2 batch delete operation: {e}")
        # Mark all as failed in case of general API error
        return {key: False for key in object_keys}
    except Exception as e:
        logger.error(f"Unexpected error during R2 batch delete: {e}")
        return {key: False for key in object_keys}
