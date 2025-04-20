# app/main.py
from datetime import datetime
import logging
from fastapi import (
    Body, FastAPI, File, UploadFile, Depends, HTTPException, Form, status, Query,
    Path # Import Path for path parameters
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from botocore.exceptions import ClientError
from bson import ObjectId, errors # Import errors
# --- Make sure Pydantic's ValidationError is imported ---
from pydantic import ValidationError

import io
import pandas as pd
from typing import Optional, List

from .config import settings
from .database import connect_to_mongo, close_mongo_connection, get_recordings_collection, get_speakers_collection
from .r2 import delete_multiple_files_from_r2, upload_file_to_r2, get_r2_public_url, delete_file_from_r2
# Import new/updated models and crud functions
from .models import (
    AudioMetadataForm, RecordingDocument, RecordingProgress,SpeakerDocument, UploadResponse, TranscriptionInput, DeleteSummaryResponse, DeleteConfirmationResponse
)
from .crud import (
    check_recording_completion, create_recording_entry, get_recordings_basic as get_recordings,
    get_all_recordings_for_export, update_transcription,
    get_spontaneous_recordings, get_or_create_speaker,
   get_speaker_by_code, get_all_speakers, get_all_speakers_for_export, # <-- Import new speaker CRUD functions,
   delete_all_speakers_from_db
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Twi Speech Data Collection API",
    description="API for uploading Twi audio recordings and metadata.",
    version="1.0.0"
)

# --- Event Handlers for DB Connection ---
@app.on_event("startup")
async def startup_db_client():
    try:
        await connect_to_mongo()
    except Exception as e:
        logger.critical(f"FATAL: Could not connect to MongoDB on startup: {e}")
        import sys
        # sys.exit("MongoDB connection failed on startup.") # Keep commented out for now if preferred

@app.on_event("shutdown")
async def shutdown_db_client():
    await close_mongo_connection()

# --- CORS Middleware ---
# Ensure allowed_origins is correctly fetched
# origins = settings.allowed_origins # Get the list from settings
# logger.info(f"Configuring CORS with origins: {origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins, # Use the property that returns the list
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Dependency for DB Collection ---
def get_collection():
    # Add error handling in case DB is not connected
    try:
        return get_recordings_collection()
    except RuntimeError as e:
        logger.error(f"Database connection error in dependency: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available"
        )

def get_spk_collection():
    try:
        return get_speakers_collection()
    except RuntimeError as e:
        logger.error(f"Database connection error for speakers: {e}")
        raise HTTPException(status_code=503, detail="DB connection error")

# --- API Endpoints ---
@app.get("/", summary="Health Check", tags=["General"])
async def read_root():
    return {"status": "ok", "message": "Welcome to the Twi Speech Data Collection API!"}

# --- Speaker Endpoints ---

@app.get(
   "/speakers/{participant_code}",
   response_model=SpeakerDocument,
   summary="Get Speaker Details by Code",
   tags=["Speakers"],
   responses={404: {"description": "Speaker not found"}}
)
async def get_speaker_details(
   participant_code: str = Path(..., description="The unique code of the participant (e.g., TWI_Speaker_001)"),
   collection = Depends(get_spk_collection)
):
   """Retrieves the details for a specific speaker using their participant code."""
   logger.info(f"Request received for speaker details: {participant_code}")
   try:
       speaker = await get_speaker_by_code(collection, participant_code)
       if speaker is None:
           logger.warning(f"Speaker not found: {participant_code}")
           raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Speaker not found")
       logger.info(f"Speaker found: {participant_code}")
       return speaker
   except HTTPException as http_exc:
       # Re-raise HTTP exceptions (like 404)
       raise http_exc
   except Exception as e:
       logger.exception(f"Failed to retrieve speaker details for {participant_code}")
       raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve speaker details.")

@app.get(
    "/speakers/",
    response_model=List[SpeakerDocument], # Still uses SpeakerDocument, now with optional fields
    summary="List All Speakers",
    tags=["Speakers"]
)
async def list_all_speakers(
    skip: int = Query(0, ge=0, description="Number of records to skip for pagination"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    spk_collection = Depends(get_spk_collection), # Speaker collection dependency
    rec_collection = Depends(get_collection)      # <-- ADD Recording collection dependency
):
    """Retrieves a list of all registered speakers with pagination and recording progress."""
    try:
        # Pass both collections to the updated CRUD function
        speakers = await get_all_speakers(spk_collection, rec_collection, skip=skip, limit=limit)
        return speakers
    except Exception as e:
        logger.exception("Failed to retrieve speakers list.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve speakers list.")


@app.get(
    "/speakers/export/excel",
    summary="Export All Speaker Details to Excel",
    tags=["Speakers"],
    response_class=StreamingResponse
)
async def export_speakers_to_excel(
    collection = Depends(get_spk_collection),    # Speaker collection
    rec_collection = Depends(get_collection)      # <-- ADD Recording collection dependency
):
    """Retrieves all speaker details, including recording progress, and exports them into an Excel (.xlsx) file."""
    try:
        logger.info("Fetching all speaker data for Excel export...")
        # Pass both collections to the updated CRUD function
        speakers_data = await get_all_speakers_for_export(collection, rec_collection)

        if not speakers_data:
            logger.warning("No speaker data found for export.")
            df = pd.DataFrame()
        else:
             logger.info(f"Retrieved {len(speakers_data)} speakers for export.")
             df = pd.DataFrame(speakers_data)
             # Optional: Explicitly set column order for the export if desired
             desired_columns = [
                 'id', 'participant_code', 'dialect', 'age_range', 'gender',
                 'total_recordings', 'recordings_complete', # <-- Added columns
                 'created_at', 'updated_at'
                ]
             # Filter and reorder - handle missing columns gracefully
             df = df.reindex(columns=[col for col in desired_columns if col in df.columns])


        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Speakers')

        output.seek(0)

        filename = f"twi_speakers_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        headers = {
            'Content-Disposition': f'attachment; filename="{filename}"'
        }

        logger.info(f"Successfully generated Excel export: {filename}")
        return StreamingResponse(
            output,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers=headers
        )

    except Exception as e:
        logger.exception("Failed to generate speaker Excel export.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate speaker Excel export.")

# --- Recording Endpoints ---

@app.delete(
    "/speakers/all",
    response_model=DeleteConfirmationResponse, # Use the new response model
    summary="Delete All Speakers (USE WITH EXTREME CAUTION)",
    tags=["Administration"], # Group with other dangerous operations
    status_code=status.HTTP_200_OK,
    responses={
        403: {"description": "Confirmation not provided"},
        500: {"description": "Error during deletion process"}
    }
)
async def delete_all_speakers(
    confirm: bool = Query(..., description="Must explicitly set to true to confirm deletion."),
    collection = Depends(get_spk_collection) # Dependency for the speakers collection
):
    """
    **EXTREME WARNING:** Deletes ALL speaker metadata from the database.
    This action is irreversible and does NOT affect recordings or R2 files.
    Requires `confirm=true` query parameter.
    """
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Deletion not confirmed. Add '?confirm=true' to the URL to proceed."
        )

    logger.warning("!!! Received request to delete ALL speaker data from the database !!!")

    try:
        # Call the CRUD function to delete speakers
        deleted_count = await delete_all_speakers_from_db(collection)

        return DeleteConfirmationResponse(
            message=f"Successfully deleted {deleted_count} speaker documents from the database.",
            deleted_count=deleted_count
        )

    except Exception as e:
        # Log the specific error from CRUD function if needed, but it should log itself
        logger.error(f"Error encountered during delete all speakers operation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during speaker deletion: {e}"
        )


@app.post(
    "/upload/audio",
    response_model=UploadResponse,
    summary="Upload Audio Recording",
    status_code=status.HTTP_201_CREATED,
    tags=["Data Collection"]
)
async def upload_audio_recording(
    # Input form fields (speaker details removed from here)
    participant_code: str = Form(...),
    prompt_id: str = Form(...),
    prompt_text: str = Form(...),
    session_id: Optional[str] = Form(None),
    # --- Add speaker details for potential creation ---
    dialect: Optional[str] = Form(None),
    age_range: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
    # --- End Speaker Details ---
    file: UploadFile = File(..., description="The audio file to upload."),
    rec_collection = Depends(get_collection), # Recordings collection
    spk_collection = Depends(get_spk_collection) # Speakers collection
):
    """Uploads audio, finds/creates speaker, saves recording linked to speaker."""

    # 1. Validate input form data (excluding speaker details now)
    try:
        metadata_input = AudioMetadataForm(
            participant_code=participant_code,
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            session_id=session_id
        )
    except ValidationError as e:
        logger.error(f"Input metadata validation error: {e}")
        raise HTTPException(status_code=422, detail=e.errors())

    logger.info(f"Upload request for participant: {participant_code}, prompt: {prompt_id}")

    file_content_type_lower: Optional[str] = None
    # File Content Type Check (Improved)
    if file.content_type:
        file_content_type_lower = file.content_type.lower()
        # Define allowed types more robustly
        allowed_audio_types = [
            "audio/m4a", "audio/mp4", "audio/aac", # Common for m4a
            "audio/mpeg", "audio/mp3",             # MP3
            "audio/wav", "audio/wave", "audio/x-wav", # WAV
            "audio/ogg",                           # OGG
            "audio/flac", "audio/x-flac",          # FLAC
            "application/octet-stream"              # Allow generic, but log warning
        ]
        if file_content_type_lower not in allowed_audio_types:
            logger.warning(f"Potentially unsupported upload content type: {file.content_type} for file {file.filename}")
            # Decide if you want to block or just warn. Warning is often better.
            # raise HTTPException(status_code=415, detail=f"Unsupported media type: {file.content_type}")
        elif file_content_type_lower == "application/octet-stream":
             logger.warning(f"Received generic content type 'application/octet-stream' for {file.filename}. Proceeding.")
    else:
        logger.warning(f"No content type provided for file {file.filename}. Proceeding.")


    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    try:
        logger.info("Step 2: Getting/Creating speaker...")
        # 2. Get or Create Speaker
        speaker_model, _, speaker_id_obj = await get_or_create_speaker(
                collection=spk_collection,
                participant_code=participant_code,
                dialect=dialect,
                age_range=age_range,
                gender=gender,
        )
        if not speaker_id_obj:
            logger.error("Failed to get speaker ObjectId.")
            raise HTTPException(status_code=500, detail="Internal error obtaining speaker ID.")
        logger.info(f"Step 2 SUCCESS: Speaker ID Obj: {speaker_id_obj}")

        logger.info("Step 3: Uploading file to R2...")
        # 3. Upload file to R2
        file_url, object_key = await upload_file_to_r2(
            file=file,
            participant_code=participant_code, # Use code for path structure
            prompt_id=prompt_id
        )
        # file_size = file.size # Note: file.size might not be reliable after reading
        # It's better to get size from the R2 response if needed, or trust the UploadFile metadata if available before reading.
        # For now, we store it based on initial UploadFile info if available
        file_size = file.size if hasattr(file, 'size') else None

        logger.info(f"Step 3 SUCCESS: File uploaded to R2: {file_url}, Key: {object_key}")


        # 4. Prepare RecordingDocument data (linking to speaker)
        recording_doc_data = RecordingDocument(
            speaker_id=str(speaker_id_obj),
            participant_code=participant_code, # Denormalized
            prompt_id=metadata_input.prompt_id,
            prompt_text=metadata_input.prompt_text,
            session_id=metadata_input.session_id,
            file_url=file_url,
            object_key=object_key,
            filename_original=file.filename,
            content_type=file_content_type_lower, # Use lowercased type
            size_bytes=file_size,
            # recording_duration should come from frontend or post-processing
        )
        logger.info("Step 4: Prepared recording document data.")

        logger.info("Step 5: Creating recording entry in DB...")
        # 5. Insert recording metadata into MongoDB
        recording_db_id = await create_recording_entry(rec_collection, recording_doc_data)
        logger.info(f"Step 5 SUCCESS: Recording entry created: {recording_db_id}")


        logger.info("Step 5b: Checking recording completion...")
        progress_data: RecordingProgress = await check_recording_completion(
                    rec_collection=rec_collection,
                    speaker_id=speaker_id_obj  # Pass the ObjectId
                )
        logger.info(f"Step 5b SUCCESS: Progress checked: {progress_data}")

        logger.info("Step 6: Returning successful response...")
        # 6. Return success response
        return UploadResponse(
            message="Upload successful",
            file_url=file_url,
            recording_db_id=recording_db_id,
            speaker_db_id=str(speaker_id_obj), # Convert speaker ObjectId to string
            participant_code=participant_code,
            prompt_id=prompt_id,
            progress=progress_data
        )

    except ClientError as e: # R2 Error
        error_code = e.response.get("Error", {}).get("Code")
        logger.error(f"R2 Upload Error ({error_code}) for {participant_code}/{prompt_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Cloud storage error: {error_code}")
    except ValueError as e: # Invalid ID format or data validation
        logger.error(f"Input/Validation Error for {participant_code}/{prompt_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e: # General DB or other errors
        logger.exception(f"An unexpected error occurred during file upload for {participant_code}/{prompt_id}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


@app.get(
    "/recordings",
    response_model=List[RecordingDocument],
    summary="List Audio Recordings",
    tags=["Data Collection"]
)
async def list_recordings(
    skip: int = Query(0, ge=0, description="Number of records to skip for pagination"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of records to return"),
    participant_code: Optional[str] = Query(None, description="Filter recordings by participant code"), # Add filter param
    collection = Depends(get_collection)
):
    """Retrieves a list of audio recording metadata entries, optionally filtered by participant."""
    try:
        recordings = await get_recordings(collection, skip=skip, limit=limit, participant_code=participant_code)
        return recordings
    except Exception as e:
        logger.exception("Failed to retrieve recordings.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve recordings.")

@app.get(
    "/recordings/export/excel",
    summary="Export Recordings Metadata to Excel",
    tags=["Data Collection"],
    response_class=StreamingResponse
)
async def export_recordings_to_excel(
    rec_collection = Depends(get_collection),
    spk_collection = Depends(get_spk_collection) # Add speaker collection dependency
):
    """Retrieves all recording metadata, merges speaker details, and exports to Excel."""
    try:
        logger.info("Fetching all recording data for Excel export...")
        recordings_data = await get_all_recordings_for_export(rec_collection, spk_collection)

        if not recordings_data:
             df = pd.DataFrame() # Create empty dataframe if no data
             logger.warning("No recording data found for export.")
        else:
             logger.info(f"Retrieved {len(recordings_data)} recordings for export.")
             df = pd.DataFrame(recordings_data)
             # Ensure columns are in a sensible order (adjust as needed)
             # df = df[desired_column_order_list]

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Recordings')

        output.seek(0)

        filename = f"twi_recordings_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        headers = {
            'Content-Disposition': f'attachment; filename="{filename}"'
        }
        logger.info(f"Successfully generated recordings Excel export: {filename}")
        return StreamingResponse(
            output,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers=headers
        )
    except Exception as e:
        logger.exception("Failed to generate recordings Excel export.")
        raise HTTPException(status_code=500, detail="Failed to generate recordings Excel export.")

@app.get(
    "/recordings/spontaneous",
    response_model=List[RecordingDocument], # Returns a list of recordings
    summary="List Spontaneous Recordings",
    tags=["Data Collection"]
)
async def list_spontaneous_recordings(
    skip: int = Query(0, ge=0, description="Number of records to skip for pagination"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of records to return"),
    collection = Depends(get_collection)
):
    """
    Retrieves a list of spontaneous audio recording metadata entries,
    filtered by prompt IDs starting with 'Spontaneous_', with pagination.
    These are typically the recordings intended for user editing/transcription.
    """
    try:
        recordings = await get_spontaneous_recordings(collection, skip=skip, limit=limit)
        return recordings
    except Exception as e:
        logger.exception("Failed to retrieve spontaneous recordings.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve spontaneous recordings.")


@app.patch(
    "/recordings/{recording_id}/transcription",
    response_model=RecordingDocument,
    summary="Add or Update Transcription",
    tags=["Transcription"],
    responses={
        404: {"description": "Recording not found"},
        400: {"description": "Invalid Recording ID format"},
        422: {"description": "Validation Error (e.g., invalid transcription data)"} # Added 422
    }
)
async def add_or_update_transcription(
    recording_id: str = Path(..., description="The unique ID of the recording to update"),
    transcription_input: TranscriptionInput = Body(...),
    *, # Ensure dependency is keyword-only
    collection = Depends(get_collection)
):
    """Adds or updates the transcription text for a specific recording."""
    logger.info(f"Received transcription update request for recording ID: {recording_id}")
    try:
        updated_recording = await update_transcription(
            collection=collection,
            recording_id_str=recording_id,
            transcription_data=transcription_input
        )

        if updated_recording is None:
            logger.warning(f"Recording ID {recording_id} not found for transcription update.")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recording not found")

        logger.info(f"Successfully processed transcription update for ID: {recording_id}")
        return updated_recording

    except ValueError as e: # Catch InvalidId errors from crud
        logger.error(f"Invalid ID format provided: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ValidationError as e: # Catch Pydantic errors if any in TranscriptionInput
        logger.error(f"Invalid transcription data: {e}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=e.errors())
    except Exception as e:
        logger.exception(f"Failed to update transcription for ID {recording_id}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update transcription.")


@app.delete(
    "/recordings/all",
    response_model=DeleteSummaryResponse,
    summary="Delete All Recordings (USE WITH CAUTION)",
    tags=["Administration"],
    status_code=status.HTTP_200_OK,
    responses={
        403: {"description": "Confirmation not provided"},
        500: {"description": "Error during deletion process"}
    }
)
async def delete_all_recordings(
    confirm: bool = Query(..., description="Must explicitly set to true to confirm deletion."),
    collection = Depends(get_collection)
):
    """
    **WARNING:** Deletes ALL recording metadata from the database AND
    attempts to delete corresponding files from Cloudflare R2.
    This action is irreversible. Requires `confirm=true` query parameter.
    """
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Deletion not confirmed. Add '?confirm=true' to the URL to proceed."
        )

    logger.warning("!!! Initiating deletion of ALL recordings and R2 files !!!")
    r2_failed_keys = []
    r2_attempted_count = 0
    db_deleted_count = None

    try:
        # 1. Get all recording documents (only need object_key and _id)
        logger.info("Fetching all recording keys for deletion...")
        all_recs_cursor = collection.find({}, {"object_key": 1, "_id": 1})
        all_recs = await all_recs_cursor.to_list(length=None)
        r2_attempted_count = len(all_recs)
        logger.info(f"Found {r2_attempted_count} records to process for deletion.")

        if not all_recs:
            return DeleteSummaryResponse(
                message="No recordings found to delete.",
                db_deleted_count=0, # Explicitly set 0
                r2_attempted_count=0,
                r2_failed_keys=[]
            )

        # 2. Attempt to delete files from R2
        logger.info("Attempting to delete files from R2...")
        keys_to_delete_from_r2 = [rec.get("object_key") for rec in all_recs if rec.get("object_key") and rec.get("object_key") != "unknown_key"]

        if keys_to_delete_from_r2:
             # Using batch delete for efficiency
             delete_results = await delete_multiple_files_from_r2(keys_to_delete_from_r2)
             r2_failed_keys = [key for key, success in delete_results.items() if not success]
             # Log sequential delete attempts if needed for debugging:
             # for key in keys_to_delete_from_r2:
             #    success = await delete_file_from_r2(key)
             #    if not success:
             #        r2_failed_keys.append(key)

        if r2_failed_keys:
             logger.warning(f"Failed to delete {len(r2_failed_keys)} objects from R2: {r2_failed_keys}")
        else:
             logger.info("All R2 delete requests processed (or objects didn't exist).")


        # 3. Delete all documents from MongoDB collection
        logger.info("Attempting to delete all documents from MongoDB collection...")
        delete_result = await collection.delete_many({})
        db_deleted_count = delete_result.deleted_count
        logger.info(f"Deleted {db_deleted_count} documents from MongoDB.")

        # 4. Return summary
        final_message = f"Delete process complete. DB Docs Deleted: {db_deleted_count}."
        if r2_failed_keys:
            final_message = f" R2 Deletion Failures: {len(r2_failed_keys)}."
        else:
            final_message = " All associated R2 objects processed successfully."


        return DeleteSummaryResponse(
            message=final_message,
            db_deleted_count=db_deleted_count,
            r2_attempted_count=r2_attempted_count,
            r2_failed_keys=r2_failed_keys
        )

    except Exception as e:
        logger.exception("An error occurred during the delete all process.")
        # Try to return partial info if possible
        return DeleteSummaryResponse(
            message=f"Error during deletion: {e}",
            db_deleted_count=db_deleted_count, # May be None if error occurred before DB delete
            r2_attempted_count=r2_attempted_count,
            r2_failed_keys=r2_failed_keys
        )


# --- Uvicorn Runner ---
if __name__ == "__main__":
    import uvicorn
    import os

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))

    logger.info(f"Starting Uvicorn server on {host}:{port}...")
    uvicorn.run("app.main:app", host=host, port=port, reload=True)
