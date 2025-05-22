from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorCursor
from pymongo import ReturnDocument
from pydantic import ValidationError
from .models import RecordingDocument, RecordingProgress, TranscriptionInput, SpeakerDocument
from bson import ObjectId, errors # Keep ObjectId import here
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from .config import settings
import pytz

EXPECTED_TOTAL_RECORDINGS = 163 # Or import
SPONTANEOUS_PROMPTS_COUNT = 8

# Get timezone from config or define directly
try:
    ghana_tz = pytz.timezone('Africa/Accra')
except pytz.UnknownTimeZoneError:
    ghana_tz = pytz.utc # Fallback to UTC

logger = logging.getLogger(__name__)


def _convert_objectid_to_str(doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Helper to convert _id and speaker_id in a fetched dict to strings."""
    if not doc:
        return None
    result = doc.copy()
    if '_id' in result and isinstance(result['_id'], ObjectId):
        result['_id'] = str(result['_id'])
    if 'speaker_id' in result and isinstance(result['speaker_id'], ObjectId):
        result['speaker_id'] = str(result['speaker_id'])
    return result

async def check_recording_completion(
    rec_collection: AsyncIOMotorCollection,
    speaker_id: ObjectId,
    required_total: int = EXPECTED_TOTAL_RECORDINGS
) -> RecordingProgress:
    """Check if a participant has completed all required recordings based on DB entries."""
    try:
        total_recordings = await rec_collection.count_documents({
            "speaker_id": speaker_id,
        })
        scripted_count = await rec_collection.count_documents({
            "speaker_id": speaker_id,
            "prompt_id": {"$not": {"$regex": "^Spontaneous_"}}
        })
        spontaneous_count = await rec_collection.count_documents({
            "speaker_id": speaker_id,
            "prompt_id": {"$regex": "^Spontaneous_"}
        })
        is_complete = (total_recordings >= required_total and
                       spontaneous_count >= SPONTANEOUS_PROMPTS_COUNT and
                       scripted_count >= (required_total - SPONTANEOUS_PROMPTS_COUNT))
        return RecordingProgress(
            total_recordings=total_recordings,
            total_required=required_total,
            is_complete=is_complete
        )
    except Exception as e:
        logger.error(f"Error checking recording completion for speaker {speaker_id}: {e}")
        return RecordingProgress(total_recordings=0, total_required=required_total, is_complete=False)

# --- Speaker CRUD ---

async def get_or_create_speaker(
    collection: AsyncIOMotorCollection,
    participant_code: str,
    dialect: Optional[str] = None,
    age_range: Optional[str] = None,
    gender: Optional[str] = None,
) -> Tuple[SpeakerDocument, bool, ObjectId]: # Return speaker model, created flag, AND the actual ObjectId
    """
    Finds speaker or creates new. If speaker exists and provided details
    (dialect, age_range, gender) are different and not None, updates the speaker record.
    Returns Pydantic model (with str ID), created flag, and ObjectId.
    """
    try:
        existing_speaker_dict = await collection.find_one({"participant_code": participant_code})

        if existing_speaker_dict:
            logger.info(f"Found existing speaker: {participant_code}")
            speaker_id_obj = existing_speaker_dict['_id'] # Get the ObjectId

            # --- Start Update Logic ---
            fields_to_update: Dict[str, Any] = {}
            # Check dialect
            if dialect is not None and dialect != existing_speaker_dict.get('dialect'):
                fields_to_update['dialect'] = dialect
            # Check age_range
            if age_range is not None and age_range != existing_speaker_dict.get('age_range'):
                fields_to_update['age_range'] = age_range
            # Check gender
            if gender is not None and gender != existing_speaker_dict.get('gender'):
                fields_to_update['gender'] = gender

            # If any field needs updating, perform the update
            if fields_to_update:
                fields_to_update['updated_at'] = datetime.now(ghana_tz)
                logger.info(f"Updating speaker {participant_code} with new details: {fields_to_update}")

                updated_speaker_dict_raw = await collection.find_one_and_update(
                    {"_id": speaker_id_obj},
                    {"$set": fields_to_update},
                    return_document=ReturnDocument.AFTER
                )

                if not updated_speaker_dict_raw:
                    # This shouldn't happen if find_one succeeded, but handle defensively
                    logger.error(f"Failed to find speaker {participant_code} during update, though it existed initially.")
                    raise Exception(f"Failed to update speaker {participant_code}")

                # Use the updated document for the rest of the process
                speaker_dict_to_process = updated_speaker_dict_raw
                logger.info(f"Speaker {participant_code} updated successfully.")
            else:
                # No updates needed, use the originally fetched document
                speaker_dict_to_process = existing_speaker_dict
            # --- End Update Logic ---

            # Ensure dates are timezone-aware if necessary before validation (should be ok if stored correctly)
            # Convert final dict (original or updated) for Pydantic model
            speaker_dict_converted = _convert_objectid_to_str(speaker_dict_to_process)
            if not speaker_dict_converted:
                 logger.error(f"Failed to convert speaker document after get/update for {participant_code}")
                 raise Exception("Failed to process speaker document.") # Or handle more gracefully

            speaker_doc = SpeakerDocument(**speaker_dict_converted)
            return speaker_doc, False, speaker_id_obj # Return model, created=False, and ObjectId

        # ---- Create New Speaker Path (No changes needed here) ----
        logger.info(f"Creating new speaker: {participant_code}")
        new_speaker_data = SpeakerDocument(
            participant_code=participant_code,
            dialect=dialect,
            age_range=age_range,
            gender=gender,
        )
        speaker_dict_to_insert = new_speaker_data.model_dump(
             exclude={'id', 'updated_at', 'total_recordings', 'recordings_complete'}, # Exclude calculated/None fields
             exclude_none=True,
             by_alias=False
        )
        # Ensure created_at is explicitly included from model default
        speaker_dict_to_insert['created_at'] = new_speaker_data.created_at

        insert_result = await collection.insert_one(speaker_dict_to_insert)
        if not insert_result.acknowledged:
            raise Exception("Speaker insertion not acknowledged.")

        speaker_id_obj = insert_result.inserted_id
        created_speaker_dict = await collection.find_one({"_id": speaker_id_obj})
        if not created_speaker_dict:
             raise Exception("Failed to fetch newly created speaker.")

        # Convert the newly created document for Pydantic model
        speaker_dict_converted = _convert_objectid_to_str(created_speaker_dict)
        if not speaker_dict_converted:
             logger.error(f"Failed to convert newly created speaker document for {participant_code}")
             raise Exception("Failed to process newly created speaker document.")

        speaker_doc = SpeakerDocument(**speaker_dict_converted)
        return speaker_doc, True, speaker_id_obj # Return model, created=True, and ObjectId

    except ValidationError as e:
        logger.error(f"Validation error during speaker get/create/update for {participant_code}: {e}")
        raise
    except Exception as e:
        logger.error(f"Database error during speaker get/create/update for {participant_code}: {e}")
        raise


# --- Rest of the crud.py functions remain the same ---

async def get_speaker_by_code(
    collection: AsyncIOMotorCollection,
    participant_code: str
) -> Optional[SpeakerDocument]:
    """Retrieves a single speaker by participant code."""
    try:
        speaker_dict_raw = await collection.find_one({"participant_code": participant_code})
        if speaker_dict_raw:
            speaker_dict_converted = _convert_objectid_to_str(speaker_dict_raw)
            if not speaker_dict_converted:
                logger.error(f"Failed to convert speaker document for code {participant_code}")
                return None
            try:
                speaker_doc = SpeakerDocument(**speaker_dict_converted)
                return speaker_doc
            except ValidationError as e:
                doc_id_str = speaker_dict_converted.get('id', 'N/A')
                logger.error(f"Pydantic validation failed for speaker doc ID {doc_id_str} (Code: {participant_code}): {e}")
                return None
        else:
            return None
    except Exception as e:
        logger.error(f"Database error fetching speaker {participant_code}: {e}")
        raise

async def get_all_speakers(
    spk_collection: AsyncIOMotorCollection,
    rec_collection: AsyncIOMotorCollection,
    skip: int = 0,
    limit: int = 100
) -> List[SpeakerDocument]:
    """Retrieves a list of speakers with pagination, including recording progress."""
    speakers_cursor = spk_collection.find({}).skip(skip).limit(limit).sort("created_at", -1)
    db_speakers_raw = await speakers_cursor.to_list(length=limit)

    validated_speakers = []
    for spk_dict_raw in db_speakers_raw:
        spk_dict_converted = _convert_objectid_to_str(spk_dict_raw)
        if not spk_dict_converted: continue

        try:
            validated_doc = SpeakerDocument(**spk_dict_converted)
            speaker_id_obj = spk_dict_raw.get('_id')
            if speaker_id_obj and isinstance(speaker_id_obj, ObjectId):
                progress = await check_recording_completion(rec_collection, speaker_id_obj)
                validated_doc.total_recordings = progress.total_recordings
                validated_doc.recordings_complete = progress.is_complete
            else:
                 logger.warning(f"Could not find valid ObjectId for speaker {spk_dict_converted.get('participant_code', 'N/A')} to check progress.")
                 validated_doc.total_recordings = 0
                 validated_doc.recordings_complete = False
            validated_speakers.append(validated_doc)
        except ValidationError as e:
            doc_id_str = spk_dict_converted.get('id', 'N/A')
            p_code = spk_dict_converted.get('participant_code', 'N/A')
            logger.error(f"Pydantic validation failed for speaker doc ID {doc_id_str} (Code: {p_code}): {e}")
            continue
        except Exception as e:
            doc_id_str = spk_dict_converted.get('id', 'N/A')
            p_code = spk_dict_converted.get('participant_code', 'N/A')
            logger.error(f"Unexpected error processing speaker doc ID {doc_id_str} (Code: {p_code}): {e}")
            continue
    return validated_speakers


async def get_all_speakers_for_export(
    spk_collection: AsyncIOMotorCollection,
    rec_collection: AsyncIOMotorCollection
) -> List[Dict[str, Any]]:
    """Retrieves all speaker documents formatted as dicts for export, including progress."""
    all_speakers_cursor = spk_collection.find({})
    speakers_list_raw = await all_speakers_cursor.to_list(length=None)

    processed_list = []
    for spk_raw in speakers_list_raw:
        spk = _convert_objectid_to_str(spk_raw)
        if not spk: continue

        speaker_id_obj = spk_raw.get('_id')
        if speaker_id_obj and isinstance(speaker_id_obj, ObjectId):
            progress = await check_recording_completion(rec_collection, speaker_id_obj)
            spk['total_recordings'] = progress.total_recordings
            spk['recordings_complete'] = progress.is_complete
        else:
             spk['total_recordings'] = 0
             spk['recordings_complete'] = False
             logger.warning(f"Could not find valid ObjectId for speaker export {spk.get('participant_code', 'N/A')} to check progress.")

        for dt_field in ['created_at', 'updated_at']:
            if dt_field in spk and isinstance(spk[dt_field], datetime):
                # Ensure timezone info before isoformat if needed, though ghana_tz should handle it
                dt_obj = spk[dt_field]
                if dt_obj.tzinfo is None:
                    dt_obj = ghana_tz.localize(dt_obj) # Or assume UTC if necessary
                spk[dt_field] = dt_obj.isoformat()


        spk.setdefault('id', 'N/A')
        spk.setdefault('dialect', None)
        spk.setdefault('age_range', None)
        spk.setdefault('gender', None)
        spk.setdefault('updated_at', None)
        spk.setdefault('total_recordings', 0)
        spk.setdefault('recordings_complete', False)

        processed_list.append(spk)

    return processed_list


# --- Recording CRUD ---

async def create_recording_entry(
    collection: AsyncIOMotorCollection,
    recording_data: RecordingDocument
) -> str:
    """Inserts a new recording metadata document."""
    try:
        recording_dict = recording_data.model_dump(
            exclude={'id'},
            exclude_none=True, # Exclude None fields like transcription initially
            by_alias=False
        )
        try:
            recording_dict['speaker_id'] = ObjectId(recording_data.speaker_id)
        except errors.InvalidId:
            raise ValueError(f"Invalid speaker_id format provided: {recording_data.speaker_id}")

        # Ensure default timestamp is included
        recording_dict['uploaded_at'] = recording_data.uploaded_at
        # Ensure default status is included
        recording_dict['transcription_status'] = recording_data.transcription_status


        logger.debug(f"Attempting to insert recording metadata: {recording_dict}")
        insert_result = await collection.insert_one(recording_dict)

        if not insert_result.acknowledged:
            raise Exception("MongoDB insertion not acknowledged.")

        inserted_id = str(insert_result.inserted_id)
        logger.info(f"Successfully inserted recording metadata with ID: {inserted_id}")
        return inserted_id

    except ValueError as e:
         logger.error(f"Error preparing recording data for DB: {e}")
         raise
    except Exception as e:
        logger.error(f"Failed to insert recording metadata into MongoDB: {e}")
        raise

async def get_recordings_basic(
    collection: AsyncIOMotorCollection,
    skip: int = 0,
    limit: int = 50,
    participant_code: Optional[str] = None
) -> List[RecordingDocument]:
    """Retrieves recording documents, converting ObjectIds to strings."""
    query_filter = {}
    if participant_code:
        query_filter["participant_code"] = participant_code

    recordings_cursor = collection.find(query_filter).skip(skip).limit(limit).sort("uploaded_at", -1)
    db_records_raw = await recordings_cursor.to_list(length=limit)

    validated_recordings = []
    for rec_dict_raw in db_records_raw:
        rec_dict_converted = _convert_objectid_to_str(rec_dict_raw)
        if not rec_dict_converted: continue

        try:
            validated_doc = RecordingDocument(**rec_dict_converted)
            validated_recordings.append(validated_doc)
        except ValidationError as e:
            doc_id_str = rec_dict_converted.get('id', 'N/A')
            logger.error(f"Pydantic validation failed for recording doc ID {doc_id_str}: {e}")
            continue
        except Exception as e:
            doc_id_str = rec_dict_converted.get('id', 'N/A')
            logger.error(f"Unexpected error processing recording doc ID {doc_id_str}: {e}")
            continue

    return validated_recordings


# --- Recording Export ---
async def get_all_recordings_for_export(
    rec_collection: AsyncIOMotorCollection,
    spk_collection: AsyncIOMotorCollection
) -> List[Dict[str, Any]]:
    """Retrieves all recordings, converts IDs, merges speaker details."""
    all_recordings_cursor = rec_collection.find()
    recordings_list_raw = await all_recordings_cursor.to_list(length=None)

    all_speakers_cursor = spk_collection.find()
    speakers_list_raw = await all_speakers_cursor.to_list(length=None)
    speakers_dict = {}
    for spk_raw in speakers_list_raw:
        spk_converted = _convert_objectid_to_str(spk_raw)
        if spk_converted and spk_converted.get('id'):
            speakers_dict[spk_converted['id']] = spk_converted

    processed_list = []
    for rec_raw in recordings_list_raw:
        rec = _convert_objectid_to_str(rec_raw)
        if not rec: continue

        speaker_id_str = rec.get('speaker_id')
        speaker_info = speakers_dict.get(speaker_id_str, {})

        rec['speaker_dialect'] = speaker_info.get('dialect')
        rec['speaker_age_range'] = speaker_info.get('age_range')
        rec['speaker_gender'] = speaker_info.get('gender')

        for dt_field in ['uploaded_at', 'transcription_updated_at']:
            if dt_field in rec and isinstance(rec[dt_field], datetime):
                dt_obj = rec[dt_field]
                if dt_obj.tzinfo is None:
                    dt_obj = ghana_tz.localize(dt_obj) # Or UTC if needed
                rec[dt_field] = dt_obj.isoformat()

        rec.setdefault('prompt_text', 'Missing Prompt Text')
        rec.setdefault('transcription', None)
        rec.setdefault('transcription_status', 'pending')
        # ... other defaults ...

        export_columns = [
            'id', 'speaker_id', 'participant_code', 'prompt_id', 'prompt_text',
            'speaker_dialect', 'speaker_age_range', 'speaker_gender',
            'file_url', 'object_key', 'filename_original', 'content_type',
            'size_bytes', 'recording_duration', 'uploaded_at', 'session_id',
            'transcription', 'transcription_status', 'transcribed_by',
            'transcription_updated_at'
        ]
        export_rec = {col: rec.get(col) for col in export_columns}
        processed_list.append(export_rec)

    return processed_list


# --- Transcription & Spontaneous ---

async def update_transcription(
    collection: AsyncIOMotorCollection,
    recording_id_str: str,
    transcription_data: TranscriptionInput,
) -> Optional[RecordingDocument]:
    """Updates transcription, converting IDs as needed."""
    try:
        obj_id = ObjectId(recording_id_str)
    except errors.InvalidId:
        logger.error(f"Invalid ObjectId format provided: {recording_id_str}")
        raise ValueError(f"Invalid recording ID format: {recording_id_str}")

    update_fields = {
        "transcription": transcription_data.transcription,
        "transcription_status": "transcribed",
        "transcription_updated_at": datetime.now(ghana_tz)
    }
    if transcription_data.transcribed_by: # Optionally update who transcribed it
        update_fields["transcribed_by"] = transcription_data.transcribed_by

    try:
        logger.info(f"Attempting to update transcription for recording ObjectId: {obj_id}")
        updated_document_dict_raw = await collection.find_one_and_update(
            {"_id": obj_id},
            {"$set": update_fields},
            return_document=ReturnDocument.AFTER
        )

        if updated_document_dict_raw:
            logger.info(f"Successfully updated transcription for ID: {recording_id_str}")
            updated_document_dict_converted = _convert_objectid_to_str(updated_document_dict_raw)
            if not updated_document_dict_converted:
                 logger.error(f"Failed to convert updated document after transcription update for ID {recording_id_str}")
                 return None
            try:
                return RecordingDocument(**updated_document_dict_converted)
            except ValidationError as e:
                 logger.error(f"Pydantic validation failed AFTER update for document ID {recording_id_str}: {e}")
                 return None
        else:
            logger.warning(f"Recording ID not found during transcription update: {recording_id_str}")
            return None

    except Exception as e:
        logger.error(f"Failed to update transcription in MongoDB for ID {recording_id_str}: {e}")
        raise


async def get_spontaneous_recordings(
    collection: AsyncIOMotorCollection,
    skip: int = 0,
    limit: int = 50
) -> List[RecordingDocument]:
    """Retrieves spontaneous recordings (already converted)."""
    query_filter = {"prompt_id": {"$regex": "^Spontaneous_"}}
    recordings_cursor = collection.find(query_filter).skip(skip).limit(limit).sort("uploaded_at", -1)
    db_records_raw = await recordings_cursor.to_list(length=limit)
    validated_recordings = []
    for rec_dict_raw in db_records_raw:
        rec_dict_converted = _convert_objectid_to_str(rec_dict_raw)
        if not rec_dict_converted: continue
        try:
            validated_doc = RecordingDocument(**rec_dict_converted)
            validated_recordings.append(validated_doc)
        except ValidationError as e:
             doc_id_str = rec_dict_converted.get('id', 'N/A')
             logger.error(f"Pydantic validation failed for spont. doc ID {doc_id_str}: {e}")
             continue
        except Exception as e:
            doc_id_str = rec_dict_converted.get('id', 'N/A')
            logger.error(f"Unexpected error processing spont. doc ID {doc_id_str}: {e}")
            continue
    return validated_recordings

async def delete_all_speakers_from_db(
    collection: AsyncIOMotorCollection
) -> int:
    """
    Deletes ALL documents from the speakers collection.
    USE WITH EXTREME CAUTION.
    """
    logger.warning("!!! Initiating deletion of ALL speaker documents from the database !!!")
    try:
        delete_result = await collection.delete_many({})
        deleted_count = delete_result.deleted_count
        logger.info(f"Successfully deleted {deleted_count} speaker documents.")
        return deleted_count
    except Exception as e:
        logger.exception("Failed to delete all speaker documents from MongoDB.")
        raise
