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
    ghana_tz = pytz.timezone('Africa/Accra') # Assuming you add TZ='Africa/Accra' to .env or keep hardcoded
except pytz.UnknownTimeZoneError:
    ghana_tz = pytz.utc # Fallback to UTC

logger = logging.getLogger(__name__)


def _convert_objectid_to_str(doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Helper to convert _id and speaker_id in a fetched dict to strings."""
    if not doc:
        return None

    # Create a copy to avoid modifying the original
    result = doc.copy()

    # Convert _id to string
    if '_id' in result and isinstance(result['_id'], ObjectId):
        result['_id'] = str(result['_id'])

    # Convert speaker_id to string
    if 'speaker_id' in result and isinstance(result['speaker_id'], ObjectId):
        result['speaker_id'] = str(result['speaker_id'])

    return result

async def check_recording_completion(
    rec_collection: AsyncIOMotorCollection,
    speaker_id: ObjectId,
    required_total: int = EXPECTED_TOTAL_RECORDINGS
) -> RecordingProgress:
    """Check if a participant has completed all required recordings."""
    try:
        total_recordings = await rec_collection.count_documents({
            "speaker_id": speaker_id,
            "transcription_status": {"$ne": "pending"}
        })

        # Count scripted and spontaneous separately
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
        logger.error(f"Error checking recording completion: {e}")
        return RecordingProgress(total_recordings=0, is_complete=False)

async def get_or_create_speaker(
    collection: AsyncIOMotorCollection,
    participant_code: str,
    dialect: Optional[str] = None,
    age_range: Optional[str] = None,
    gender: Optional[str] = None,
) -> Tuple[SpeakerDocument, bool, ObjectId]: # Return speaker model, created flag, AND the actual ObjectId
    """Finds speaker or creates new. Returns Pydantic model (with str ID), created flag, and ObjectId."""
    try:
        existing_speaker_dict = await collection.find_one({"participant_code": participant_code})
        if existing_speaker_dict:
            logger.info(f"Found existing speaker: {participant_code}")
            speaker_id_obj = existing_speaker_dict['_id'] # Get the ObjectId
            speaker_doc = SpeakerDocument(**_convert_objectid_to_str(existing_speaker_dict))
            return speaker_doc, False, speaker_id_obj # Return model, flag, and ObjectId

        # Create new
        logger.info(f"Creating new speaker: {participant_code}")
        # Create Pydantic model first for validation (it won't have an ID yet)
        new_speaker_data = SpeakerDocument(
            participant_code=participant_code,
            dialect=dialect,
            age_range=age_range,
            gender=gender,
        )
        # Get dict for insertion, exclude the 'id' field as it's None
        speaker_dict_to_insert = new_speaker_data.model_dump(
             exclude={'id', 'updated_at'}, # Exclude None fields for insertion
             exclude_none=True,
             by_alias=False # Use model field names for dict keys
        )

        insert_result = await collection.insert_one(speaker_dict_to_insert)
        if not insert_result.acknowledged:
            raise Exception("Speaker insertion not acknowledged.")

        speaker_id_obj = insert_result.inserted_id # Get the new ObjectId
        # Fetch back to get the full doc with _id
        created_speaker_dict = await collection.find_one({"_id": speaker_id_obj})
        if not created_speaker_dict:
             raise Exception("Failed to fetch newly created speaker.")

        speaker_doc = SpeakerDocument(**_convert_objectid_to_str(created_speaker_dict))
        return speaker_doc, True, speaker_id_obj # Return model, flag, and ObjectId

    except ValidationError as e:
        logger.error(f"Validation error during speaker get/create for {participant_code}: {e}")
        raise
    except Exception as e:
        logger.error(f"Database error during speaker get/create for {participant_code}: {e}")
        raise


async def create_recording_entry(
    collection: AsyncIOMotorCollection,
    # Takes speaker_id as ObjectId, converts other data
    recording_data: RecordingDocument # Input model now uses str for IDs
) -> str: # Returns string ID of the created recording
    """Inserts a new recording metadata document."""
    try:
        # Prepare dict for MongoDB, converting speaker_id back to ObjectId
        recording_dict = recording_data.model_dump(
            exclude={'id'}, # Exclude the 'id' field if present
            by_alias=False # Use model field names
        )
        # Convert speaker_id string back to ObjectId for storage
        try:
            recording_dict['speaker_id'] = ObjectId(recording_data.speaker_id)
        except errors.InvalidId:
            raise ValueError(f"Invalid speaker_id format provided: {recording_data.speaker_id}")

        logger.debug(f"Attempting to insert recording metadata: {recording_dict}")
        insert_result = await collection.insert_one(recording_dict)

        if not insert_result.acknowledged:
            raise Exception("MongoDB insertion not acknowledged.")

        inserted_id = str(insert_result.inserted_id)
        logger.info(f"Successfully inserted recording metadata with ID: {inserted_id}")
        return inserted_id

    except ValueError as e: # Catch InvalidId error from conversion
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
        rec_dict_converted = _convert_objectid_to_str(rec_dict_raw) # Convert IDs
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


# Option 2: Fetch with speaker details using $lookup (more advanced)
# This would return a combined document structure, needing a new Pydantic model
# async def get_recordings_with_speaker(...):
#     pipeline = [
#         {'$match': query_filter},
#         {'$sort': {'uploaded_at': -1}},
#         {'$skip': skip},
#         {'$limit': limit},
#         {'$lookup': {
#             'from': 'speakers', # The speakers collection name
#             'localField': 'speaker_id',
#             'foreignField': '_id',
#             'as': 'speaker_info'
#         }},
#         {'$unwind': '$speaker_info'} # Or handle cases where speaker might be missing
#     ]
#     results = await collection.aggregate(pipeline).to_list(length=limit)

# --- NEW FUNCTION for Exporting ---
async def get_all_recordings_for_export(
    rec_collection: AsyncIOMotorCollection,
    spk_collection: AsyncIOMotorCollection
) -> List[Dict[str, Any]]:
    """Retrieves all recordings, converts IDs, merges speaker details."""
    all_recordings_cursor = rec_collection.find()
    recordings_list_raw = await all_recordings_cursor.to_list(length=None)

    # Fetch speakers, convert their IDs for lookup dict
    all_speakers_cursor = spk_collection.find()
    speakers_list_raw = await all_speakers_cursor.to_list(length=None)
    speakers_dict = {}
    for spk_raw in speakers_list_raw:
        spk_converted = _convert_objectid_to_str(spk_raw)
        if spk_converted and spk_converted.get('id'):
            speakers_dict[spk_converted['id']] = spk_converted # Use str id as key

    processed_list = []
    for rec_raw in recordings_list_raw:
        rec = _convert_objectid_to_str(rec_raw) # Convert Recording's IDs
        if not rec: continue

        speaker_id_str = rec.get('speaker_id') # This is now a string
        speaker_info = speakers_dict.get(speaker_id_str, {}) # Lookup using string ID

        # Merge speaker details
        rec['speaker_dialect'] = speaker_info.get('dialect')
        rec['speaker_age_range'] = speaker_info.get('age_range')
        rec['speaker_gender'] = speaker_info.get('gender')

        # Convert datetimes
        for dt_field in ['uploaded_at', 'transcription_updated_at', 'created_at', 'updated_at']: # Include speaker dates too
            if dt_field in rec and isinstance(rec[dt_field], datetime):
                rec[dt_field] = rec[dt_field].isoformat()
            elif dt_field in speaker_info and isinstance(speaker_info[dt_field], datetime):
                 # Add speaker dates if needed for export, prefixing them
                 rec[f'speaker_{dt_field}'] = speaker_info[dt_field].isoformat()


        # Ensure defaults
        rec.setdefault('prompt_text', 'Missing Prompt Text')
        rec.setdefault('transcription', None)
        # ... etc ...

        processed_list.append(rec)

    return processed_list


async def update_transcription(
    collection: AsyncIOMotorCollection,
    recording_id_str: str, # Input is string ID
    transcription_data: TranscriptionInput,
) -> Optional[RecordingDocument]:
    """Updates transcription, converting IDs as needed."""
    try:
        obj_id = ObjectId(recording_id_str) # Convert input string to ObjectId for DB query
    except errors.InvalidId:
        logger.error(f"Invalid ObjectId format provided: {recording_id_str}")
        raise ValueError(f"Invalid recording ID format: {recording_id_str}")

    update_fields = {
        "transcription": transcription_data.transcription,
        "transcription_status": "transcribed",
        "transcription_updated_at": datetime.now(ghana_tz)
    }

    try:
        logger.info(f"Attempting to update transcription for recording ObjectId: {obj_id}")
        updated_document_dict_raw = await collection.find_one_and_update(
            {"_id": obj_id},
            {"$set": update_fields},
            return_document=ReturnDocument.AFTER
        )

        if updated_document_dict_raw:
            logger.info(f"Successfully updated transcription for ID: {recording_id_str}")
            # Convert result's ObjectIds back to str before validating
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
    # Reuse get_recordings_basic, passing the filter (modification needed there)
    # For now, implementing filter here:
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
