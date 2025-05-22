# app/models.py
from pydantic import BaseModel, Field, field_validator # field_validator might be preferred in Pydantic v2+
from typing import List, Optional, Any
from datetime import datetime
import pytz
from bson import ObjectId # Import ObjectId


# Timezone for Ghana
ghana_tz = pytz.timezone('Africa/Accra')

class SpeakerDocument(BaseModel):
    # Use str for the ID field exposed via API
    id: Optional[str] = Field(None, alias="_id", description="MongoDB document ID")
    participant_code: str = Field(...) # Keep unique index constraint in DB
    dialect: Optional[str] = Field(None)
    age_range: Optional[str] = Field(None)
    gender: Optional[str] = Field(None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(ghana_tz))
    updated_at: Optional[datetime] = None
    total_recordings: Optional[int] = Field(None, description="Calculated total recordings by this speaker")
    recordings_complete: Optional[bool] = Field(None, description="Calculated flag indicating if all required recordings are present")

    @field_validator('participant_code')
    def participant_code_must_be_valid(cls, v):
        if not v or not v.startswith("TWI_Speaker_"):
            raise ValueError('participant_code must start with "TWI_Speaker_"')
        return v.strip()

    class Config:
        populate_by_name = True
        # arbitrary_types_allowed = False # No longer needed for ObjectId
        # json_encoders = {} # No longer needed for ObjectId
        json_schema_extra = {
            "example": {
                # Use string representation for ID in examples
                "id": "660a8e7a9f7e3c9d4f8a1b2c",
                "participant_code": "TWI_Speaker_001",
                "dialect": "Asante",
                "age_range": "26-35",
                "gender": "Female",
                "created_at": "2025-04-01T10:00:00+00:00",
                "updated_at": None,
                "recordings_complete": False,
                "total_recordings": 0,
            }
        }

# --- MODIFIED: AudioMetadataForm ---
# Added prompt_text as a required field
class AudioMetadataForm(BaseModel):
    # Participant code is still needed to link/find the speaker
    participant_code: str = Field(...)
    prompt_id: str = Field(...)
    prompt_text: str = Field(...)
    session_id: Optional[str] = Field(None)
     # Speaker details are NOT part of the recording *input* form anymore
     # dialect: Optional[str] = Field(None)
     # age_range: Optional[str] = Field(None)
     # gender: Optional[str] = Field(None)

    @field_validator('participant_code')
    def participant_code_must_be_valid(cls, v):
        # Keep validation here for the input form
        if not v or not v.startswith("TWI_Speaker_"):
            raise ValueError('participant_code must start with "TWI_Speaker_"')
        return v.strip()

# --- TranscriptionInput remains the same ---
class TranscriptionInput(BaseModel):
    transcription: str = Field(..., description="The transcribed text for the audio recording.")

class RecordingProgress(BaseModel):
    total_recordings: int
    total_required: int = Field(default=163)  # Your expected total
    is_complete: bool = Field(default=False)


# --- RecordingDocument (inherits the new prompt_text field) ---
class RecordingDocument(BaseModel):
    id: Optional[str] = Field(None, alias="_id", description="MongoDB document ID")
    # Use str for the speaker_id when representing in API model
    speaker_id: str = Field(...)
    participant_code: str = Field(...)
    prompt_id: str = Field(...)
    prompt_text: str = Field(...)
    session_id: Optional[str] = Field(None)
    file_url: str = Field(...)
    object_key: str = Field(...)
    filename_original: str = Field(...)
    content_type: Optional[str] = Field(None)
    size_bytes: Optional[int] = Field(None)
    recording_duration: Optional[int] = Field(None, description="Duration in milliseconds")
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(ghana_tz))

    # Transcription Fields
    transcription: Optional[str] = Field(None)
    transcription_status: str = Field(default="pending")
    transcribed_by: Optional[str] = Field(None)
    transcription_updated_at: Optional[datetime] = Field(None)

    class Config:
        populate_by_name = True
        # arbitrary_types_allowed = False
        # json_encoders = {}
        json_schema_extra = {
            "example": {
                # Use strings for IDs in examples
                "id": "65cbb8d1f7e5a6c9f0b1d1e1",
                "speaker_id": "660a8e7a9f7e3c9d4f8a1b2c",
                "participant_code": "TWI_Speaker_001",
                "prompt_id": "ScriptA_10",
                "prompt_text": "Wo din de sɛn?",
                "session_id": "session_abc_123",
                "file_url": "https://...",
                "object_key": "recordings/...",
                "filename_original": "...",
                "content_type": "audio/mp4",
                "size_bytes": 95582,
                "recording_duration": 5320,
                "uploaded_at": "2025-04-11T10:30:00+00:00",
                "transcription": "What is your name?",
                "transcription_status": "transcribed",
                "transcribed_by": "validator_01",
                "transcription_updated_at": "2025-04-12T14:00:00+00:00"
            }
        }

class DeleteSummaryResponse(BaseModel):
    message: str
    db_deleted_count: Optional[int] = None
    r2_attempted_count: int
    r2_failed_keys: List[str]

class DeleteConfirmationResponse(BaseModel):
    """Simple response for confirming deletion operations."""
    message: str
    deleted_count: int

class UploadResponse(BaseModel):
    message: str
    file_url: str
    recording_db_id: str
    speaker_db_id: str
    participant_code: str
    prompt_id: str
    progress: RecordingProgress
