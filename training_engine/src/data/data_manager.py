"""
Data Management Module for Twi Speech Recognition Training

This module handles data loading, preprocessing, and management for training
Twi speech-to-text models. It integrates with the existing backend to fetch
audio recordings and transcriptions from MongoDB and Cloudflare R2.

Key Features:
- Integration with existing MongoDB/R2 backend
- Audio preprocessing and feature extraction
- Text normalization for Twi language
- Data augmentation pipeline
- Efficient data loading with caching
- Multi-dialect support
- Quality assurance and filtering

Author: Twi Speech Recognition Team
"""

import os
import sys
import logging
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import pickle
from collections import defaultdict
from datetime import datetime
import shutil

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import librosa
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from .script_parser import TwiPrompt, ScriptParser
from ..utils.async_utils import run_async, ensure_async
from ..utils.simple_async import run_async_fresh
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Training dataset classes will be imported when needed to avoid circular import
create_dataset = None
TwiSpeechToTextDataset = None
TwiSpeechTranslationDataset = None
TwiMultilingualDataset = None
TwiCrossLingualDataset = None

# Audio recording
try:
    import sounddevice as sd
except ImportError:
    sd = None
    logger.warning("sounddevice not installed. Audio recording functionality will be disabled.")

# Database connections
import motor.motor_asyncio
import boto3
from botocore.exceptions import ClientError

# Text processing
import re
import unicodedata
from transformers import Wav2Vec2Processor

logger = logging.getLogger(__name__)


# Import shared data structures
from .structures import (
    AudioSample,
    DatasetStats,
    AudioRecording,
    RecordingSession,
    TrainingExample,
    ProcessingConfig,
    TrainingConfig
)
class TwiTextProcessor:
    """Text preprocessing for Twi language"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tone_marks = ['́', '̀', '̂', '̃', '̄', '̆', '̇']

        # Twi orthography normalization mappings
        self.orthography_mappings = {
            'ɔ': 'ɔ',  # Ensure consistent encoding
            'ɛ': 'ɛ',
            'ŋ': 'ŋ',
            # Add more mappings as needed
        }

        # Common contractions and abbreviations
        self.contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am",
        }

    def normalize_text(self, text: str) -> str:
        """Normalize Twi text"""
        if not text:
            return ""

        # Unicode normalization
        text = unicodedata.normalize('NFC', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # Standardize orthography
        if self.config.get('standardize_orthography', True):
            for old, new in self.orthography_mappings.items():
                text = text.replace(old, new)

        # Handle tone marks
        if not self.config.get('preserve_tone_marks', True):
            for tone in self.tone_marks:
                text = text.replace(tone, '')

        # Handle punctuation
        if self.config.get('remove_punctuation', False):
            text = re.sub(r'[^\w\s]', '', text)

        # Case normalization
        if self.config.get('lowercase', True):
            text = text.lower()

        # Handle contractions (for code-switching)
        if self.config.get('expand_contractions', True):
            for contraction, expansion in self.contractions.items():
                text = text.replace(contraction, expansion)

        return text

    def is_valid_text(self, text: str) -> bool:
        """Check if text is valid for training"""
        if not text or len(text.strip()) == 0:
            return False

        # Check length constraints
        min_length = self.config.get('min_text_length', 1)
        max_length = self.config.get('max_text_length', 500)

        if not (min_length <= len(text) <= max_length):
            return False

        # Check for invalid characters
        invalid_chars = self.config.get('invalid_characters', [])
        if any(char in text for char in invalid_chars):
            return False

        return True


class TwiAudioProcessor:
    """Audio preprocessing for Twi speech"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.target_sr = config.get('target_sample_rate', 16000)
        self.target_channels = config.get('target_channels', 1)

        # Audio augmentation pipeline
        self.augmentation_pipeline = self._create_augmentation_pipeline()

    def _create_augmentation_pipeline(self):
        """Create audio augmentation pipeline"""
        if not self.config.get('augmentation', {}).get('enabled', False):
            return None

        augmentations = []

        # Gaussian noise
        if self.config.get('augmentation', {}).get('gaussian_noise', {}).get('enabled', False):
            augmentations.append(
                AddGaussianNoise(
                    min_amplitude=0.001,
                    max_amplitude=0.015,
                    p=0.3
                )
            )

        # Time stretching
        if self.config.get('augmentation', {}).get('time_stretch', {}).get('enabled', False):
            augmentations.append(
                TimeStretch(
                    min_rate=0.8,
                    max_rate=1.2,
                    p=0.3
                )
            )

        # Pitch shift
        if self.config.get('augmentation', {}).get('pitch_shift', {}).get('enabled', False):
            augmentations.append(
                PitchShift(
                    min_semitones=-2,
                    max_semitones=2,
                    p=0.3
                )
            )

        return Compose(augmentations) if augmentations else None

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file"""
        try:
            # Try different audio loading methods
            try:
                audio, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
            except Exception:
                audio, sr = sf.read(audio_path)
                if sr != self.target_sr:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
                    sr = self.target_sr

            # Ensure mono audio
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            return audio, sr

        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            raise

    def preprocess_audio(self, audio: np.ndarray, sr: int, apply_augmentation: bool = False) -> np.ndarray:
        """Preprocess audio"""
        # Normalize audio
        if self.config.get('normalize_audio', True):
            audio = librosa.util.normalize(audio)

        # Remove silence
        if self.config.get('trim_silence', True):
            audio, _ = librosa.effects.trim(audio, top_db=20)

        # Apply augmentation if enabled
        if apply_augmentation and self.augmentation_pipeline:
            audio = self.augmentation_pipeline(samples=audio, sample_rate=sr)

        return audio

    def extract_features(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract audio features"""
        features = {}

        # Raw audio
        features['audio'] = audio

        # Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=80,
            fmax=8000
        )
        features['mel_spectrogram'] = librosa.power_to_db(mel_spec)

        # MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=13
        )
        features['mfcc'] = mfccs

        return features

    def validate_audio_quality(self, audio: np.ndarray, sr: int) -> Tuple[bool, float]:
        """Validate audio quality"""
        # Check duration
        duration = len(audio) / sr
        min_duration = self.config.get('min_duration', 0.5)
        max_duration = self.config.get('max_duration', 30.0)

        if not (min_duration <= duration <= max_duration):
            return False, 0.0

        # Check for silence
        if np.max(np.abs(audio)) < 1e-4:
            return False, 0.0

        # Calculate quality score (simplified)
        # In practice, you might use more sophisticated metrics
        rms = np.sqrt(np.mean(audio**2))
        snr_estimate = 20 * np.log10(rms / (np.std(audio) + 1e-10))

        quality_score = np.clip(snr_estimate / 20.0, 0.0, 1.0)

        return quality_score > 0.3, quality_score


class TwiSpeechDataset(Dataset):
    """PyTorch Dataset for Twi speech data"""

    def __init__(self, samples: List[AudioSample], processor: Wav2Vec2Processor,
                 audio_processor: TwiAudioProcessor, text_processor: TwiTextProcessor,
                 is_training: bool = True):
        self.samples = samples
        self.processor = processor
        self.audio_processor = audio_processor
        self.text_processor = text_processor
        self.is_training = is_training

        logger.info(f"Created dataset with {len(samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        try:
            # Load and preprocess audio
            audio, sr = self.audio_processor.load_audio(sample.audio_path)
            audio = self.audio_processor.preprocess_audio(
                audio, sr, apply_augmentation=self.is_training
            )

            # Process text
            text = self.text_processor.normalize_text(sample.transcription)

            # Tokenize
            inputs = self.processor(
                audio,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=320000  # 20 seconds at 16kHz
            )

            # Tokenize labels using the tokenizer directly
            labels = self.processor.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512  # Maximum text length
            )

            return {
                'input_values': inputs.input_values.squeeze(),
                'attention_mask': inputs.attention_mask.squeeze(),
                'labels': labels.input_ids.squeeze(),
                'speaker_id': sample.speaker_id,
                'dialect': sample.dialect,
            }

        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            # Return a dummy sample to avoid breaking the batch
            return self._get_dummy_sample()

    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """Return a dummy sample for error cases"""
        dummy_audio = np.zeros(16000)  # 1 second of silence
        dummy_text = "dummy"

        inputs = self.processor(
            dummy_audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        # Use tokenizer directly for text processing
        labels = self.processor.tokenizer(
            dummy_text,
            return_tensors="pt",
            padding=True
        )

        return {
            'input_values': inputs.input_values.squeeze(),
            'attention_mask': inputs.attention_mask.squeeze(),
            'labels': labels.input_ids.squeeze(),
            'speaker_id': "dummy",
            'dialect': "unknown",
        }


class TwiDataManager:
    """Main data management class"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.text_processor = TwiTextProcessor(config.get('text', {}))
        self.audio_processor = TwiAudioProcessor(config.get('audio', {}))

        # Script parser for recording
        self.script_path = self.config.get('script_file', 'script_actual.ts')
        self.script_parser = None
        self.sections = []
        self.prompts = []
        if os.path.exists(self.script_path):
            try:
                self.script_parser = ScriptParser(self.script_path)
                self.sections, self.prompts = self.script_parser.parse_script()
            except Exception as e:
                logger.error(f"Failed to parse script file {self.script_path}: {e}")
        else:
            logger.warning(f"Script file not found at {self.script_path}, recording from script will not be available.")

        # Recording state
        self.is_recording = False
        self.current_recording = None
        self.recordings: List[AudioSample] = []
        self.current_session: Optional[RecordingSession] = None

        # Audio device check
        self._check_audio_device()

        # Audio settings
        self.sample_rate = self.config.get('audio', {}).get('sample_rate', 16000)
        self.channels = 1
        self.dtype = np.float32

        # Database connections
        self.mongo_client = None
        self.r2_client = None
        self.database = None
        self.use_fallback_data = False
        self.bucket_name = None

        # Cache
        self.cache_dir = Path(self.config.get('cache_dir', 'data/cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize connections (gracefully handle failures)
        self._initialized = False
        self.mongo_client = None
        self.database = None
        self.r2_client = None

        # Defer actual connection initialization until needed
        # This avoids event loop issues during __init__
        self._connection_attempted = False

    async def _initialize_connections(self):
        """Initialize database connections"""
        if self._connection_attempted:
            return

        self._connection_attempted = True

        try:
            # MongoDB connection with environment variable fallbacks
            mongodb_config = self.config.get('mongodb', {})
            connection_uri = mongodb_config.get('uri') or os.getenv('MONGODB_URI')

            # Clean up the connection URI - remove None values and empty strings
            if connection_uri:
                connection_uri = connection_uri.strip()
                if not connection_uri or connection_uri == 'None':
                    connection_uri = None
            database_name = (
                mongodb_config.get('database') or
                os.getenv('MONGO_DB_NAME', 'twi_speech_data')
            )

            if not connection_uri:
                logger.warning("No MongoDB URI found. Set MONGODB_URI env var. Falling back to dummy data.")
                self.use_fallback_data = True
                return

            try:
                self.mongo_client = motor.motor_asyncio.AsyncIOMotorClient(
                    connection_uri,
                    serverSelectionTimeoutMS=5000
                )
                self.database = self.mongo_client[database_name]
                await self.mongo_client.admin.command('ping')
                logger.info("MongoDB connection established")
            except Exception as mongo_error:
                logger.warning(f"MongoDB connection failed: {mongo_error}. Falling back to dummy data.")
                self.use_fallback_data = True
                return

            # R2 connection with environment variable fallbacks
            r2_config = self.config.get('r2_storage', {})
            endpoint_url = r2_config.get('endpoint_url') or os.getenv('R2_ENDPOINT_URL')
            access_key = r2_config.get('access_key') or os.getenv('CLOUDFLARE_ACCESS_KEY_ID')
            secret_key = r2_config.get('secret_key') or os.getenv('CLOUDFLARE_SECRET_ACCESS_KEY')
            self.bucket_name = r2_config.get('bucket_name') or os.getenv('R2_BUCKET_NAME')

            if not all([endpoint_url, access_key, secret_key, self.bucket_name]):
                logger.warning("R2 credentials not fully configured. Audio downloads will be skipped.")
                return

            try:
                self.r2_client = boto3.client(
                    's3',
                    endpoint_url=endpoint_url,
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    region_name='auto'
                )
                logger.info("R2 connection established")
            except Exception as r2_error:
                logger.warning(f"R2 connection failed: {r2_error}. Audio downloads will be skipped.")
        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
            logger.info("Falling back to dummy data for development")
            self.use_fallback_data = True

    async def fetch_data_from_backend(self) -> List[AudioSample]:
        """Fetch data from the existing backend"""
        # Ensure connections are initialized
        if not self._connection_attempted:
            await self._initialize_connections()
            self._initialized = True

        samples = []

        # Use fallback data if database is not available
        if self.use_fallback_data or self.database is None:
            logger.info("Database not available, skipping backend fetch.")
            return []

        try:
            # Fetch recordings from MongoDB
            recordings_collection = self.database.get_collection('audio_recordings')
            speakers_collection = self.database.get_collection('speakers')

            # Query filters
            filters = self.config.get('mongodb', {}).get('filters', {})
            query = {}

            if filters.get('transcription_status'):
                query['transcription_status'] = filters['transcription_status']

            if filters.get('min_duration'):
                query['recording_duration'] = {'$gte': filters['min_duration'] * 1000}  # Convert to ms

            if filters.get('exclude_empty_transcriptions'):
                query['transcription'] = {'$ne': None, '$ne': ''}

            # Fetch recordings
            cursor = recordings_collection.find(query)
            recordings = await cursor.to_list(length=None)

            # Fetch speakers
            speakers_cursor = speakers_collection.find()
            speakers = await speakers_cursor.to_list(length=None)
            speakers_dict = {str(s['_id']): s for s in speakers}

            # Process recordings
            for recording in recordings:
                logging.info(f"Processing recording {recording}")
                speaker_id = str(recording.get('speaker_id', ''))
                speaker_info = speakers_dict.get(speaker_id, {})

                # Download audio file
                audio_path = await self._download_audio_file(recording)

                if audio_path:
                    # Get prompt text and prompt_id from recording metadata
                    prompt_text = recording.get('prompt_text', '')
                    prompt_id = recording.get('prompt_id', '')

                    # Use transcription field as the meaning (English translation)
                    # If transcription is null or empty, fetch from script_actual.ts
                    transcription = recording.get('transcription')
                    meaning = transcription if transcription else None

                    # Log metadata source
                    if meaning:
                        logger.debug(f"Using transcription from MongoDB for {prompt_id}: {meaning[:50]}...")

                    # If transcription is empty/null and we have prompt_id, fetch from script parser
                    if not meaning and prompt_id and self.script_parser:
                        # Look up the prompt by ID
                        prompt = self.script_parser.get_prompt_by_id(prompt_id)
                        if prompt:
                            meaning = prompt.meaning
                            logger.info(f"Fetched meaning from script_actual.ts for {prompt_id}: {meaning[:50]}...")
                            # Also use prompt text from script if not in MongoDB
                            if not prompt_text:
                                prompt_text = prompt.text
                                logger.debug(f"Using prompt text from script_actual.ts for {prompt_id}")

                    if not meaning:
                        logger.warning(f"No English translation found for prompt_id: {prompt_id}")

                    # Handle duration - ensure it's not None
                    duration = recording.get('recording_duration', 0)
                    if duration is None:
                        duration = 0
                    duration = float(duration) / 1000.0  # Convert to seconds

                    # Extract original filename from object_key
                    object_key = recording.get('object_key', '')
                    original_filename = object_key.split('/')[-1] if object_key else None

                    # Try to load cached metadata to enrich the sample data
                    cached_metadata = None
                    if audio_path:
                        cached_metadata = self._load_cached_metadata(audio_path)
                        if cached_metadata:
                            logger.info(f"✅ Found cached metadata for {prompt_id}, enriching sample data")
                            logger.debug(f"Cached metadata keys: {list(cached_metadata.keys())}")

                            # Log original values
                            logger.debug(f"Before enrichment - prompt_text: '{prompt_text}', meaning: '{meaning}', duration: {duration}")

                            # Use cached data to fill in missing information
                            if not prompt_text and cached_metadata.get('prompt_text'):
                                prompt_text = cached_metadata['prompt_text']
                                logger.info(f"✓ Updated prompt_text from cache: '{prompt_text[:50]}...'")
                            if not meaning and cached_metadata.get('transcription'):
                                meaning = cached_metadata['transcription']
                                logger.info(f"✓ Updated meaning from cache: '{meaning[:50]}...' if meaning else 'None'")
                            if not original_filename and cached_metadata.get('filename_original'):
                                original_filename = cached_metadata['filename_original']
                                logger.info(f"✓ Updated original_filename from cache: '{original_filename}'")
                            if not duration and cached_metadata.get('recording_duration'):
                                duration = float(cached_metadata['recording_duration']) / 1000.0
                                logger.info(f"✓ Updated duration from cache: {duration}s")

                            # Log final values
                            logger.debug(f"After enrichment - prompt_text: '{prompt_text}', meaning: '{meaning}', duration: {duration}")
                        else:
                            logger.warning(f"❌ No cached metadata found for {prompt_id} at {audio_path}")

                    sample = AudioSample(
                        audio_path=audio_path,
                        transcription=meaning,  # This is the English translation (from transcription or script)
                        duration=duration,
                        speaker_id=speaker_id,
                        dialect=speaker_info.get('dialect', 'unknown'),
                        gender=speaker_info.get('gender'),
                        age_range=speaker_info.get('age_range'),
                        prompt_id=prompt_id,
                        prompt_text=prompt_text,  # Also store as prompt_text
                        meaning=meaning,  # This is the English translation (from transcription or script)
                        session_id=recording.get('session_id'),
                        original_filename=original_filename,
                    )

                    # Validate sample
                    if self._validate_sample(sample):
                        samples.append(sample)
                        logger.debug(f"Added sample: {prompt_id} - {prompt_text[:30]}... -> {meaning[:30] if meaning else 'NO TRANSLATION'}...")

            # Summary statistics
            samples_with_transcription = sum(1 for s in samples if s.meaning)
            samples_from_mongodb = sum(1 for s in samples for r in recordings if r.get('_id') and r.get('transcription') and s.prompt_id == r.get('prompt_id'))
            samples_from_script = samples_with_transcription - samples_from_mongodb

            logger.info(f"Fetched {len(samples)} valid samples from backend")
            logger.info(f"  - {samples_with_transcription} samples have English translations")
            logger.info(f"  - {samples_from_mongodb} translations from MongoDB")
            logger.info(f"  - {samples_from_script} translations fetched from script_actual.ts")
            logger.info(f"  - {len(samples) - samples_with_transcription} samples missing translations")

        except Exception as e:
            logger.error(f"Failed to fetch data from backend: {e}")
            return []

        return samples

    def organize_downloaded_data(self, samples: List[AudioSample]) -> List[AudioSample]:
        """Organize downloaded audio files and metadata into recordings directory structure"""
        if not samples:
            return samples

        recordings_dir = self.config.get('recordings_dir', 'data/recordings')
        organized_samples = []

        logger.info(f"Organizing {len(samples)} samples into {recordings_dir}")

        for sample in samples:
            try:
                # Create speaker directory
                speaker_dir = Path(recordings_dir) / sample.speaker_id
                speaker_dir.mkdir(parents=True, exist_ok=True)

                # Copy audio file from cache to recordings directory
                if sample.audio_path and Path(sample.audio_path).exists():
                    # Try to load cached metadata first
                    cached_metadata = self._load_cached_metadata(sample.audio_path)

                    # Use original filename if available, otherwise generate one
                    if sample.original_filename:
                        new_filename = sample.original_filename
                    else:
                        # Generate filename from prompt_id and timestamp
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        extension = Path(sample.audio_path).suffix or '.wav'
                        new_filename = f"{sample.speaker_id}_{sample.prompt_id}_{timestamp}{extension}"

                    # Copy audio file
                    new_audio_path = speaker_dir / new_filename
                    import shutil
                    logger.info(f"Copying audio file: {sample.audio_path} -> {new_audio_path}")
                    shutil.copy2(sample.audio_path, new_audio_path)
                    logger.info(f"✓ Successfully copied audio file to recordings directory")

                    # Create metadata JSON file
                    metadata_filename = new_filename.rsplit('.', 1)[0] + '_metadata.json'
                    metadata_path = speaker_dir / metadata_filename

                    # Prepare metadata - combine sample data with cached metadata
                    # Use cached metadata values as primary source when available
                    metadata = {
                        'prompt_id': cached_metadata.get('prompt_id', sample.prompt_id) if cached_metadata else sample.prompt_id,
                        'prompt_text': cached_metadata.get('prompt_text', sample.prompt_text) if cached_metadata else sample.prompt_text,
                        'meaning': sample.meaning,
                        'transcription': sample.transcription,
                        'speaker_id': cached_metadata.get('speaker_id', sample.speaker_id) if cached_metadata else sample.speaker_id,
                        'dialect': sample.dialect,
                        'gender': sample.gender,
                        'age_range': sample.age_range,
                        'duration': sample.duration,
                        'session_id': cached_metadata.get('session_id', sample.session_id) if cached_metadata else sample.session_id,
                        'quality_score': sample.quality_score,
                        'original_filename': cached_metadata.get('filename_original', sample.original_filename) if cached_metadata else sample.original_filename,
                        'organized_at': datetime.now().isoformat(),
                        'source': 'cloud_backend'
                    }

                    # Add cached metadata fields if available
                    if cached_metadata:
                        metadata.update({
                            'cached_at': cached_metadata.get('cached_at'),
                            'mongodb_transcription': cached_metadata.get('transcription'),
                            'object_key': cached_metadata.get('object_key'),
                            'content_type': cached_metadata.get('content_type'),
                            'size_bytes': cached_metadata.get('size_bytes'),
                            'recording_duration': cached_metadata.get('recording_duration'),
                            'uploaded_at': cached_metadata.get('uploaded_at'),
                            'transcription_status': cached_metadata.get('transcription_status'),
                            'transcribed_by': cached_metadata.get('transcribed_by'),
                            'transcription_updated_at': cached_metadata.get('transcription_updated_at'),
                        })
                        logger.debug(f"Enhanced metadata with cached data for {sample.prompt_id}")
                    else:
                        logger.warning(f"No cached metadata found for {sample.prompt_id} during organization")

                    # Save metadata
                    logger.info(f"Saving metadata: {metadata_path}")
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                    logger.info(f"✓ Successfully saved metadata file")

                    # Update sample with new path
                    organized_sample = AudioSample(
                        audio_path=str(new_audio_path),
                        transcription=sample.transcription,
                        duration=sample.duration,
                        speaker_id=sample.speaker_id,
                        dialect=sample.dialect,
                        gender=sample.gender,
                        age_range=sample.age_range,
                        prompt_id=sample.prompt_id,
                        prompt_text=sample.prompt_text,
                        meaning=sample.meaning,
                        session_id=sample.session_id,
                        quality_score=sample.quality_score,
                        original_filename=sample.original_filename,
                    )

                    organized_samples.append(organized_sample)
                    logger.info(f"✅ Organized {sample.prompt_id}: {Path(sample.audio_path).name} -> {new_audio_path.name}")

                else:
                    logger.warning(f"Audio file not found for sample {sample.prompt_id}: {sample.audio_path}")
                    organized_samples.append(sample)

            except Exception as e:
                logger.error(f"Failed to organize sample {sample.prompt_id}: {e}")
                organized_samples.append(sample)

        logger.info(f"🎉 Successfully organized {len(organized_samples)} samples into recordings directory: {recordings_dir}")

        # Show summary of what was organized
        if organized_samples:
            speakers_organized = set(s.speaker_id for s in organized_samples)
            logger.info(f"📊 Organization summary:")
            logger.info(f"   - Speakers: {len(speakers_organized)} ({', '.join(speakers_organized)})")
            logger.info(f"   - Total files: {len(organized_samples)}")
            logger.info(f"   - Recordings directory: {Path(recordings_dir).absolute()}")

            # Check if files actually exist
            existing_files = 0
            for sample in organized_samples:
                if Path(sample.audio_path).exists():
                    existing_files += 1
            logger.info(f"   - Files verified on disk: {existing_files}/{len(organized_samples)}")

        # Create summary index
        self._create_data_index(organized_samples, recordings_dir)

        return organized_samples

    def process_audio_batch(self, samples: List[AudioSample], apply_augmentation: bool = False, max_workers: int = 4) -> List[AudioSample]:
        """Process a batch of audio samples with optional augmentation"""
        logger.info(f"Processing {len(samples)} audio samples (augmentation={apply_augmentation})")

        processed_samples = []

        def process_single_audio(sample: AudioSample) -> AudioSample:
            """Process a single audio sample"""
            try:
                if not sample.audio_path or not Path(sample.audio_path).exists():
                    logger.warning(f"Audio file not found: {sample.audio_path}")
                    return sample

                # Load audio
                audio, sr = self.audio_processor.load_audio(sample.audio_path)

                # Preprocess audio (includes normalization, silence removal, optional augmentation)
                processed_audio = self.audio_processor.preprocess_audio(
                    audio, sr, apply_augmentation=apply_augmentation
                )

                # Calculate updated duration
                new_duration = len(processed_audio) / sr

                # Save processed audio (overwrite or save to new location)
                if apply_augmentation:
                    # Save augmented version with suffix
                    audio_path = Path(sample.audio_path)
                    augmented_path = audio_path.parent / f"{audio_path.stem}_aug{audio_path.suffix}"
                    sf.write(str(augmented_path), processed_audio, sr)

                    # Create new sample with augmented audio
                    return AudioSample(
                        audio_path=str(augmented_path),
                        transcription=sample.transcription,
                        duration=new_duration,
                        speaker_id=sample.speaker_id,
                        dialect=sample.dialect,
                        gender=sample.gender,
                        age_range=sample.age_range,
                        prompt_id=f"{sample.prompt_id}_aug",
                        prompt_text=sample.prompt_text,
                        meaning=sample.meaning,
                        session_id=sample.session_id,
                        quality_score=sample.quality_score,
                        original_filename=sample.original_filename,
                    )
                else:
                    # Overwrite original with processed version
                    sf.write(sample.audio_path, processed_audio, sr)

                    # Update duration in existing sample
                    sample.duration = new_duration
                    return sample

            except Exception as e:
                logger.error(f"Error processing audio {sample.audio_path}: {e}")
                return sample

        # Process in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_sample = {
                executor.submit(process_single_audio, sample): sample
                for sample in samples
            }

            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_sample), total=len(samples), desc="Processing audio"):
                processed_sample = future.result()
                processed_samples.append(processed_sample)

        logger.info(f"Audio processing completed: {len(processed_samples)} samples processed")
        return processed_samples

    def extract_features_batch(self, samples: List[AudioSample], feature_type: str = "wav2vec2", max_workers: int = 4) -> List[Dict[str, Any]]:
        """Extract features from audio samples in batch"""
        logger.info(f"Extracting {feature_type} features from {len(samples)} samples")

        features_list = []

        def extract_single_features(sample: AudioSample) -> Dict[str, Any]:
            """Extract features from a single audio sample"""
            try:
                if not sample.audio_path or not Path(sample.audio_path).exists():
                    logger.warning(f"Audio file not found: {sample.audio_path}")
                    return None

                # Load audio
                audio, sr = self.audio_processor.load_audio(sample.audio_path)

                if feature_type == "mfcc":
                    # Extract MFCC features
                    mfcc = librosa.feature.mfcc(
                        y=audio,
                        sr=sr,
                        n_mfcc=13,
                        n_fft=2048,
                        hop_length=512
                    )
                    features = mfcc.T  # Transpose to (time, features)

                elif feature_type == "mel_spectrogram":
                    # Extract Mel spectrogram
                    mel_spec = librosa.feature.melspectrogram(
                        y=audio,
                        sr=sr,
                        n_mels=80,
                        n_fft=2048,
                        hop_length=512
                    )
                    features = librosa.power_to_db(mel_spec).T

                elif feature_type == "wav2vec2":
                    # For wav2vec2, we'll store raw audio and let the model handle feature extraction
                    # Resample to 16kHz if needed
                    if sr != 16000:
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                        sr = 16000
                    features = audio

                else:
                    raise ValueError(f"Unsupported feature type: {feature_type}")

                return {
                    'features': features,
                    'sample_rate': sr,
                    'sample_id': sample.prompt_id,
                    'speaker_id': sample.speaker_id,
                    'transcription': sample.transcription,
                    'meaning': sample.meaning,
                    'feature_type': feature_type,
                    'audio_path': sample.audio_path,
                    'duration': sample.duration,
                    'metadata': {
                        'dialect': sample.dialect,
                        'gender': sample.gender,
                        'age_range': sample.age_range,
                        'quality_score': sample.quality_score
                    }
                }

            except Exception as e:
                logger.error(f"Error extracting features from {sample.audio_path}: {e}")
                return None

        # Extract features in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_sample = {
                executor.submit(extract_single_features, sample): sample
                for sample in samples
            }

            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_sample), total=len(samples), desc="Extracting features"):
                features_data = future.result()
                if features_data is not None:
                    features_list.append(features_data)

        logger.info(f"Feature extraction completed: {len(features_list)} feature sets extracted")
        return features_list

    def augment_dataset(self, samples: List[AudioSample], augmentation_factor: int = 2, max_workers: int = 4) -> List[AudioSample]:
        """Create augmented versions of the dataset"""
        logger.info(f"Creating {augmentation_factor}x augmented dataset from {len(samples)} samples")

        augmented_samples = []

        # Keep original samples
        augmented_samples.extend(samples)

        # Create augmented versions
        for aug_round in range(augmentation_factor - 1):
            logger.info(f"Creating augmentation round {aug_round + 1}/{augmentation_factor - 1}")

            # Process samples with augmentation enabled
            round_augmented = self.process_audio_batch(
                samples,
                apply_augmentation=True,
                max_workers=max_workers
            )

            # Update prompt IDs to include round number
            for sample in round_augmented:
                if sample.prompt_id.endswith('_aug'):
                    sample.prompt_id = sample.prompt_id.replace('_aug', f'_aug_{aug_round + 1}')

            augmented_samples.extend(round_augmented)

        logger.info(f"Dataset augmentation completed: {len(augmented_samples)} total samples")
        return augmented_samples

    def prepare_training_data(self, samples: List[AudioSample],
                            feature_type: str = "wav2vec2",
                            training_type: str = "speech_to_text",
                            apply_augmentation: bool = True,
                            augmentation_factor: int = 2,
                            max_workers: int = 4,
                            **dataset_kwargs) -> Dict[str, Any]:
        """Complete pipeline to prepare data for training with configurable training type"""
        logger.info(f"Starting training data preparation pipeline for {training_type}")

        # Step 1: Initial audio processing (normalize, trim silence, etc.)
        logger.info("Step 1: Processing audio files...")
        processed_samples = self.process_audio_batch(samples, apply_augmentation=False, max_workers=max_workers)

        # Step 2: Data augmentation (if enabled)
        if apply_augmentation and augmentation_factor > 1:
            logger.info("Step 2: Augmenting dataset...")
            augmented_samples = self.augment_dataset(processed_samples, augmentation_factor, max_workers)
        else:
            logger.info("Step 2: Skipping augmentation")
            augmented_samples = processed_samples

        # Step 3: Prepare training datasets based on type
        logger.info(f"Step 3: Creating {training_type} datasets...")
        train_samples, val_samples, test_samples = self.split_data(augmented_samples)

        # Step 4: Create training-specific data structure
        training_data = {
            'train_samples': train_samples,
            'validation_samples': val_samples,
            'test_samples': test_samples,
            'training_type': training_type,
            'feature_type': feature_type,
            'total_samples': len(augmented_samples),
            'augmentation_applied': apply_augmentation,
            'augmentation_factor': augmentation_factor if apply_augmentation else 1,
            'dataset_kwargs': dataset_kwargs,
            'metadata': {
                'speakers': list(set(s.speaker_id for s in augmented_samples)),
                'dialects': list(set(s.dialect for s in augmented_samples)),
                'total_duration': sum(s.duration for s in augmented_samples),
                'preparation_timestamp': datetime.now().isoformat(),
                'training_scenarios': self._get_training_scenario_info(training_type)
            }
        }

        logger.info("Training data preparation completed")
        logger.info(f"  - Training type: {training_type}")
        logger.info(f"  - Total samples: {training_data['total_samples']}")
        logger.info(f"  - Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
        logger.info(f"  - Feature type: {feature_type}")
        logger.info(f"  - Augmentation: {augmentation_factor}x" if apply_augmentation else "  - No augmentation")

        return training_data

    def _split_features_data(self, features_data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split features data into train/validation/test sets"""
        # Group by speaker to ensure speaker-independent splits
        speakers_data = {}
        for features in features_data:
            speaker_id = features['speaker_id']
            if speaker_id not in speakers_data:
                speakers_data[speaker_id] = []
            speakers_data[speaker_id].append(features)

        # Split speakers
        speakers = list(speakers_data.keys())
        train_speakers, temp_speakers = train_test_split(speakers, test_size=0.2, random_state=42)
        val_speakers, test_speakers = train_test_split(temp_speakers, test_size=0.5, random_state=42)

        # Collect features by split
        train_features = []
        val_features = []
        test_features = []

        for speaker in train_speakers:
            train_features.extend(speakers_data[speaker])
        for speaker in val_speakers:
            val_features.extend(speakers_data[speaker])
        for speaker in test_speakers:
            test_features.extend(speakers_data[speaker])

        return train_features, val_features, test_features

    def save_training_data(self, training_data: Dict[str, Any], output_dir: str = "data/processed"):
        """Save prepared training data to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save each split
        for split_name in ['train', 'validation', 'test']:
            split_data = training_data[split_name]
            split_path = output_path / f"{split_name}_{timestamp}.pkl"

            with open(split_path, 'wb') as f:
                pickle.dump(split_data, f)

            logger.info(f"Saved {split_name} data: {split_path} ({len(split_data)} samples)")

        # Save metadata
        metadata_path = output_path / f"metadata_{timestamp}.json"
        metadata = {
            'feature_type': training_data['feature_type'],
            'total_samples': training_data['total_samples'],
            'augmentation_applied': training_data['augmentation_applied'],
            'augmentation_factor': training_data['augmentation_factor'],
            'splits': {
                'train': len(training_data['train']),
                'validation': len(training_data['validation']),
                'test': len(training_data['test'])
            },
            'speakers': training_data['metadata']['speakers'],
            'dialects': training_data['metadata']['dialects'],
            'total_duration': training_data['metadata']['total_duration'],
            'preparation_timestamp': training_data['metadata']['preparation_timestamp']
        }

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved metadata: {metadata_path}")
        return output_path

    def _get_training_scenario_info(self, training_type: str) -> Dict[str, str]:
        """Get information about the training scenario"""
        scenarios = {
            'speech_to_text': {
                'input': 'Twi Audio',
                'output': 'Twi Text (prompt_text)',
                'description': 'Standard ASR - transcribe Twi speech to Twi text',
                'use_case': 'Speech recognition for Twi language'
            },
            'translation': {
                'input': 'Twi Audio',
                'output': 'English Text (meaning)',
                'description': 'Speech translation - translate Twi speech to English text',
                'use_case': 'Direct translation from Twi audio to English'
            },
            'multilingual': {
                'input': 'Twi Audio',
                'output': 'Either Twi or English Text (task-dependent)',
                'description': 'Multilingual model - one model for both transcription and translation',
                'use_case': 'Unified model that can do both ASR and translation'
            },
            'cross_lingual': {
                'input': 'Twi Audio + Twi Text',
                'output': 'English Text',
                'description': 'Multimodal translation - use both audio and text for better translation',
                'use_case': 'Enhanced translation with dual input modalities'
            }
        }
        return scenarios.get(training_type, {'description': 'Unknown training type'})

    def create_pytorch_datasets(self, training_data: Dict[str, Any],
                               processor,
                               is_training: bool = True) -> Dict[str, Dataset]:
        """Create PyTorch datasets from prepared training data"""

        # Import training datasets here to avoid circular import
        try:
            from .training_datasets import create_dataset
        except ImportError as e:
            raise ImportError(f"Training dataset classes not available: {e}")

        training_type = training_data.get('training_type', 'speech_to_text')
        dataset_kwargs = training_data.get('dataset_kwargs', {})

        logger.info(f"Creating PyTorch datasets for {training_type}")

        datasets = {}

        for split_name in ['train', 'validation', 'test']:
            samples_key = f'{split_name}_samples'
            if samples_key in training_data:
                samples = training_data[samples_key]

                dataset = create_dataset(
                    samples=samples,
                    dataset_type=training_type,
                    processor=processor,
                    audio_processor=self.audio_processor,
                    text_processor=self.text_processor,
                    is_training=(split_name == 'train' and is_training),
                    **dataset_kwargs
                )

                datasets[split_name] = dataset
                logger.info(f"Created {split_name} dataset: {len(dataset)} samples")

        return datasets

    def _create_data_index(self, samples: List[AudioSample], recordings_dir: str):
        """Create a comprehensive index of all organized data"""
        try:
            index_data = {
                'created_at': datetime.now().isoformat(),
                'total_samples': len(samples),
                'recordings_directory': recordings_dir,
                'speakers': {},
                'prompts': {},
                'dialects': {},
                'summary': {
                    'total_duration': sum(s.duration for s in samples),
                    'unique_speakers': len(set(s.speaker_id for s in samples)),
                    'unique_prompts': len(set(s.prompt_id for s in samples)),
                    'samples_with_translations': sum(1 for s in samples if s.meaning),
                }
            }

            # Group by speakers
            for sample in samples:
                speaker_id = sample.speaker_id
                if speaker_id not in index_data['speakers']:
                    index_data['speakers'][speaker_id] = {
                        'sample_count': 0,
                        'total_duration': 0,
                        'dialect': sample.dialect,
                        'gender': sample.gender,
                        'age_range': sample.age_range,
                        'prompts': []
                    }

                speaker_data = index_data['speakers'][speaker_id]
                speaker_data['sample_count'] += 1
                speaker_data['total_duration'] += sample.duration
                speaker_data['prompts'].append(sample.prompt_id)

            # Group by prompts
            for sample in samples:
                prompt_id = sample.prompt_id
                if prompt_id not in index_data['prompts']:
                    index_data['prompts'][prompt_id] = {
                        'prompt_text': sample.prompt_text,
                        'meaning': sample.meaning,
                        'sample_count': 0,
                        'speakers': []
                    }

                prompt_data = index_data['prompts'][prompt_id]
                prompt_data['sample_count'] += 1
                if sample.speaker_id not in prompt_data['speakers']:
                    prompt_data['speakers'].append(sample.speaker_id)

            # Group by dialects
            for sample in samples:
                dialect = sample.dialect
                if dialect not in index_data['dialects']:
                    index_data['dialects'][dialect] = {
                        'sample_count': 0,
                        'speakers': set(),
                        'total_duration': 0
                    }

                dialect_data = index_data['dialects'][dialect]
                dialect_data['sample_count'] += 1
                dialect_data['speakers'].add(sample.speaker_id)
                dialect_data['total_duration'] += sample.duration

            # Convert sets to lists for JSON serialization
            for dialect_data in index_data['dialects'].values():
                dialect_data['speakers'] = list(dialect_data['speakers'])
                dialect_data['unique_speakers'] = len(dialect_data['speakers'])

            # Save index file
            index_path = Path(recordings_dir) / 'data_index.json'
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Created data index: {index_path}")
            logger.info(f"  - {index_data['summary']['unique_speakers']} speakers")
            logger.info(f"  - {index_data['summary']['unique_prompts']} unique prompts")
            logger.info(f"  - {index_data['summary']['total_duration']:.1f} seconds total duration")

        except Exception as e:
            logger.error(f"Failed to create data index: {e}")

    def fetch_data_from_backend_sync(self) -> List[AudioSample]:
        """Synchronous wrapper for fetch_data_from_backend with proper event loop handling"""
        logger.info("🔄 Starting data fetch from backend...")

        # Use run_async_fresh to ensure a clean event loop
        samples = run_async_fresh(self.fetch_data_from_backend())

        logger.info(f"📥 Downloaded {len(samples)} samples to cache directory")

        # Organize downloaded data into recordings directory
        if samples:
            logger.info("🗂️ Organizing downloaded data into recordings directory...")
            samples = self.organize_downloaded_data(samples)
            logger.info(f"✅ Data organization complete! {len(samples)} samples now in data/recordings/")
        else:
            logger.warning("⚠️ No samples downloaded from backend")

        return samples

    def _create_dummy_samples(self) -> List[AudioSample]:
        """Create dummy samples for development when database is not available"""
        dummy_samples = []

        # Create some sample audio files if they don't exist
        dummy_audio_dir = self.cache_dir / "dummy_audio"
        dummy_audio_dir.mkdir(exist_ok=True)

        # E-commerce Twi sample sentences for testing (from actual dataset)
        twi_sentences = [
            ("Bue adwadie app no", "Open the shopping app"),
            ("Kɔ fie page no so", "Go to homepage"),
            ("Fa to cart no mu", "Go to cart"),
            ("Hwehwɛ ntadeɛ beaeɛ", "Search for clothing"),
            ("Hwehwɛ mfidie beaeɛ", "Search for electronics"),
            ("Fa yei to me cart no mu", "Add this to my cart"),
            ("Tua ka seesei ara", "Pay now"),
            ("Tua fa mobile money no so", "Select mobile money payment"),
            ("Kyerɛ me kwan a mɛfa so atɔ adeɛ", "Guide me through purchasing a product"),
            ("Baako", "One"),
            ("Mmienu", "Two"),
            ("Mmiɛnsa", "Three"),
            ("Adeɛ yi boɔ yɛ sɛn?", "How much is this product?"),
            ("Kyerɛ me nnoɔma a ɛmmoro sidi ɔha", "Show me items under 100 cedis"),
            ("Mepɛ express delivery", "I want express delivery"),
            ("Fa me kɔ ntadeɛ beaeɛ ho", "Take me to the clothing section"),
            ("Wowɔ yei wɔ kɔla afoforɔ mu?", "Do you have this in other colors?"),
            ("Fa yei toto yei ho", "Compare these two items"),
            ("Mepɛ sɛ mesan de adeɛ yi ma mo", "I want to return this product"),
            ("Me werɛ afi me akaont ho kodeɛ", "I've forgotten my account password"),
            ("Mɛtumi de mobile money atua ka?", "Can I pay using mobile money?"),
            ("Me nnoɔma no wɔ hen seesei ara?", "Where is my product now?"),
            ("Kyerɛ me nnoɔma a wɔtaa tɔ no abom", "Show me items commonly bought together"),
            ("Hwehwɛ telefon a ɛwɔ camera papa", "Search for phones with good cameras"),
            ("Kenkan awerɛkyekyerɛ no ma me", "Read the reviews for me"),
            ("Mema wo akwaaba", "I welcome you"),
            ("Ɛhɛ na wo fi?", "Where are you from?"),
            ("Me fi Kumasi", "I'm from Kumasi"),
            ("Asante pii", "Thank you very much"),
            ("Mepɛ sɛ mehwɛ nnoɔma", "I want to look at products"),
        ]

        dialects = ['Asante', 'Akuapem', 'Fante']
        genders = ['male', 'female']
        age_ranges = ['18-25', '26-35', '36-45', '46-55']

        for i, (twi_text, english_meaning) in enumerate(twi_sentences):
            # Create dummy audio file path (we'll create silent audio files)
            audio_filename = f"dummy_audio_{i:03d}.wav"
            audio_path = dummy_audio_dir / audio_filename

            # Create a simple silent audio file if it doesn't exist
            if not audio_path.exists():
                self._create_dummy_audio_file(str(audio_path))

            sample = AudioSample(
                audio_path=str(audio_path),
                transcription=twi_text,
                duration=2.5 + (i * 0.1),  # Varying durations
                speaker_id=f"speaker_{i % 8:03d}",  # 8 different speakers for better distribution
                dialect=dialects[i % len(dialects)],
                gender=genders[i % len(genders)],
                age_range=age_ranges[i % len(age_ranges)],
                prompt_id=f"ecommerce_{i:03d}",
                session_id=f"session_{i // 5:03d}",  # Group by sessions (5 per session)
            )
            dummy_samples.append(sample)

        logger.info(f"Created {len(dummy_samples)} e-commerce dummy samples for development")
        return dummy_samples

    def _create_dummy_audio_file(self, audio_path: str):
        """Create a dummy silent audio file for testing"""
        try:
            import numpy as np
            import soundfile as sf

            # Create 2.5 seconds of silent audio at 16kHz
            sample_rate = 16000
            duration = 2.5
            samples = int(sample_rate * duration)

            # Create silent audio with very small amount of noise to avoid issues
            audio_data = np.random.normal(0, 0.001, samples).astype(np.float32)

            # Save as WAV file
            sf.write(audio_path, audio_data, sample_rate)
            logger.debug(f"Created dummy audio file: {audio_path}")

        except Exception as e:
            logger.error(f"Failed to create dummy audio file {audio_path}: {e}")
            # Create an empty file as last resort
            Path(audio_path).touch()

    async def _download_audio_file(self, recording: Dict[str, Any]) -> Optional[str]:
        """Download audio file from R2 storage and save metadata"""
        object_key = recording.get('object_key')
        if not object_key:
            logger.warning("Recording has no object_key, cannot download.")
            return None

        try:
            # Create local file path preserving original filename
            # Extract filename from object_key (last part after /)
            original_filename = object_key.split('/')[-1]
            # Use a subdirectory based on hash to avoid conflicts
            file_hash = hashlib.md5(object_key.encode()).hexdigest()[:8]
            local_dir = self.cache_dir / file_hash
            local_dir.mkdir(exist_ok=True)
            local_path = local_dir / original_filename

            # Create metadata file path
            metadata_filename = original_filename.rsplit('.', 1)[0] + '_metadata.json'
            metadata_path = local_dir / metadata_filename

            # Check if file already exists
            if local_path.exists() and local_path.stat().st_size > 0:
                logger.debug(f"Found cached file: {local_path}")
                # Check if metadata exists, if not create it
                if not metadata_path.exists():
                    await self._save_metadata_to_cache(recording, metadata_path)
                return str(local_path)

            if not self.r2_client or not self.bucket_name:
                logger.warning("R2 client not configured. Skipping download.")
                return None

            logger.info(f"Downloading {object_key} from R2 bucket {self.bucket_name}...")

            # Boto3's get_object is a synchronous call.
            # We run it in an executor to avoid blocking the asyncio event loop.
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,  # Use default executor
                lambda: self.r2_client.get_object(Bucket=self.bucket_name, Key=object_key)
            )

            # Read the content from the streaming body
            audio_data = response['Body'].read()

            # Save to local file asynchronously
            async with aiofiles.open(local_path, 'wb') as f:
                await f.write(audio_data)

            # Save metadata alongside the audio file
            await self._save_metadata_to_cache(recording, metadata_path)

            logger.info(f"Successfully downloaded and cached to {local_path}")
            logger.info(f"Successfully saved metadata to {metadata_path}")
            return str(local_path)

        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.error(f"R2 Error: Object key '{object_key}' not found in bucket '{self.bucket_name}'.")
            else:
                logger.error(f"R2 ClientError downloading {object_key}: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred downloading {object_key}: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def _save_metadata_to_cache(self, recording: Dict[str, Any], metadata_path: Path) -> None:
        """Save recording metadata to cache directory"""
        try:
            # Prepare metadata from recording
            metadata = {
                'prompt_id': recording.get('prompt_id', ''),
                'prompt_text': recording.get('prompt_text', ''),
                'transcription': recording.get('transcription'),
                'speaker_id': str(recording.get('speaker_id', '')),
                'session_id': recording.get('session_id'),
                'object_key': recording.get('object_key', ''),
                'filename_original': recording.get('filename_original', ''),
                'content_type': recording.get('content_type', ''),
                'size_bytes': recording.get('size_bytes'),
                'recording_duration': recording.get('recording_duration'),
                'uploaded_at': recording.get('uploaded_at').isoformat() if recording.get('uploaded_at') else None,
                'transcription_status': recording.get('transcription_status', ''),
                'transcribed_by': recording.get('transcribed_by'),
                'transcription_updated_at': recording.get('transcription_updated_at').isoformat() if recording.get('transcription_updated_at') else None,
                'cached_at': datetime.now().isoformat(),
                'source': 'mongodb_backend'
            }

            # Save metadata to JSON file
            async with aiofiles.open(metadata_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(metadata, indent=2, ensure_ascii=False, default=str))

            logger.debug(f"Metadata saved to cache: {metadata_path}")

        except Exception as e:
            logger.error(f"Failed to save metadata to cache {metadata_path}: {e}")
            # Don't raise - metadata save failure shouldn't stop audio download

    def _load_cached_metadata(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """Load metadata from cache directory if it exists"""
        try:
            audio_file = Path(audio_path)
            # Construct metadata file path
            metadata_filename = audio_file.stem + '_metadata.json'
            metadata_path = audio_file.parent / metadata_filename

            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                logger.debug(f"Loaded cached metadata from: {metadata_path}")
                return metadata
            else:
                logger.debug(f"No cached metadata found at: {metadata_path}")
                return None

        except Exception as e:
            logger.error(f"Failed to load cached metadata for {audio_path}: {e}")
            return None

    def _validate_sample(self, sample: AudioSample) -> bool:
        """Validate a data sample"""
        try:
            # Check if audio file exists
            if not os.path.exists(sample.audio_path):
                return False

            # Validate text
            if not self.text_processor.is_valid_text(sample.transcription):
                return False

            # Validate audio
            audio, sr = self.audio_processor.load_audio(sample.audio_path)
            is_valid, quality_score = self.audio_processor.validate_audio_quality(audio, sr)

            if not is_valid:
                return False

            # Update quality score
            sample.quality_score = quality_score

            return True

        except Exception as e:
            logger.error(f"Failed to validate sample {sample.audio_path}: {e}")
            return False

    def _check_audio_device(self):
        """Check if audio recording device is available"""
        try:
            if sd is None:
                raise RuntimeError("sounddevice module not available")

            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            if not input_devices:
                raise RuntimeError("No audio input devices found")

            # Set default input device
            default_input = sd.query_devices(kind='input')
            logger.info(f"Using audio device: {default_input['name']}")

        except Exception as e:
            logger.error(f"Audio device error: {e}. Recording will not be possible.")

    def start_session(self, speaker_id: str) -> str:
        """Start a new recording session"""
        session_id = f"{speaker_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.current_session = RecordingSession(
            session_id=session_id,
            speaker_id=speaker_id,
            date=datetime.now().isoformat(),
            prompts_recorded=[],
            total_duration=0.0,
            quality_score=0.0
        )

        # Create speaker directory
        recordings_dir = self.config.get('recordings_dir', 'data/recordings')
        os.makedirs(recordings_dir, exist_ok=True)
        speaker_dir = os.path.join(recordings_dir, speaker_id)
        os.makedirs(speaker_dir, exist_ok=True)

        logger.info(f"Started recording session: {session_id}")
        return session_id

    def record_prompt(self, prompt: TwiPrompt, speaker_id: str, max_duration: float = None) -> Optional[AudioSample]:
        """Record audio for a single prompt"""
        if not self.current_session:
            raise RuntimeError("No active recording session. Call start_session() first.")

        max_duration = max_duration or self.config.get('audio', {}).get('max_duration', 15.0)
        recordings_dir = self.config.get('recordings_dir', 'data/recordings')

        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        filename = f"{speaker_id}_{prompt.id}_{timestamp}.wav"
        file_path = os.path.join(recordings_dir, speaker_id, filename)

        print(f"\n🎤 Recording: {prompt.text}")
        print(f"   Meaning: {prompt.meaning}")
        print(f"▶️ Press ENTER to start recording, then ENTER again to stop...")

        input()

        # Start recording
        print("🔴 RECORDING... Press ENTER to stop")

        recorded_audio = []
        self.is_recording = True

        def audio_callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio callback status: {status}")
            if self.is_recording:
                recorded_audio.append(indata.copy())

        try:
            if sd is None:
                raise RuntimeError("sounddevice module not available for recording")

            with sd.InputStream(
                callback=audio_callback,
                channels=self.channels,
                samplerate=self.sample_rate,
                dtype=self.dtype
            ):
                input()
        finally:
            self.is_recording = False

        if not recorded_audio:
            logger.warning("No audio recorded")
            return None

        # Combine audio chunks
        audio_data = np.concatenate(recorded_audio, axis=0)
        duration = len(audio_data) / self.sample_rate

        # Validate recording duration
        min_duration = self.config.get('audio', {}).get('min_duration', 0.5)
        if duration < min_duration:
            logger.warning(f"Recording too short: {duration:.2f}s (min: {min_duration}s)")
            return None

        if duration > max_duration:
            logger.warning(f"Recording too long: {duration:.2f}s (max: {max_duration}s). Trimming...")
            max_samples = int(max_duration * self.sample_rate)
            audio_data = audio_data[:max_samples]
            duration = max_duration

        # Save audio file
        self._save_audio(audio_data, file_path)

        # Create recording object
        recording = AudioRecording(
            filename=filename,
            prompt_id=prompt.id,
            prompt_text=prompt.text,
            prompt_meaning=prompt.meaning,
            speaker_id=speaker_id,
            duration=duration,
            sample_rate=self.sample_rate,
            recording_date=datetime.now().isoformat(),
            file_path=file_path,
            quality_score=self._assess_audio_quality(audio_data)
        )

        self.recordings.append(recording)
        if self.current_session:
            self.current_session.prompts_recorded.append(prompt.id)
            self.current_session.total_duration += duration

        logger.info(f"Recording saved: {filename} ({duration:.2f}s, Quality: {recording.quality_score:.2f}/10)")
        return recording

    def _save_audio(self, audio_data: np.ndarray, file_path: str):
        """Save audio data to WAV file"""
        audio_int16 = (audio_data * 32767).astype(np.int16)
        with wave.open(file_path, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

    def _assess_audio_quality(self, audio_data: np.ndarray) -> float:
        """Assess audio quality"""
        rms = np.sqrt(np.mean(audio_data**2))
        clipping_ratio = np.sum(np.abs(audio_data) > 0.95) / len(audio_data)
        silence_threshold = 0.01
        non_silent_ratio = np.sum(np.abs(audio_data) > silence_threshold) / len(audio_data)

        quality_score = 10.0
        if clipping_ratio > 0.01: quality_score -= 3.0
        if non_silent_ratio < 0.3: quality_score -= 2.0
        if rms < 0.01: quality_score -= 2.0
        if rms > 0.5: quality_score -= 1.0
        return max(0.0, quality_score)

    def record_section(self, section_id: str, speaker_id: str) -> List[AudioRecording]:
        """Record all prompts in a section"""
        if not self.script_parser:
            logger.error("Script parser not initialized.")
            return []

        section_prompts = self.script_parser.get_prompts_by_section(section_id)
        if not section_prompts:
            logger.error(f"No prompts found for section: {section_id}")
            return []

        section = next((s for s in self.sections if s.id == section_id), None)
        if section:
            print(f"\n📋 Recording Section: {section.title} ({len(section_prompts)} prompts)")

        recordings = []
        for i, prompt in enumerate(section_prompts, 1):
            print(f"\n📍 Prompt {i}/{len(section_prompts)}")
            recording = self.record_prompt(prompt, speaker_id)
            if recording:
                recordings.append(recording)

            if i < len(section_prompts):
                choice = input("Continue to next prompt? (y/n/skip): ").lower()
                if choice == 'n': break
                elif choice == 'skip': continue
        return recordings

    def load_local_data(self) -> List[AudioSample]:
        """Load data from local recordings directory"""
        logger.info("Attempting to load data from local recordings directory...")
        local_samples = []
        recordings_dir = self.config.get('recordings_dir', 'data/recordings')

        if not os.path.exists(recordings_dir):
            logger.warning(f"Local recordings directory not found: {recordings_dir}")
            return []

        for speaker_id in os.listdir(recordings_dir):
            speaker_dir = os.path.join(recordings_dir, speaker_id)
            if not os.path.isdir(speaker_dir):
                continue

            logger.debug(f"Scanning speaker directory: {speaker_dir}")

            # Method 1: Load organized metadata files (from cloud downloads)
            metadata_files = [f for f in os.listdir(speaker_dir) if f.endswith('_metadata.json')]
            for metadata_file in metadata_files:
                json_path = os.path.join(speaker_dir, metadata_file)
                try:
                    logger.debug(f"Loading organized metadata: {json_path}")
                    with open(json_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)

                    # Get corresponding audio file
                    audio_filename = metadata_file.replace('_metadata.json', '')
                    # Try different extensions
                    for ext in ['.wav', '.m4a', '.mp3']:
                        audio_path = os.path.join(speaker_dir, audio_filename + ext)
                        if os.path.exists(audio_path):
                            break
                    else:
                        logger.warning(f"Audio file not found for metadata: {metadata_file}")
                        continue

                    sample = AudioSample(
                        audio_path=audio_path,
                        transcription=metadata.get('transcriptions'),
                        duration=metadata.get('duration', 0.0),
                        speaker_id=metadata.get('speaker_id', speaker_id),
                        dialect=metadata.get('dialect', 'unknown'),
                        gender=metadata.get('gender'),
                        age_range=metadata.get('age_range'),
                        prompt_id=metadata.get('prompt_id', ''),
                        prompt_text=metadata.get('prompt_text'),
                        meaning=metadata.get('transcriptions'),
                        session_id=metadata.get('session_id'),
                        quality_score=metadata.get('quality_score', 1.0),
                        original_filename=metadata.get('original_filename')
                    )

                    if self._validate_sample(sample):
                        local_samples.append(sample)
                        logger.debug(f"Loaded organized sample: {sample.prompt_id}")
                    else:
                        logger.warning(f"Sample validation failed: {sample.prompt_id}")

                except Exception as e:
                    logger.error(f"Error loading organized metadata {json_path}: {e}")

            # Method 2: Load legacy recording session files (from local recordings)
            legacy_files = [f for f in os.listdir(speaker_dir) if f.startswith('recordings_') and f.endswith('.json')]
            for filename in legacy_files:
                json_path = os.path.join(speaker_dir, filename)
                try:
                    logger.debug(f"Loading legacy recording session: {json_path}")
                    with open(json_path, 'r') as f:
                        recording_data_list = json.load(f)

                    for rec_data in recording_data_list:
                        audio_path = rec_data.get('file_path')
                        if not audio_path or not os.path.exists(audio_path):
                            continue

                        sample = AudioSample(
                            audio_path=audio_path,
                            transcription=rec_data.get('prompt_text', ''),
                            duration=rec_data.get('duration', 0.0),
                            speaker_id=rec_data.get('speaker_id', speaker_id),
                            dialect='unknown',
                            gender=None,
                            age_range=None,
                            prompt_id=rec_data.get('prompt_id', ''),
                            session_id=rec_data.get('session_id', ''),
                            quality_score=rec_data.get('quality_score')
                        )
                        if self._validate_sample(sample):
                            local_samples.append(sample)
                            logger.debug(f"Loaded legacy sample: {sample.prompt_id}")

                except Exception as e:
                    logger.error(f"Error loading legacy recording metadata {json_path}: {e}")

            # Method 3: Load standalone audio files without metadata (fallback)
            audio_files = [f for f in os.listdir(speaker_dir)
                          if f.endswith(('.wav', '.m4a', '.mp3')) and
                          not any(f.replace(ext, '_metadata.json') for ext in ['.wav', '.m4a', '.mp3'])]

            for audio_file in audio_files:
                audio_path = os.path.join(speaker_dir, audio_file)
                try:
                    # Create basic sample for standalone audio files
                    sample = AudioSample(
                        audio_path=audio_path,
                        transcription="Unknown transcription",
                        duration=0.0,  # Will be calculated if needed
                        speaker_id=speaker_id,
                        dialect='unknown',
                        prompt_id=audio_file.replace('.wav', '').replace('.m4a', '').replace('.mp3', ''),
                        prompt_text="Unknown prompt",
                        meaning="Unknown meaning",
                        original_filename=audio_file
                    )

                    if self._validate_sample(sample):
                        local_samples.append(sample)
                        logger.debug(f"Loaded standalone audio: {audio_file}")

                except Exception as e:
                    logger.error(f"Error loading standalone audio {audio_file}: {e}")

        logger.info(f"Loaded {len(local_samples)} valid samples from local recordings.")
        logger.info(f"  - Organized samples: {len([s for s in local_samples if s.meaning and s.meaning != 'Unknown meaning'])}")
        logger.info(f"  - Legacy samples: {len([s for s in local_samples if s.session_id])}")
        logger.info(f"  - Standalone files: {len([s for s in local_samples if s.meaning == 'Unknown meaning'])}")

        return local_samples


    def finish_session(self):
        """Finish the current recording session"""
        if not self.current_session:
            logger.warning("No active recording session")
            return

        if self.recordings:
            avg_quality = sum(r.quality_score or 0 for r in self.recordings) / len(self.recordings)
            self.current_session.quality_score = avg_quality

        recordings_dir = self.config.get('recordings_dir', 'data/recordings')
        speaker_dir = os.path.join(recordings_dir, self.current_session.speaker_id)

        session_file = os.path.join(speaker_dir, f"session_{self.current_session.session_id}.json")
        with open(session_file, 'w') as f:
            json.dump(asdict(self.current_session), f, indent=2)

        recordings_file = os.path.join(speaker_dir, f"recordings_{self.current_session.session_id}.json")
        with open(recordings_file, 'w') as f:
            json.dump([asdict(r) for r in self.recordings], f, indent=2, ensure_ascii=False)

        logger.info("Session Summary:")
        logger.info(f"   - Speaker: {self.current_session.speaker_id}")
        logger.info(f"   - Prompts Recorded: {len(self.current_session.prompts_recorded)}")
        logger.info(f"   - Total Duration: {self.current_session.total_duration:.2f}s")
        logger.info(f"   - Average Quality: {self.current_session.quality_score:.2f}/10")
        logger.info(f"   - Session Data: {session_file}")

        self.current_session = None
        self.recordings = []

    def split_data(self, samples: List[AudioSample]) -> Tuple[List[AudioSample], List[AudioSample], List[AudioSample]]:
        """Split data into train/val/test sets"""
        split_config = self.config.get('data_split', {})
        method = split_config.get('method', 'stratified')

        if method == 'stratified':
            return self._stratified_split(samples, split_config)
        elif method == 'speaker_independent':
            return self._speaker_independent_split(samples, split_config)
        else:
            return self._random_split(samples, split_config)

    def _stratified_split(self, samples: List[AudioSample], config: Dict[str, Any]) -> Tuple[List[AudioSample], List[AudioSample], List[AudioSample]]:
        """Stratified split based on dialect and other factors"""
        # Check if we have enough samples for stratified split
        if len(samples) < 30:  # Too few samples for meaningful stratification
            logger.warning(f"Too few samples ({len(samples)}) for stratified split, falling back to random split")
            return self._random_split(samples, config)

        # Create stratification key
        stratify_keys = []
        for sample in samples:
            key = f"{sample.dialect}_{sample.gender}_{sample.age_range}"
            stratify_keys.append(key)

        # Check if we have at least 2 samples per class
        from collections import Counter
        key_counts = Counter(stratify_keys)
        min_count = min(key_counts.values())
        if min_count < 2:
            logger.warning(f"Some classes have only {min_count} sample(s), falling back to random split")
            return self._random_split(samples, config)

        # Split ratios
        train_ratio = config.get('train_ratio', 0.7)
        val_ratio = config.get('validation_ratio', 0.15)
        test_ratio = config.get('test_ratio', 0.15)

        try:
            # First split: train vs (val + test)
            train_samples, temp_samples, _, temp_keys = train_test_split(
                samples, stratify_keys,
                test_size=(val_ratio + test_ratio),
                stratify=stratify_keys,
                random_state=42
            )

            # Second split: val vs test
            val_samples, test_samples = train_test_split(
                temp_samples,
                test_size=test_ratio / (val_ratio + test_ratio),
                stratify=temp_keys,
                random_state=42
            )

            return train_samples, val_samples, test_samples
        except ValueError as e:
            logger.warning(f"Stratified split failed: {e}, falling back to random split")
            return self._random_split(samples, config)

    def _speaker_independent_split(self, samples: List[AudioSample], config: Dict[str, Any]) -> Tuple[List[AudioSample], List[AudioSample], List[AudioSample]]:
        """Speaker-independent split"""
        # Group samples by speaker
        speaker_samples = defaultdict(list)
        for sample in samples:
            speaker_samples[sample.speaker_id].append(sample)

        speakers = list(speaker_samples.keys())

        # Split speakers
        train_ratio = config.get('train_ratio', 0.7)
        val_ratio = config.get('validation_ratio', 0.15)

        train_speakers, temp_speakers = train_test_split(
            speakers, test_size=(1 - train_ratio), random_state=42
        )

        val_speakers, test_speakers = train_test_split(
            temp_speakers, test_size=0.5, random_state=42
        )

        # Collect samples
        train_samples = []
        val_samples = []
        test_samples = []

        for speaker_id in train_speakers:
            train_samples.extend(speaker_samples[speaker_id])

        for speaker_id in val_speakers:
            val_samples.extend(speaker_samples[speaker_id])

        for speaker_id in test_speakers:
            test_samples.extend(speaker_samples[speaker_id])

        return train_samples, val_samples, test_samples

    def _random_split(self, samples: List[AudioSample], config: Dict[str, Any]) -> Tuple[List[AudioSample], List[AudioSample], List[AudioSample]]:
        """Random split"""
        train_ratio = config.get('train_ratio', 0.7)
        val_ratio = config.get('validation_ratio', 0.15)

        train_samples, temp_samples = train_test_split(
            samples, test_size=(1 - train_ratio), random_state=42
        )

        val_samples, test_samples = train_test_split(
            temp_samples, test_size=0.5, random_state=42
        )

        return train_samples, val_samples, test_samples

    def create_datasets(self, train_samples: List[AudioSample], val_samples: List[AudioSample],
                       test_samples: List[AudioSample], processor: Wav2Vec2Processor) -> Tuple[TwiSpeechDataset, TwiSpeechDataset, TwiSpeechDataset]:
        """Create PyTorch datasets"""
        train_dataset = TwiSpeechDataset(
            train_samples, processor, self.audio_processor, self.text_processor, is_training=True
        )

        val_dataset = TwiSpeechDataset(
            val_samples, processor, self.audio_processor, self.text_processor, is_training=False
        )

        test_dataset = TwiSpeechDataset(
            test_samples, processor, self.audio_processor, self.text_processor, is_training=False
        )

        return train_dataset, val_dataset, test_dataset

    def create_dataloader(self, dataset: TwiSpeechDataset, batch_size: int,
                         shuffle: bool = False, is_training: bool = False) -> DataLoader:
        """Create PyTorch DataLoader"""
        # Create weighted sampler for balanced training
        sampler = None
        if is_training and self.config.get('balanced_sampling', False):
            sampler = self._create_balanced_sampler(dataset)
            shuffle = False  # Disable shuffle when using sampler

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.get('dataloader', {}).get('num_workers', 4),
            pin_memory=self.config.get('dataloader', {}).get('pin_memory', True),
            persistent_workers=self.config.get('dataloader', {}).get('persistent_workers', True),
            collate_fn=self._collate_fn
        )

    def _create_balanced_sampler(self, dataset: TwiSpeechDataset) -> WeightedRandomSampler:
        """Create balanced sampler for training"""
        # Count samples per dialect
        dialect_counts = defaultdict(int)
        for sample in dataset.samples:
            dialect_counts[sample.dialect] += 1

        # Calculate weights
        weights = []
        for sample in dataset.samples:
            weight = 1.0 / dialect_counts[sample.dialect]
            weights.append(weight)

        return WeightedRandomSampler(weights, len(weights))

    def _collate_fn(self, batch):
        """Custom collate function for batching"""
        # Handle variable length sequences
        input_values = [item['input_values'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]

        # Pad sequences
        max_length = max(len(seq) for seq in input_values)

        padded_input_values = []
        padded_attention_masks = []
        padded_labels = []

        for i in range(len(batch)):
            # Pad input values
            padded_input = torch.zeros(max_length)
            padded_input[:len(input_values[i])] = input_values[i]
            padded_input_values.append(padded_input)

            # Pad attention masks
            padded_mask = torch.zeros(max_length)
            padded_mask[:len(attention_masks[i])] = attention_masks[i]
            padded_attention_masks.append(padded_mask)

            # Pad labels
            padded_label = torch.full((max_length,), -100)  # -100 is ignored in loss
            padded_label[:len(labels[i])] = labels[i]
            padded_labels.append(padded_label)

        return {
            'input_values': torch.stack(padded_input_values),
            'attention_mask': torch.stack(padded_attention_masks),
            'labels': torch.stack(padded_labels),
            'speaker_ids': [item['speaker_id'] for item in batch],
            'dialects': [item['dialect'] for item in batch],
        }

    def compute_dataset_stats(self, samples: List[AudioSample]) -> DatasetStats:
        """Compute dataset statistics"""
        total_duration = sum(sample.duration for sample in samples)
        durations = [sample.duration for sample in samples]

        dialect_counts = defaultdict(int)
        gender_counts = defaultdict(int)
        speakers = set()

        all_text = []
        for sample in samples:
            dialect_counts[sample.dialect] += 1
            if sample.gender:
                gender_counts[sample.gender] += 1
            speakers.add(sample.speaker_id)
            all_text.append(sample.transcription)

        # Vocabulary analysis
        vocabulary = set()
        total_chars = 0
        for text in all_text:
            words = text.split()
            vocabulary.update(words)
            total_chars += len(text)

        return DatasetStats(
            total_samples=len(samples),
            total_duration=total_duration,
            avg_duration=total_duration / len(samples) if samples else 0,
            min_duration=min(durations) if durations else 0,
            max_duration=max(durations) if durations else 0,
            dialect_distribution=dict(dialect_counts),
            gender_distribution=dict(gender_counts),
            speaker_count=len(speakers),
            vocabulary_size=len(vocabulary),
            character_count=total_chars
        )

    def prepare_datasets(self) -> Tuple[List[AudioSample], List[AudioSample], List[AudioSample]]:
        """Main method to prepare all datasets"""
        import asyncio

        async def _prepare_async():
            logger.info("Preparing datasets...")
            samples = []

            # 1. Try to fetch from backend
            if not self.use_fallback_data:
                samples = await self.fetch_data_from_backend()

            # 2. If no cloud data, try local recordings
            if not samples:
                logger.info("No data from backend, checking for local recordings...")
                samples = self.load_local_data()

            # 3. If still no data, use dummy data
            if not samples:
                logger.warning("No cloud or local data found. Falling back to dummy data for development.")
                samples = self._create_dummy_samples()

            if not samples:
                raise ValueError("No valid samples found from any source.")

            # Compute and log statistics
            stats = self.compute_dataset_stats(samples)
            logger.info(f"Dataset statistics:")
            logger.info(f"  Total samples: {stats.total_samples}")
            logger.info(f"  Total duration: {stats.total_duration:.2f} hours")
            logger.info(f"  Average duration: {stats.avg_duration:.2f} seconds")
            logger.info(f"  Speakers: {stats.speaker_count}")
            logger.info(f"  Dialects: {stats.dialect_distribution}")
            logger.info(f"  Vocabulary size: {stats.vocabulary_size}")

            # Split data
            train_samples, val_samples, test_samples = self.split_data(samples)

            logger.info(f"Data split:")
            logger.info(f"  Training: {len(train_samples)} samples")
            logger.info(f"  Validation: {len(val_samples)} samples")
            logger.info(f"  Test: {len(test_samples)} samples")

            # Return raw samples - datasets will be created by trainer after processor is available
            return train_samples, val_samples, test_samples

        # Use run_async_fresh to ensure a clean event loop
        return run_async_fresh(_prepare_async())

    def prepare_complete_training_pipeline(self,
                                         feature_type: str = "wav2vec2",
                                         training_type: str = "speech_to_text",
                                         apply_augmentation: bool = True,
                                         augmentation_factor: int = 2,
                                         save_processed_data: bool = True,
                                         max_workers: int = 4,
                                         **dataset_kwargs) -> Dict[str, Any]:
        """Complete end-to-end pipeline from data fetch to training-ready data"""
        logger.info(f"Starting complete training pipeline for {training_type}")

        # Step 1: Get raw samples (from backend, local, or dummy)
        logger.info("Fetching raw audio samples...")
        samples = self.fetch_data_from_backend_sync()

        if not samples:
            raise ValueError("No samples available for training pipeline")

        # Step 2: Prepare training data with full pipeline
        training_data = self.prepare_training_data(
            samples=samples,
            feature_type=feature_type,
            training_type=training_type,
            apply_augmentation=apply_augmentation,
            augmentation_factor=augmentation_factor,
            max_workers=max_workers,
            **dataset_kwargs
        )

        # Step 3: Save processed data (optional)
        if save_processed_data:
            output_path = self.save_training_data(training_data)
            training_data['saved_to'] = str(output_path)

        logger.info("Complete training pipeline finished successfully")
        return training_data

    def save_cache(self, data: Any, cache_key: str):
        """Save data to cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

    def load_cache(self, cache_key: str) -> Optional[Any]:
        """Load data from cache

        Args:
            cache_key: Unique identifier for cached data

        Returns:
            Cached data if exists, None otherwise
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_key}: {e}")
                return None
        return None
