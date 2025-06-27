"""
Shared Data Structures for Twi Speech Recognition Training Engine

This module contains all shared data structures and classes to avoid circular imports.
It should be imported by other modules that need these structures.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AudioSample:
    """Data structure for audio samples"""
    audio_path: str
    transcription: str
    duration: float
    speaker_id: str
    dialect: str
    gender: Optional[str] = None
    age_range: Optional[str] = None
    prompt_id: str = ""
    prompt_text: Optional[str] = None
    meaning: Optional[str] = None
    session_id: Optional[str] = None
    quality_score: float = 1.0
    original_filename: Optional[str] = None


@dataclass
class DatasetStats:
    """Dataset statistics"""
    total_samples: int
    total_duration: float
    avg_duration: float
    min_duration: float
    max_duration: float
    dialect_distribution: Dict[str, int]
    gender_distribution: Dict[str, int]
    speaker_count: int
    vocabulary_size: int
    character_count: int


@dataclass
class AudioRecording:
    """Represents a single audio recording"""
    filename: str
    prompt_id: str
    prompt_text: str
    prompt_meaning: str
    speaker_id: str
    duration: float
    sample_rate: int
    recording_date: str
    file_path: str
    quality_score: Optional[float] = None


@dataclass
class RecordingSession:
    """Represents a recording session"""
    session_id: str
    speaker_id: str
    start_time: str
    end_time: Optional[str] = None
    total_recordings: int = 0
    total_duration: float = 0.0
    quality_score: Optional[float] = None
    completed: bool = False


@dataclass
class TrainingExample:
    """Enhanced training example with multiple text targets"""
    audio_path: str
    audio_features: Optional[np.ndarray] = None
    twi_text: str = ""           # Twi transcription (prompt_text)
    english_text: str = ""       # English translation (meaning)
    speaker_id: str = ""
    dialect: str = ""
    duration: float = 0.0
    metadata: Dict[str, Any] = None


@dataclass
class ProcessingConfig:
    """Configuration for audio processing"""
    sample_rate: int = 16000
    channels: int = 1
    normalize_audio: bool = True
    trim_silence: bool = True
    apply_augmentation: bool = False
    augmentation_strength: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for training"""
    training_type: str = "speech_to_text"  # speech_to_text, translation, multilingual, cross_lingual
    feature_type: str = "wav2vec2"  # wav2vec2, mfcc, mel_spectrogram
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 10
    augmentation_factor: int = 2
    max_workers: int = 4
    save_processed_data: bool = True


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    wer: float = 0.0  # Word Error Rate
    cer: float = 0.0  # Character Error Rate
    bleu: float = 0.0  # BLEU score (for translation)
    loss: float = 0.0
    accuracy: float = 0.0
    perplexity: float = 0.0


@dataclass
class ExperimentResult:
    """Results from a training experiment"""
    experiment_id: str
    model_type: str
    training_type: str
    dataset_size: int
    training_time: float
    best_metrics: ModelMetrics
    config: Dict[str, Any]
    model_path: Optional[str] = None
    logs_path: Optional[str] = None
