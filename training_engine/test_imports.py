#!/usr/bin/env python3
"""
Test script to verify all imports are working correctly in the Twi Speech Training Engine
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

print("Testing imports for Twi Speech Training Engine...")
print("=" * 60)

# Test core imports
try:
    print("Testing core imports...")
    import torch
    import torchaudio
    import numpy as np
    print("✓ Core ML libraries imported successfully")
except ImportError as e:
    print(f"✗ Error importing core libraries: {e}")

# Test audio processing imports
try:
    print("\nTesting audio processing imports...")
    import librosa
    import soundfile as sf
    try:
        import sounddevice as sd
        print("✓ Audio processing libraries imported successfully (including sounddevice)")
    except ImportError:
        print("⚠ Audio processing libraries imported (sounddevice not available)")
except ImportError as e:
    print(f"✗ Error importing audio libraries: {e}")

# Test data processing imports
try:
    print("\nTesting data processing imports...")
    import pandas as pd
    from sklearn.model_selection import train_test_split
    print("✓ Data processing libraries imported successfully")
except ImportError as e:
    print(f"✗ Error importing data processing libraries: {e}")

# Test database imports
try:
    print("\nTesting database imports...")
    import motor.motor_asyncio
    import boto3
    print("✓ Database libraries imported successfully")
except ImportError as e:
    print(f"✗ Error importing database libraries: {e}")

# Test project-specific imports
print("\nTesting project-specific imports...")

# Test src modules
try:
    from src.models.advanced_speech_model import SuperiorTwiSpeechModel
    print("✓ Models module imported successfully")
except ImportError as e:
    print(f"✗ Error importing models: {e}")

try:
    from src.features.advanced_feature_extractor import SuperiorFeatureExtractor
    from src.features.dataset_utils import DatasetManager
    print("✓ Features modules imported successfully")
except ImportError as e:
    print(f"✗ Error importing features: {e}")

try:
    from src.trainers.superior_trainer import SuperiorTrainer
    print("✓ Trainers module imported successfully")
except ImportError as e:
    print(f"✗ Error importing trainers: {e}")

try:
    from src.data.data_manager import TwiDataManager
    from src.data.audio_recorder import interactive_recording_session
    print("✓ Data modules imported successfully")
except ImportError as e:
    print(f"✗ Error importing data modules: {e}")

try:
    from src.config.config_loader import load_config, ConfigLoader
    print("✓ Config module imported successfully")
except ImportError as e:
    print(f"✗ Error importing config: {e}")

try:
    from src.utils.logging_utils import setup_logging
    from src.utils.metrics import compute_wer
    print("✓ Utils modules imported successfully")
except ImportError as e:
    print(f"✗ Error importing utils: {e}")

try:
    from src.evaluation.evaluator import SpeechEvaluator
    print("✓ Evaluation module imported successfully")
except ImportError as e:
    print(f"✗ Error importing evaluation: {e}")

# Test configuration loading
print("\nTesting configuration loading...")
try:
    from src.config.config_loader import load_config
    config_path = current_dir / "configs" / "default_config.yaml"
    if config_path.exists():
        config = load_config(config_path)
        print(f"✓ Configuration loaded successfully from {config_path}")
        print(f"  - Audio sample rate: {config.get('audio', {}).get('sample_rate', 'Not set')}")
        print(f"  - Model type: {config.get('model', {}).get('type', 'Not set')}")
    else:
        print(f"⚠ Config file not found at {config_path}, testing with defaults...")
        config = load_config("non_existent.yaml", use_defaults=True)
        print("✓ Default configuration loaded successfully")
except Exception as e:
    print(f"✗ Error loading configuration: {e}")

# Test environment variables
print("\nChecking environment variables...")
env_vars = {
    'MONGODB_URI': os.getenv('MONGODB_URI', 'Not set'),
    'R2_BUCKET_NAME': os.getenv('R2_BUCKET_NAME', 'Not set'),
    'R2_ACCESS_KEY_ID': os.getenv('R2_ACCESS_KEY_ID', 'Not set'),
    'R2_ENDPOINT_URL': os.getenv('R2_ENDPOINT_URL', 'Not set'),
}

for var, value in env_vars.items():
    if value != 'Not set':
        print(f"✓ {var}: Set")
    else:
        print(f"⚠ {var}: Not set (will use fallback)")

print("\n" + "=" * 60)
print("Import test completed!")
print("\nTo fix any missing dependencies, run:")
print("  pip install -r requirements.txt")
print("\nTo set environment variables (optional), use:")
print("  export MONGODB_URI='your_mongodb_uri'")
print("  export R2_BUCKET_NAME='your_bucket_name'")
print("  # etc...")
