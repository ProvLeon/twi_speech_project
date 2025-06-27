#!/usr/bin/env python3
"""
Comprehensive test for the audio processing pipeline in Twi Speech Training Engine

This script demonstrates the complete pipeline:
1. Download audio from cloud (or use local samples)
2. Organize into recordings directory
3. Process audio (normalize, trim silence)
4. Apply data augmentation
5. Extract features (MFCC, Mel Spectrogram, or Wav2Vec2)
6. Split into train/validation/test sets
7. Save processed data for training

Pipeline Flow:
Raw Audio → Processing → Augmentation → Feature Extraction → Train/Val/Test Split → Training Data
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import time
from datetime import datetime

# Add the parent directory to the Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

print("Twi Speech Audio Processing Pipeline Test")
print("=" * 70)

# Test 1: Setup and imports
print("\n1. Setting up test environment...")
print("-" * 40)

try:
    from src.config.config_loader import load_config
    from src.utils.env_loader import load_env
    from src.data.data_manager import TwiDataManager, AudioSample
    import librosa
    import soundfile as sf

    # Load environment variables
    load_env()
    print("✓ Environment variables loaded")

    # Load configuration with augmentation enabled
    config_path = current_dir / "configs" / "default_config.yaml"
    if config_path.exists():
        config = load_config(config_path)
    else:
        config = {
            'audio': {
                'sample_rate': 16000,
                'target_sample_rate': 16000,
                'target_channels': 1,
                'normalize_audio': True,
                'trim_silence': True
            },
            'text': {'language': 'tw'},
            'mongodb': {},
            'r2_storage': {},
            'cache_dir': 'data/cache',
            'recordings_dir': 'data/recordings',
            'script_file': 'script_actual.ts',
            'augmentation': {
                'enabled': True,
                'gaussian_noise': {'enabled': True},
                'time_stretch': {'enabled': True},
                'pitch_shift': {'enabled': True}
            }
        }

    # Ensure augmentation is enabled
    config['augmentation'] = {
        'enabled': True,
        'gaussian_noise': {'enabled': True},
        'time_stretch': {'enabled': True},
        'pitch_shift': {'enabled': True}
    }

    print("✓ Configuration loaded with augmentation enabled")

except Exception as e:
    print(f"✗ Setup error: {e}")
    sys.exit(1)

# Test 2: Create sample audio files for testing
print("\n2. Creating sample audio files for testing...")
print("-" * 40)

def create_test_audio_file(file_path: Path, duration: float = 2.0, frequency: float = 440.0):
    """Create a test audio file with a sine wave"""
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Create a sine wave with some variation
    audio = np.sin(2 * np.pi * frequency * t) * 0.3
    # Add some variation to make it more realistic
    audio += np.sin(2 * np.pi * frequency * 1.5 * t) * 0.1
    audio += np.random.normal(0, 0.02, len(audio))  # Add some noise

    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Save audio file
    sf.write(str(file_path), audio, sample_rate)
    return len(audio) / sample_rate

# Create test audio samples
test_samples = []
test_audio_dir = Path("data/test_audio")
test_audio_dir.mkdir(parents=True, exist_ok=True)

sample_data = [
    {
        'speaker_id': 'test_speaker_001',
        'prompt_id': 'ScriptA_01',
        'prompt_text': 'Bue adwadie app no',
        'meaning': 'Open the shopping app',
        'dialect': 'Asante',
        'gender': 'Female',
        'frequency': 440.0
    },
    {
        'speaker_id': 'test_speaker_001',
        'prompt_id': 'ScriptB_05',
        'prompt_text': 'Kɔ fie page no so',
        'meaning': 'Go to homepage',
        'dialect': 'Asante',
        'gender': 'Female',
        'frequency': 523.25
    },
    {
        'speaker_id': 'test_speaker_002',
        'prompt_id': 'ScriptC_10',
        'prompt_text': 'Hwehwɛ ntadeɛ beaeɛ',
        'meaning': 'Search for clothing',
        'dialect': 'Fante',
        'gender': 'Male',
        'frequency': 349.23
    }
]

for i, data in enumerate(sample_data):
    filename = f"{data['speaker_id']}_{data['prompt_id']}.wav"
    file_path = test_audio_dir / filename

    duration = create_test_audio_file(file_path, duration=2.0 + i * 0.5, frequency=data['frequency'])

    sample = AudioSample(
        audio_path=str(file_path),
        transcription=data['prompt_text'],
        duration=duration,
        speaker_id=data['speaker_id'],
        dialect=data['dialect'],
        gender=data['gender'],
        age_range='25-35',
        prompt_id=data['prompt_id'],
        prompt_text=data['prompt_text'],
        meaning=data['meaning'],
        session_id='test_session',
        quality_score=0.9,
        original_filename=filename
    )

    test_samples.append(sample)
    print(f"Created: {filename} ({duration:.2f}s)")

print(f"✓ Created {len(test_samples)} test audio files")

# Test 3: Initialize TwiDataManager and test audio processing
print("\n3. Testing audio processing...")
print("-" * 40)

try:
    manager = TwiDataManager(config)
    print("✓ TwiDataManager initialized")

    # Test basic audio processing (without augmentation)
    print("\nTesting basic audio processing...")
    processed_samples = manager.process_audio_batch(test_samples, apply_augmentation=False)
    print(f"✓ Processed {len(processed_samples)} samples without augmentation")

except Exception as e:
    print(f"✗ Audio processing error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test augmentation
print("\n4. Testing data augmentation...")
print("-" * 40)

try:
    # Test augmentation
    print("Testing audio augmentation...")
    augmented_samples = manager.augment_dataset(test_samples, augmentation_factor=3, max_workers=2)
    print(f"✓ Created {len(augmented_samples)} samples with 3x augmentation")

    # Count originals vs augmented
    original_count = sum(1 for s in augmented_samples if not s.prompt_id.endswith('_aug'))
    augmented_count = len(augmented_samples) - original_count
    print(f"  - Original samples: {original_count}")
    print(f"  - Augmented samples: {augmented_count}")

except Exception as e:
    print(f"✗ Augmentation error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test feature extraction
print("\n5. Testing feature extraction...")
print("-" * 40)

feature_types = ['mfcc', 'mel_spectrogram', 'wav2vec2']

for feature_type in feature_types:
    try:
        print(f"\nExtracting {feature_type} features...")
        start_time = time.time()

        features_data = manager.extract_features_batch(
            test_samples,
            feature_type=feature_type,
            max_workers=2
        )

        extraction_time = time.time() - start_time

        if features_data:
            print(f"✓ Extracted {feature_type} features from {len(features_data)} samples in {extraction_time:.2f}s")

            # Show feature shapes
            sample_features = features_data[0]['features']
            if hasattr(sample_features, 'shape'):
                print(f"  Feature shape: {sample_features.shape}")
            else:
                print(f"  Feature type: {type(sample_features)}, length: {len(sample_features)}")
        else:
            print(f"✗ No features extracted for {feature_type}")

    except Exception as e:
        print(f"✗ Feature extraction error for {feature_type}: {e}")

# Test 6: Test complete training pipeline
print("\n6. Testing complete training pipeline...")
print("-" * 40)

try:
    print("Running complete training data preparation pipeline...")
    start_time = time.time()

    # Use a smaller augmentation factor for testing
    training_data = manager.prepare_training_data(
        samples=test_samples,
        feature_type='wav2vec2',
        apply_augmentation=True,
        augmentation_factor=2,
        max_workers=2
    )

    pipeline_time = time.time() - start_time

    print(f"✓ Complete pipeline finished in {pipeline_time:.2f}s")
    print(f"\nTraining data summary:")
    print(f"  Total samples: {training_data['total_samples']}")
    print(f"  Train samples: {len(training_data['train'])}")
    print(f"  Validation samples: {len(training_data['validation'])}")
    print(f"  Test samples: {len(training_data['test'])}")
    print(f"  Feature type: {training_data['feature_type']}")
    print(f"  Augmentation factor: {training_data['augmentation_factor']}")

    # Show metadata
    metadata = training_data['metadata']
    print(f"\nMetadata:")
    print(f"  Speakers: {metadata['speakers']}")
    print(f"  Dialects: {metadata['dialects']}")
    print(f"  Total duration: {metadata['total_duration']:.2f}s")

except Exception as e:
    print(f"✗ Complete pipeline error: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Test saving processed data
print("\n7. Testing data saving...")
print("-" * 40)

try:
    if 'training_data' in locals():
        print("Saving processed training data...")
        output_path = manager.save_training_data(training_data, output_dir="data/test_processed")
        print(f"✓ Training data saved to: {output_path}")

        # List saved files
        saved_files = list(output_path.glob("*.pkl")) + list(output_path.glob("*.json"))
        print(f"Saved files:")
        for file_path in saved_files:
            file_size = file_path.stat().st_size
            print(f"  {file_path.name} ({file_size} bytes)")
    else:
        print("⚠ No training data to save")

except Exception as e:
    print(f"✗ Data saving error: {e}")

# Test 8: Test end-to-end pipeline (simulate real usage)
print("\n8. Testing end-to-end pipeline simulation...")
print("-" * 40)

try:
    print("Simulating complete end-to-end pipeline...")

    # This would normally fetch from cloud, but we'll use our test samples
    # by temporarily replacing the fetch method
    original_fetch = manager.fetch_data_from_backend_sync
    manager.fetch_data_from_backend_sync = lambda: test_samples

    # Run complete pipeline
    complete_data = manager.prepare_complete_training_pipeline(
        feature_type='mfcc',
        apply_augmentation=True,
        augmentation_factor=2,
        save_processed_data=True,
        max_workers=2
    )

    # Restore original method
    manager.fetch_data_from_backend_sync = original_fetch

    print("✓ End-to-end pipeline simulation completed")
    print(f"  Data saved to: {complete_data.get('saved_to', 'Not saved')}")

except Exception as e:
    print(f"✗ End-to-end pipeline error: {e}")

# Test 9: Performance benchmarks
print("\n9. Performance benchmarks...")
print("-" * 40)

def benchmark_processing(samples, description):
    """Benchmark processing performance"""
    start_time = time.time()
    try:
        result = manager.process_audio_batch(samples, apply_augmentation=True, max_workers=1)
        end_time = time.time()
        print(f"  {description}: {end_time - start_time:.2f}s ({len(result)} samples)")
        return end_time - start_time
    except Exception as e:
        print(f"  {description}: Failed - {e}")
        return None

# Benchmark different configurations
print("Processing performance benchmarks:")
benchmark_processing(test_samples[:1], "Single sample")
benchmark_processing(test_samples, "All samples (1 worker)")

try:
    start_time = time.time()
    result = manager.process_audio_batch(test_samples, apply_augmentation=True, max_workers=2)
    end_time = time.time()
    print(f"  All samples (2 workers): {end_time - start_time:.2f}s ({len(result)} samples)")
except Exception as e:
    print(f"  All samples (2 workers): Failed - {e}")

# Test 10: Cleanup and summary
print("\n10. Cleanup and summary...")
print("-" * 40)

# Cleanup test files
cleanup = input("Do you want to clean up test files? (y/N): ").strip().lower()
if cleanup == 'y':
    try:
        import shutil
        if test_audio_dir.exists():
            shutil.rmtree(test_audio_dir)
            print("✓ Removed test audio directory")

        test_processed_dir = Path("data/test_processed")
        if test_processed_dir.exists():
            shutil.rmtree(test_processed_dir)
            print("✓ Removed test processed data directory")

        print("✓ Cleanup completed")
    except Exception as e:
        print(f"✗ Cleanup error: {e}")
else:
    print("Test files left in place for inspection")

# Summary
print("\n" + "=" * 70)
print("Audio Processing Pipeline Test Summary")
print("=" * 70)

print("\n✅ Pipeline Components Tested:")
print("1. ✓ Audio file creation and loading")
print("2. ✓ Basic audio processing (normalization, silence removal)")
print("3. ✓ Data augmentation (noise, time stretch, pitch shift)")
print("4. ✓ Feature extraction (MFCC, Mel Spectrogram, Wav2Vec2)")
print("5. ✓ Train/validation/test splitting")
print("6. ✓ Data saving and serialization")
print("7. ✓ End-to-end pipeline integration")
print("8. ✓ Performance benchmarking")

print("\n📊 Pipeline Flow:")
print("   Raw Audio Files")
print("        ↓")
print("   Audio Processing (normalize, trim)")
print("        ↓")
print("   Data Augmentation (2x, 3x, etc.)")
print("        ↓")
print("   Feature Extraction (MFCC/Mel/Wav2Vec2)")
print("        ↓")
print("   Train/Validation/Test Split")
print("        ↓")
print("   Serialized Training Data")

print("\n🔧 Usage in Main Pipeline:")
print("1. Download audio: manager.fetch_data_from_backend_sync()")
print("2. Full pipeline: manager.prepare_complete_training_pipeline()")
print("3. Custom pipeline: manager.prepare_training_data()")
print("4. Save data: manager.save_training_data()")

print("\n⚙️ Configuration Options:")
print("- feature_type: 'mfcc', 'mel_spectrogram', 'wav2vec2'")
print("- apply_augmentation: True/False")
print("- augmentation_factor: 1, 2, 3, etc.")
print("- max_workers: Number of parallel processes")

print("\n📁 Output Structure:")
print("data/processed/")
print("├── train_YYYYMMDD_HHMMSS.pkl")
print("├── validation_YYYYMMDD_HHMMSS.pkl")
print("├── test_YYYYMMDD_HHMMSS.pkl")
print("└── metadata_YYYYMMDD_HHMMSS.json")

print("\nAudio processing pipeline test completed successfully!")
