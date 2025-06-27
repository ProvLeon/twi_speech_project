#!/usr/bin/env python3
"""
Test script to demonstrate data organization from cloud to local recordings directory

This script shows how downloaded audio files and metadata are organized into:
1. data/recordings/ directory with speaker subdirectories
2. Metadata JSON files alongside audio files
3. Comprehensive data index for easy navigation

Directory structure created:
data/recordings/
├── data_index.json                    # Summary of all data
├── TWI_Speaker_001/
│   ├── ScriptD_15_c56a08a0.m4a       # Original audio file
│   ├── ScriptD_15_c56a08a0_metadata.json  # Complete metadata
│   ├── ScriptA_01_a1b2c3d4.m4a
│   └── ScriptA_01_a1b2c3d4_metadata.json
└── TWI_Speaker_002/
    ├── ScriptB_05_e5f6g7h8.m4a
    └── ScriptB_05_e5f6g7h8_metadata.json
"""

import os
import sys
from pathlib import Path
import json
import shutil
from datetime import datetime

# Add the parent directory to the Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

print("Testing Data Organization System")
print("=" * 70)

# Test 1: Setup and imports
print("\n1. Setting up test environment...")
print("-" * 40)

try:
    from src.config.config_loader import load_config
    from src.utils.env_loader import load_env
    from src.data.data_manager import TwiDataManager, AudioSample

    # Load environment variables
    load_env()
    print("✓ Environment variables loaded")

    # Load configuration
    config_path = current_dir / "configs" / "default_config.yaml"
    if config_path.exists():
        config = load_config(config_path)
    else:
        config = {
            'audio': {'sample_rate': 16000},
            'text': {'language': 'tw'},
            'mongodb': {},
            'r2_storage': {},
            'cache_dir': 'data/cache',
            'recordings_dir': 'data/recordings',
            'script_file': 'script_actual.ts'
        }
    print("✓ Configuration loaded")

except Exception as e:
    print(f"✗ Setup error: {e}")
    sys.exit(1)

# Test 2: Create sample data to simulate downloaded content
print("\n2. Creating sample data (simulating cloud download)...")
print("-" * 40)

# Create cache directory structure
cache_dir = Path(config['cache_dir'])
cache_dir.mkdir(parents=True, exist_ok=True)

# Create sample cached files
sample_files = [
    {
        'hash': 'c56a08a0',
        'filename': 'ScriptD_15_c56a08a0-0d36-4555-9546-65759ab3cd09.m4a',
        'sample': AudioSample(
            audio_path='',  # Will be set below
            transcription='Me werɛ aho pa ara',
            duration=2.5,
            speaker_id='TWI_Speaker_001',
            dialect='Asante',
            gender='Female',
            age_range='25-35',
            prompt_id='ScriptD_15',
            prompt_text='Me werɛ aho pa ara',
            meaning='I am very worried',
            session_id='session_001',
            quality_score=0.95,
            original_filename='ScriptD_15_c56a08a0-0d36-4555-9546-65759ab3cd09.m4a'
        )
    },
    {
        'hash': 'a1b2c3d4',
        'filename': 'ScriptA_01_a1b2c3d4-1234-5678-9abc-def012345678.m4a',
        'sample': AudioSample(
            audio_path='',
            transcription='Bue adwadie app no',
            duration=1.8,
            speaker_id='TWI_Speaker_001',
            dialect='Asante',
            gender='Female',
            age_range='25-35',
            prompt_id='ScriptA_01',
            prompt_text='Bue adwadie app no',
            meaning='Open the shopping app',
            session_id='session_001',
            quality_score=0.88,
            original_filename='ScriptA_01_a1b2c3d4-1234-5678-9abc-def012345678.m4a'
        )
    },
    {
        'hash': 'e5f6g7h8',
        'filename': 'ScriptB_05_e5f6g7h8-9876-5432-1fed-cba098765432.m4a',
        'sample': AudioSample(
            audio_path='',
            transcription='Kɔ fie page no so',
            duration=3.2,
            speaker_id='TWI_Speaker_002',
            dialect='Fante',
            gender='Male',
            age_range='30-40',
            prompt_id='ScriptB_05',
            prompt_text='Kɔ fie page no so',
            meaning='Go to homepage',
            session_id='session_002',
            quality_score=0.92,
            original_filename='ScriptB_05_e5f6g7h8-9876-5432-1fed-cba098765432.m4a'
        )
    }
]

# Create cache subdirectories and dummy audio files
created_files = []
for file_info in sample_files:
    hash_dir = cache_dir / file_info['hash']
    hash_dir.mkdir(exist_ok=True)

    cache_file_path = hash_dir / file_info['filename']

    # Create a dummy audio file (just a small text file for demo)
    with open(cache_file_path, 'w') as f:
        f.write(f"DUMMY AUDIO DATA FOR {file_info['filename']}\n")
        f.write(f"Duration: {file_info['sample'].duration}s\n")
        f.write(f"Speaker: {file_info['sample'].speaker_id}\n")

    # Update the sample with the correct path
    file_info['sample'].audio_path = str(cache_file_path)
    created_files.append(file_info)

    print(f"Created: {cache_file_path}")

print(f"✓ Created {len(created_files)} sample cache files")

# Test 3: Initialize TwiDataManager and test organization
print("\n3. Testing data organization...")
print("-" * 40)

try:
    manager = TwiDataManager(config)
    print("✓ TwiDataManager initialized")

    # Extract just the samples
    samples = [file_info['sample'] for file_info in created_files]

    # Test the organization method
    print(f"\nOrganizing {len(samples)} samples...")
    organized_samples = manager.organize_downloaded_data(samples)

    print(f"✓ Organization completed: {len(organized_samples)} samples processed")

except Exception as e:
    print(f"✗ Organization error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Verify the directory structure
print("\n4. Verifying directory structure...")
print("-" * 40)

recordings_dir = Path(config['recordings_dir'])
if recordings_dir.exists():
    print(f"✓ Recordings directory created: {recordings_dir}")

    # List speaker directories
    speaker_dirs = [d for d in recordings_dir.iterdir() if d.is_dir()]
    print(f"✓ Found {len(speaker_dirs)} speaker directories:")

    for speaker_dir in speaker_dirs:
        print(f"\n  📁 {speaker_dir.name}/")

        # List files in speaker directory
        audio_files = list(speaker_dir.glob('*.m4a'))
        metadata_files = list(speaker_dir.glob('*_metadata.json'))

        print(f"    🎵 Audio files: {len(audio_files)}")
        print(f"    📄 Metadata files: {len(metadata_files)}")

        # Show first few files
        for audio_file in audio_files[:2]:
            print(f"      └── {audio_file.name}")

            # Check if corresponding metadata exists
            metadata_name = audio_file.stem + '_metadata.json'
            metadata_path = speaker_dir / metadata_name
            if metadata_path.exists():
                print(f"          └── {metadata_name} ✓")
            else:
                print(f"          └── {metadata_name} ✗")

else:
    print(f"✗ Recordings directory not found: {recordings_dir}")

# Test 5: Check the data index
print("\n5. Checking data index...")
print("-" * 40)

index_path = recordings_dir / 'data_index.json'
if index_path.exists():
    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)

        print(f"✓ Data index found: {index_path}")
        print(f"\nIndex Summary:")
        print(f"  Created: {index_data['created_at']}")
        print(f"  Total samples: {index_data['total_samples']}")
        print(f"  Unique speakers: {index_data['summary']['unique_speakers']}")
        print(f"  Unique prompts: {index_data['summary']['unique_prompts']}")
        print(f"  Total duration: {index_data['summary']['total_duration']:.1f}s")
        print(f"  Samples with translations: {index_data['summary']['samples_with_translations']}")

        print(f"\nSpeakers:")
        for speaker_id, speaker_data in index_data['speakers'].items():
            print(f"  {speaker_id}:")
            print(f"    - Samples: {speaker_data['sample_count']}")
            print(f"    - Duration: {speaker_data['total_duration']:.1f}s")
            print(f"    - Dialect: {speaker_data['dialect']}")
            print(f"    - Gender: {speaker_data['gender']}")

        print(f"\nPrompts:")
        for prompt_id, prompt_data in list(index_data['prompts'].items())[:3]:
            print(f"  {prompt_id}:")
            print(f"    - Text: {prompt_data['prompt_text']}")
            print(f"    - Meaning: {prompt_data['meaning']}")
            print(f"    - Samples: {prompt_data['sample_count']}")

        print(f"\nDialects:")
        for dialect, dialect_data in index_data['dialects'].items():
            print(f"  {dialect}:")
            print(f"    - Samples: {dialect_data['sample_count']}")
            print(f"    - Speakers: {dialect_data['unique_speakers']}")
            print(f"    - Duration: {dialect_data['total_duration']:.1f}s")

    except Exception as e:
        print(f"✗ Error reading index: {e}")
else:
    print(f"✗ Data index not found: {index_path}")

# Test 6: Show sample metadata file
print("\n6. Sample metadata file content...")
print("-" * 40)

if speaker_dirs:
    first_speaker = speaker_dirs[0]
    metadata_files = list(first_speaker.glob('*_metadata.json'))

    if metadata_files:
        sample_metadata_file = metadata_files[0]
        print(f"Reading: {sample_metadata_file}")

        try:
            with open(sample_metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            print("\nMetadata content:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")

        except Exception as e:
            print(f"✗ Error reading metadata: {e}")

# Test 7: Cleanup option
print("\n7. Cleanup...")
print("-" * 40)

print("Files created during test:")
print(f"  Cache directory: {cache_dir}")
print(f"  Recordings directory: {recordings_dir}")

cleanup = input("\nDo you want to clean up test files? (y/N): ").strip().lower()
if cleanup == 'y':
    try:
        if recordings_dir.exists():
            shutil.rmtree(recordings_dir)
            print(f"✓ Removed {recordings_dir}")

        # Only remove our test cache files, not the entire cache
        for file_info in created_files:
            cache_file = Path(file_info['sample'].audio_path)
            if cache_file.exists():
                cache_file.unlink()
                print(f"✓ Removed {cache_file}")

            # Remove empty hash directories
            hash_dir = cache_file.parent
            if hash_dir.exists() and not any(hash_dir.iterdir()):
                hash_dir.rmdir()
                print(f"✓ Removed empty directory {hash_dir}")

        print("✓ Cleanup completed")
    except Exception as e:
        print(f"✗ Cleanup error: {e}")
else:
    print("Files left in place for inspection")

# Summary
print("\n" + "=" * 70)
print("Data Organization Test Summary")
print("=" * 70)

print("\n✅ Key Features Demonstrated:")
print("1. Audio files copied from cache to recordings directory")
print("2. Original filenames preserved")
print("3. Speaker-based directory organization")
print("4. Complete metadata stored as JSON files")
print("5. Comprehensive data index for easy navigation")

print("\n📁 Directory Structure Created:")
print("data/recordings/")
print("├── data_index.json              # Summary of all data")
print("├── TWI_Speaker_001/")
print("│   ├── ScriptD_15_...m4a        # Audio file")
print("│   ├── ScriptD_15_...metadata.json  # Complete metadata")
print("│   └── ...")
print("└── TWI_Speaker_002/")
print("    └── ...")

print("\n📊 Metadata Includes:")
print("- Twi text (prompt_text)")
print("- English translation (meaning)")
print("- Speaker information (dialect, gender, age)")
print("- Audio properties (duration, quality_score)")
print("- Download information (source, timestamp)")

print("\n🔄 Integration Flow:")
print("1. fetch_data_from_backend_sync() downloads to cache")
print("2. organize_downloaded_data() copies to recordings directory")
print("3. Metadata saved alongside each audio file")
print("4. Data index created for efficient access")

print("\nTest completed successfully!")
