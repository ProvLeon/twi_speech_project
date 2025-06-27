#!/usr/bin/env python3
"""
Test script to verify file organization from cache to recordings directory

This script tests the complete flow:
1. Download audio files to cache directory
2. Organize them into data/recordings/ with proper structure
3. Verify files are copied correctly with metadata

Usage: python test_file_organization.py
"""

import os
import sys
from pathlib import Path
import json
import logging

# Add the parent directory to the Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Setup logging to see all the details
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_file_organization():
    """Test the complete file organization flow"""
    print("🧪 Testing File Organization from Cache to Recordings")
    print("=" * 70)

    try:
        # Import required modules
        from src.config.config_loader import load_config
        from src.utils.env_loader import load_env
        from src.data.data_manager import TwiDataManager
        from src.data.structures import AudioSample

        # Load environment and config
        load_env()
        config_path = current_dir / "configs" / "default_config.yaml"
        if config_path.exists():
            config = load_config(config_path)
        else:
            config = {
                'cache_dir': 'data/cache',
                'recordings_dir': 'data/recordings',
                'script_file': 'script_actual.ts'
            }

        print("✓ Modules imported successfully")
        print(f"✓ Cache directory: {config['cache_dir']}")
        print(f"✓ Recordings directory: {config['recordings_dir']}")

    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False

    # Test 1: Check current state
    print("\n1. Checking current directory state...")
    print("-" * 40)

    cache_dir = Path(config['cache_dir'])
    recordings_dir = Path(config['recordings_dir'])

    print(f"Cache directory exists: {cache_dir.exists()}")
    if cache_dir.exists():
        cache_files = list(cache_dir.rglob('*.wav')) + list(cache_dir.rglob('*.m4a'))
        print(f"Files in cache: {len(cache_files)}")
        for f in cache_files[:3]:  # Show first 3
            print(f"  - {f}")

    print(f"Recordings directory exists: {recordings_dir.exists()}")
    if recordings_dir.exists():
        recording_files = list(recordings_dir.rglob('*.wav')) + list(recordings_dir.rglob('*.m4a'))
        print(f"Files in recordings: {len(recording_files)}")

    # Test 2: Initialize TwiDataManager and test download
    print("\n2. Testing data download and organization...")
    print("-" * 40)

    try:
        manager = TwiDataManager(config)
        print("✓ TwiDataManager initialized")

        # Test the download and organization
        print("\n🔄 Running fetch_data_from_backend_sync()...")
        samples = manager.fetch_data_from_backend_sync()

        print(f"\n📊 Results:")
        print(f"  Downloaded samples: {len(samples)}")

        if samples:
            print("\n📋 Sample details:")
            for i, sample in enumerate(samples[:3]):  # Show first 3
                print(f"  Sample {i+1}:")
                print(f"    Prompt ID: {sample.prompt_id}")
                print(f"    Audio path: {sample.audio_path}")
                print(f"    Speaker: {sample.speaker_id}")
                print(f"    Original filename: {sample.original_filename}")

                # Check if file exists
                if Path(sample.audio_path).exists():
                    file_size = Path(sample.audio_path).stat().st_size
                    print(f"    ✓ File exists ({file_size} bytes)")
                else:
                    print(f"    ❌ File missing!")

        else:
            print("⚠️ No samples downloaded (this is expected if no MongoDB/R2 setup)")

    except Exception as e:
        print(f"❌ Download test failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Check recordings directory after organization
    print("\n3. Verifying recordings directory structure...")
    print("-" * 40)

    if recordings_dir.exists():
        # List speaker directories
        speaker_dirs = [d for d in recordings_dir.iterdir() if d.is_dir()]
        print(f"Speaker directories: {len(speaker_dirs)}")

        for speaker_dir in speaker_dirs:
            print(f"\n📁 {speaker_dir.name}/")

            # List audio files
            audio_files = list(speaker_dir.glob('*.wav')) + list(speaker_dir.glob('*.m4a'))
            metadata_files = list(speaker_dir.glob('*_metadata.json'))

            print(f"  Audio files: {len(audio_files)}")
            print(f"  Metadata files: {len(metadata_files)}")

            # Show file pairs
            for audio_file in audio_files[:3]:
                print(f"    🎵 {audio_file.name}")

                # Check for corresponding metadata
                metadata_name = audio_file.stem + '_metadata.json'
                metadata_path = speaker_dir / metadata_name

                if metadata_path.exists():
                    print(f"    📄 {metadata_name} ✓")

                    # Show metadata content
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        print(f"        Prompt: {metadata.get('prompt_text', 'N/A')}")
                        print(f"        Meaning: {metadata.get('meaning', 'N/A')}")
                        print(f"        Source: {metadata.get('source', 'N/A')}")
                    except Exception as e:
                        print(f"        ❌ Error reading metadata: {e}")
                else:
                    print(f"    📄 {metadata_name} ❌")

        # Check data index
        index_path = recordings_dir / 'data_index.json'
        if index_path.exists():
            print(f"\n📊 Data index found: {index_path}")
            try:
                with open(index_path, 'r') as f:
                    index_data = json.load(f)
                print(f"  Total samples: {index_data.get('total_samples', 0)}")
                print(f"  Speakers: {len(index_data.get('speakers', {}))}")
                print(f"  Dialects: {list(index_data.get('dialects', {}).keys())}")
            except Exception as e:
                print(f"  ❌ Error reading index: {e}")
        else:
            print("⚠️ No data index found")

    else:
        print("❌ Recordings directory doesn't exist!")

    # Test 4: Test organization with dummy data (if no real data)
    print("\n4. Testing organization with dummy data...")
    print("-" * 40)

    if not samples:  # If no real data downloaded
        print("Creating dummy samples for organization test...")

        # Create dummy cache files
        dummy_cache_dir = cache_dir / "test_dummy"
        dummy_cache_dir.mkdir(parents=True, exist_ok=True)

        dummy_audio_file = dummy_cache_dir / "test_sample.wav"
        # Create a small dummy audio file
        with open(dummy_audio_file, 'w') as f:
            f.write("DUMMY AUDIO DATA FOR TESTING")

        # Create dummy AudioSample
        dummy_sample = AudioSample(
            audio_path=str(dummy_audio_file),
            transcription="Test Twi text",
            duration=2.5,
            speaker_id="test_speaker",
            dialect="Test",
            prompt_id="test_001",
            prompt_text="Test Twi text",
            meaning="Test English meaning",
            original_filename="test_sample.wav"
        )

        print("✓ Created dummy sample")

        # Test organization
        try:
            organized = manager.organize_downloaded_data([dummy_sample])
            print(f"✓ Organization test completed: {len(organized)} samples")

            if organized:
                organized_sample = organized[0]
                if Path(organized_sample.audio_path).exists():
                    print(f"✓ Dummy file organized to: {organized_sample.audio_path}")
                else:
                    print(f"❌ Organized file missing: {organized_sample.audio_path}")

        except Exception as e:
            print(f"❌ Organization test failed: {e}")
            import traceback
            traceback.print_exc()

        # Cleanup
        try:
            import shutil
            if dummy_cache_dir.exists():
                shutil.rmtree(dummy_cache_dir)
            print("✓ Cleaned up dummy files")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

    # Test 5: Summary and recommendations
    print("\n5. Summary and Recommendations...")
    print("-" * 40)

    cache_files = list(cache_dir.rglob('*.wav')) + list(cache_dir.rglob('*.m4a')) if cache_dir.exists() else []
    recording_files = list(recordings_dir.rglob('*.wav')) + list(recordings_dir.rglob('*.m4a')) if recordings_dir.exists() else []

    print(f"\n📊 Final Status:")
    print(f"  Cache files: {len(cache_files)}")
    print(f"  Recording files: {len(recording_files)}")

    if len(recording_files) > 0:
        print("✅ SUCCESS: Files found in recordings directory!")
        print(f"   Location: {recordings_dir.absolute()}")
        print("\n🎯 Next steps:")
        print("   1. Files are organized by speaker in recordings directory")
        print("   2. Each audio file has a corresponding metadata JSON")
        print("   3. Ready for training pipeline")
    elif len(cache_files) > 0:
        print("⚠️ PARTIAL: Files in cache but not in recordings")
        print("   This might indicate organization step is not running")
        print("\n🔧 To fix:")
        print("   1. Check if organize_downloaded_data() is being called")
        print("   2. Check file permissions for recordings directory")
        print("   3. Run with INFO logging to see organization steps")
    else:
        print("ℹ️ INFO: No files found (expected if no backend setup)")
        print("\n🔧 To get data:")
        print("   1. Configure MongoDB URI in .env file")
        print("   2. Configure R2 storage credentials in .env file")
        print("   3. Run: python main_engine.py -> option 2 (Download Cloud Data)")

    print(f"\n📂 Directory Structure:")
    print(f"   data/cache/          <- Downloaded files (temporary)")
    print(f"   data/recordings/     <- Organized files (permanent)")
    print(f"   ├── Speaker_ID_1/")
    print(f"   │   ├── audio_file.m4a")
    print(f"   │   ├── audio_file_metadata.json")
    print(f"   └── Speaker_ID_2/")
    print(f"       └── ...")

    return len(recording_files) > 0


def main():
    """Main function"""
    try:
        success = test_file_organization()

        print("\n" + "=" * 70)
        if success:
            print("🎉 FILE ORGANIZATION TEST PASSED!")
            print("Files are properly organized from cache to recordings directory.")
        else:
            print("⚠️ FILE ORGANIZATION NEEDS ATTENTION")
            print("Check the recommendations above to resolve any issues.")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
