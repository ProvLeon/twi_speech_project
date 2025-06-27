#!/usr/bin/env python3
"""
Test script to verify metadata saving in cache during download

This script tests the fixed metadata caching functionality to ensure
that metadata is saved alongside audio files in the cache directory.
"""

import os
import sys
import asyncio
import logging
import json
from pathlib import Path
from datetime import datetime

# Add src to Python path
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
sys.path.insert(0, str(src_dir))

from src.data.data_manager import TwiDataManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def check_cache_metadata():
    """Check existing cache directories for metadata files"""
    cache_dir = Path("data/cache")

    if not cache_dir.exists():
        print("❌ Cache directory not found")
        return False

    print(f"📁 Checking cache directory: {cache_dir.absolute()}")

    # List all subdirectories in cache
    cache_subdirs = [d for d in cache_dir.iterdir() if d.is_dir()]
    print(f"Found {len(cache_subdirs)} cache subdirectories")

    total_audio_files = 0
    total_metadata_files = 0

    for subdir in cache_subdirs:
        print(f"\n📂 Checking {subdir.name}:")

        audio_files = list(subdir.glob("*.m4a")) + list(subdir.glob("*.wav")) + list(subdir.glob("*.mp3"))
        metadata_files = list(subdir.glob("*_metadata.json"))

        total_audio_files += len(audio_files)
        total_metadata_files += len(metadata_files)

        print(f"   Audio files: {len(audio_files)}")
        print(f"   Metadata files: {len(metadata_files)}")

        # Check each audio file for corresponding metadata
        for audio_file in audio_files:
            metadata_name = audio_file.stem + "_metadata.json"
            metadata_path = subdir / metadata_name

            if metadata_path.exists():
                print(f"   ✅ {audio_file.name} -> {metadata_name}")

                # Try to load and display metadata
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)

                    print(f"      Prompt ID: {metadata.get('prompt_id', 'N/A')}")
                    print(f"      Prompt Text: {metadata.get('prompt_text', 'N/A')[:50]}...")
                    print(f"      Speaker ID: {metadata.get('speaker_id', 'N/A')}")
                    print(f"      Cached At: {metadata.get('cached_at', 'N/A')}")

                except Exception as e:
                    print(f"      ❌ Error reading metadata: {e}")

            else:
                print(f"   ❌ {audio_file.name} -> MISSING {metadata_name}")

    print(f"\n📊 Cache Summary:")
    print(f"   Total audio files: {total_audio_files}")
    print(f"   Total metadata files: {total_metadata_files}")
    print(f"   Coverage: {(total_metadata_files/total_audio_files*100):.1f}%" if total_audio_files > 0 else "   Coverage: N/A")

    return total_metadata_files > 0

async def test_single_download():
    """Test downloading a single file and check metadata saving"""
    print("\n🧪 Testing single file download with metadata...")

    try:
        # Load config
        config_path = Path("configs/default_config.yaml")
        if not config_path.exists():
            print("❌ Config file not found")
            return False

        # Initialize data manager
        manager = TwiDataManager(str(config_path))

        # Create a dummy recording object to test
        dummy_recording = {
            'object_key': 'test/dummy_file.m4a',
            'prompt_id': 'TEST_001',
            'prompt_text': 'Test prompt text',
            'speaker_id': 'TEST_SPEAKER',
            'transcription': 'Test transcription',
            'session_id': 'TEST_SESSION',
            'filename_original': 'test_file.m4a',
            'content_type': 'audio/mp4',
            'size_bytes': 12345,
            'recording_duration': 3000,
            'uploaded_at': datetime.now(),
            'transcription_status': 'completed',
            'transcribed_by': 'test_user',
            'transcription_updated_at': datetime.now()
        }

        # Test the metadata saving function directly
        import hashlib
        file_hash = hashlib.md5(dummy_recording['object_key'].encode()).hexdigest()[:8]
        cache_subdir = Path("data/cache") / file_hash
        cache_subdir.mkdir(parents=True, exist_ok=True)

        metadata_path = cache_subdir / "test_metadata.json"

        print(f"Testing metadata save to: {metadata_path}")
        await manager._save_metadata_to_cache(dummy_recording, metadata_path)

        # Check if metadata was saved
        if metadata_path.exists():
            print("✅ Metadata file created successfully")

            with open(metadata_path, 'r', encoding='utf-8') as f:
                saved_metadata = json.load(f)

            print("📋 Saved metadata contents:")
            for key, value in saved_metadata.items():
                print(f"   {key}: {value}")

            return True
        else:
            print("❌ Metadata file was not created")
            return False

    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metadata_loading():
    """Test loading cached metadata"""
    print("\n🧪 Testing metadata loading from cache...")

    try:
        config_path = Path("configs/default_config.yaml")
        if not config_path.exists():
            print("❌ Config file not found")
            return False

        manager = TwiDataManager(str(config_path))

        # Find an existing audio file in cache
        cache_dir = Path("data/cache")
        audio_files = []

        for subdir in cache_dir.iterdir():
            if subdir.is_dir():
                audio_files.extend(subdir.glob("*.m4a"))
                audio_files.extend(subdir.glob("*.wav"))
                audio_files.extend(subdir.glob("*.mp3"))

        if not audio_files:
            print("❌ No audio files found in cache")
            return False

        # Test loading metadata for the first audio file
        test_audio = audio_files[0]
        print(f"Testing metadata loading for: {test_audio}")

        metadata = manager._load_cached_metadata(str(test_audio))

        if metadata:
            print("✅ Metadata loaded successfully")
            print("📋 Loaded metadata:")
            for key, value in metadata.items():
                print(f"   {key}: {value}")
            return True
        else:
            print("❌ No metadata found or failed to load")
            return False

    except Exception as e:
        print(f"❌ Error during metadata loading test: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("🔍 Testing Metadata Cache Functionality")
    print("=" * 50)

    # Test 1: Check existing cache for metadata
    print("\n1️⃣ Checking existing cache metadata...")
    has_existing_metadata = check_cache_metadata()

    # Test 2: Test metadata saving
    print("\n2️⃣ Testing metadata saving...")
    save_test_passed = await test_single_download()

    # Test 3: Test metadata loading
    print("\n3️⃣ Testing metadata loading...")
    load_test_passed = test_metadata_loading()

    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"   Existing metadata in cache: {'✅' if has_existing_metadata else '❌'}")
    print(f"   Metadata saving test: {'✅' if save_test_passed else '❌'}")
    print(f"   Metadata loading test: {'✅' if load_test_passed else '❌'}")

    if save_test_passed and load_test_passed:
        print("\n🎉 All tests passed! Metadata caching is working correctly.")
        return True
    else:
        print("\n❌ Some tests failed. Check the logs above for details.")
        return False

if __name__ == "__main__":
    # Change to the script directory
    os.chdir(Path(__file__).parent)

    success = asyncio.run(main())
    sys.exit(0 if success else 1)
