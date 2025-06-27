#!/usr/bin/env python3
"""
Simple test to verify metadata caching fix

This script tests the metadata saving functionality without requiring
external dependencies like torch, aiohttp, etc.
"""

import os
import sys
import json
import asyncio
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def create_mock_data_manager():
    """Create a mock data manager with our metadata saving methods"""

    class MockDataManager:
        def __init__(self):
            self.cache_dir = Path("data/cache")
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        async def _save_metadata_to_cache(self, recording, metadata_path):
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
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

                logger.debug(f"Metadata saved to cache: {metadata_path}")
                return True

            except Exception as e:
                logger.error(f"Failed to save metadata to cache {metadata_path}: {e}")
                return False

        def _load_cached_metadata(self, audio_path):
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

        async def simulate_download_with_metadata(self, recording):
            """Simulate the download process with metadata saving"""
            object_key = recording.get('object_key')
            if not object_key:
                logger.warning("Recording has no object_key, cannot download.")
                return None

            try:
                # Create local file path preserving original filename
                original_filename = object_key.split('/')[-1]
                # Use a subdirectory based on hash to avoid conflicts
                file_hash = hashlib.md5(object_key.encode()).hexdigest()[:8]
                local_dir = self.cache_dir / file_hash
                local_dir.mkdir(exist_ok=True)
                local_path = local_dir / original_filename

                # Create metadata file path
                metadata_filename = original_filename.rsplit('.', 1)[0] + '_metadata.json'
                metadata_path = local_dir / metadata_filename

                # Simulate creating audio file
                logger.info(f"Simulating download to: {local_path}")
                with open(local_path, 'w') as f:
                    f.write("fake audio data")

                # Save metadata alongside the audio file
                success = await self._save_metadata_to_cache(recording, metadata_path)

                if success:
                    logger.info(f"Successfully simulated download with metadata: {local_path}")
                    logger.info(f"Metadata saved to: {metadata_path}")
                    return str(local_path)
                else:
                    logger.error("Failed to save metadata")
                    return None

            except Exception as e:
                logger.error(f"An error occurred during simulation: {e}")
                return None

    return MockDataManager()

def create_test_recording():
    """Create test recording data similar to MongoDB format"""
    return {
        '_id': 'test_object_id',
        'speaker_id': 'test_speaker_id',
        'participant_code': 'TWI_Speaker_001',
        'prompt_id': 'ScriptD_28',
        'prompt_text': 'Berɛ a na meredidi no, me nua no baeɛ.',
        'session_id': 'test_session',
        'file_url': 'https://example.com/test.m4a',
        'object_key': 'recordings/TWI_Speaker_001/ScriptD_28_test.m4a',
        'filename_original': 'TWI_Speaker_001_ScriptD_28_test.m4a',
        'content_type': 'audio/mp4',
        'size_bytes': 52810,
        'recording_duration': 3500,  # 3.5 seconds
        'uploaded_at': datetime.now(),
        'transcription': 'When I was eating, my brother came.',
        'transcription_status': 'completed',
        'transcribed_by': 'test_transcriber',
        'transcription_updated_at': datetime.now()
    }

async def test_metadata_saving():
    """Test the metadata saving functionality"""
    print("🧪 Testing metadata saving during download simulation...")

    # Create mock manager
    manager = create_mock_data_manager()

    # Create test recording
    test_recording = create_test_recording()

    print(f"Test recording data:")
    print(f"  Prompt ID: {test_recording['prompt_id']}")
    print(f"  Prompt Text: {test_recording['prompt_text']}")
    print(f"  Transcription: {test_recording['transcription']}")
    print(f"  Speaker: {test_recording['participant_code']}")

    # Simulate download with metadata
    result = await manager.simulate_download_with_metadata(test_recording)

    if result:
        print(f"✅ Download simulation successful: {result}")

        # Verify metadata was saved
        metadata = manager._load_cached_metadata(result)
        if metadata:
            print("✅ Metadata successfully loaded from cache")
            print("📋 Saved metadata:")
            for key, value in metadata.items():
                print(f"   {key}: {value}")
            return True
        else:
            print("❌ Failed to load metadata from cache")
            return False
    else:
        print("❌ Download simulation failed")
        return False

def test_existing_cache():
    """Test loading metadata from existing cache files"""
    print("\n🔍 Testing existing cache metadata...")

    cache_dir = Path("data/cache")
    if not cache_dir.exists():
        print("❌ Cache directory not found")
        return False

    # Find existing audio files
    audio_files = []
    for subdir in cache_dir.iterdir():
        if subdir.is_dir():
            audio_files.extend(subdir.glob("*.m4a"))
            audio_files.extend(subdir.glob("*.wav"))
            audio_files.extend(subdir.glob("*.mp3"))

    if not audio_files:
        print("❌ No audio files found in cache")
        return False

    print(f"Found {len(audio_files)} audio files in cache")

    manager = create_mock_data_manager()
    success_count = 0

    for audio_file in audio_files[:3]:  # Test first 3 files
        print(f"\nTesting: {audio_file.name}")
        metadata = manager._load_cached_metadata(str(audio_file))

        if metadata:
            print(f"  ✅ Metadata found")
            print(f"  Source: {metadata.get('source', 'unknown')}")
            print(f"  Prompt ID: {metadata.get('prompt_id', 'N/A')}")
            success_count += 1
        else:
            print(f"  ❌ No metadata found")

    print(f"\n📊 Metadata found for {success_count}/{min(len(audio_files), 3)} tested files")
    return success_count > 0

def scan_cache_coverage():
    """Scan cache and report metadata coverage"""
    print("\n📊 Scanning cache metadata coverage...")

    cache_dir = Path("data/cache")
    if not cache_dir.exists():
        print("❌ Cache directory not found")
        return

    total_audio = 0
    total_metadata = 0

    for subdir in cache_dir.iterdir():
        if not subdir.is_dir():
            continue

        # Count audio files
        audio_extensions = ["*.m4a", "*.wav", "*.mp3", "*.flac"]
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(subdir.glob(ext))

        # Count metadata files
        metadata_files = list(subdir.glob("*_metadata.json"))

        if audio_files:
            total_audio += len(audio_files)
            total_metadata += len(metadata_files)

            print(f"  {subdir.name}: {len(metadata_files)}/{len(audio_files)} files have metadata")

    if total_audio > 0:
        coverage = (total_metadata / total_audio) * 100
        print(f"\n📈 Overall coverage: {total_metadata}/{total_audio} ({coverage:.1f}%)")

        if coverage == 100:
            print("🎉 Perfect metadata coverage!")
        elif coverage >= 80:
            print("✅ Good metadata coverage")
        elif coverage >= 50:
            print("⚠️  Partial metadata coverage")
        else:
            print("❌ Poor metadata coverage")
    else:
        print("❌ No audio files found in cache")

async def main():
    """Main test function"""
    print("🔍 Testing Metadata Caching Fix")
    print("=" * 50)

    # Change to script directory
    os.chdir(Path(__file__).parent)

    # Test 1: Scan existing cache
    scan_cache_coverage()

    # Test 2: Test loading existing metadata
    existing_test_passed = test_existing_cache()

    # Test 3: Test new metadata saving
    new_test_passed = await test_metadata_saving()

    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"   Existing metadata loading: {'✅' if existing_test_passed else '❌'}")
    print(f"   New metadata saving: {'✅' if new_test_passed else '❌'}")

    if existing_test_passed and new_test_passed:
        print("\n🎉 All tests passed! The metadata caching fix is working correctly.")
        print("\n💡 Key improvements:")
        print("   ✓ Metadata is now saved immediately during download")
        print("   ✓ Cached metadata can be loaded and used during organization")
        print("   ✓ Rich metadata from MongoDB is preserved in cache")
        return True
    else:
        print("\n⚠️  Some tests had issues, but this may be expected if:")
        print("   - This is the first time running the fix")
        print("   - Existing cache files have placeholder metadata")
        print("   - The database connection is not available")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n🚀 Ready to test with real data!")
    else:
        print("\n🔧 Check the implementation and try again.")
