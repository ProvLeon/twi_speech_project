#!/usr/bin/env python3
"""
Utility script to retroactively add metadata to existing cache files

This script scans the cache directory for audio files that don't have
corresponding metadata files and attempts to create them by re-fetching
the data from MongoDB.

Usage:
    python fix_cache_metadata.py [--dry-run] [--force]
"""

import os
import sys
import asyncio
import logging
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

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

class CacheMetadataFixer:
    """Utility class to fix missing metadata in cache"""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.manager = None
        self.cache_dir = Path("data/cache")

    async def initialize(self):
        """Initialize the data manager"""
        try:
            self.manager = TwiDataManager(self.config_path)
            await self.manager._initialize_connections()
            logger.info("✅ Data manager initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize data manager: {e}")
            return False

    def scan_cache_for_missing_metadata(self) -> List[Dict[str, Any]]:
        """Scan cache directory for audio files without metadata"""
        missing_metadata = []

        if not self.cache_dir.exists():
            logger.warning("Cache directory not found")
            return missing_metadata

        logger.info(f"Scanning cache directory: {self.cache_dir.absolute()}")

        # Scan all subdirectories
        for subdir in self.cache_dir.iterdir():
            if not subdir.is_dir():
                continue

            # Find audio files
            audio_extensions = ['*.m4a', '*.wav', '*.mp3', '*.flac']
            audio_files = []
            for ext in audio_extensions:
                audio_files.extend(subdir.glob(ext))

            for audio_file in audio_files:
                # Check if metadata exists
                metadata_name = audio_file.stem + "_metadata.json"
                metadata_path = subdir / metadata_name

                if not metadata_path.exists():
                    missing_metadata.append({
                        'audio_path': audio_file,
                        'metadata_path': metadata_path,
                        'subdir': subdir,
                        'filename': audio_file.name
                    })
                    logger.debug(f"Missing metadata: {audio_file.name}")

        logger.info(f"Found {len(missing_metadata)} audio files without metadata")
        return missing_metadata

    def extract_info_from_filename(self, filename: str) -> Dict[str, str]:
        """Extract information from filename patterns"""
        info = {}

        # Try to extract prompt_id from common patterns
        # Example: ScriptD_28_a6c28307-50fa-4095-ab01-aaa69d14df92.m4a
        # Example: TWI_Speaker_001_ScriptD_28_1745175731530_eb4c1015.m4a

        parts = filename.split('_')

        # Look for ScriptX_Y pattern
        for i, part in enumerate(parts):
            if part.startswith('Script') and i + 1 < len(parts):
                try:
                    # Try to find prompt_id like ScriptD_28
                    if parts[i + 1].isdigit():
                        info['prompt_id'] = f"{part}_{parts[i + 1]}"
                        break
                except:
                    pass

        # Look for speaker pattern
        for part in parts:
            if 'Speaker' in part:
                info['speaker_hint'] = part
                break

        return info

    async def fetch_recording_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """Try to fetch recording data from MongoDB by filename"""
        if not self.manager or not self.manager.database:
            return None

        try:
            recordings_collection = self.manager.database.get_collection('audio_recordings')

            # Try different filename fields
            queries = [
                {'filename_original': filename},
                {'object_key': {'$regex': filename}},
                {'object_key': {'$regex': filename.replace('_', '.*')}},
            ]

            for query in queries:
                recording = await recordings_collection.find_one(query)
                if recording:
                    logger.debug(f"Found recording for {filename} using query: {query}")
                    return recording

            return None

        except Exception as e:
            logger.error(f"Error fetching recording for {filename}: {e}")
            return None

    async def create_metadata_from_filename(self, audio_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create basic metadata from filename when DB lookup fails"""
        filename = audio_info['filename']
        extracted_info = self.extract_info_from_filename(filename)

        metadata = {
            'prompt_id': extracted_info.get('prompt_id', 'unknown'),
            'prompt_text': '',
            'transcription': None,
            'speaker_id': extracted_info.get('speaker_hint', 'unknown'),
            'session_id': None,
            'object_key': f"inferred/{filename}",
            'filename_original': filename,
            'content_type': 'audio/mp4',
            'size_bytes': audio_info['audio_path'].stat().st_size,
            'recording_duration': None,
            'uploaded_at': None,
            'transcription_status': 'unknown',
            'transcribed_by': None,
            'transcription_updated_at': None,
            'cached_at': datetime.now().isoformat(),
            'source': 'inferred_from_filename',
            'note': 'Metadata created from filename analysis - may be incomplete'
        }

        return metadata

    async def fix_single_metadata(self, audio_info: Dict[str, Any], force: bool = False) -> bool:
        """Fix metadata for a single audio file"""
        audio_path = audio_info['audio_path']
        metadata_path = audio_info['metadata_path']
        filename = audio_info['filename']

        try:
            # Skip if metadata already exists and not forcing
            if metadata_path.exists() and not force:
                logger.debug(f"Metadata already exists for {filename}")
                return True

            # Try to fetch from database first
            recording = await self.fetch_recording_by_filename(filename)

            if recording:
                logger.info(f"Found DB record for {filename}")
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
                    'source': 'mongodb_backend',
                    'fixed_at': datetime.now().isoformat()
                }
            else:
                logger.warning(f"No DB record found for {filename}, creating from filename")
                metadata = await self.create_metadata_from_filename(audio_info)

            # Save metadata
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"✅ Created metadata for {filename}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to create metadata for {filename}: {e}")
            return False

    async def fix_all_metadata(self, dry_run: bool = False, force: bool = False) -> Dict[str, int]:
        """Fix metadata for all missing files"""
        missing_files = self.scan_cache_for_missing_metadata()

        if not missing_files:
            logger.info("🎉 All cache files already have metadata!")
            return {'total': 0, 'fixed': 0, 'failed': 0}

        logger.info(f"Found {len(missing_files)} files needing metadata")

        if dry_run:
            logger.info("🔍 DRY RUN - No files will be modified")
            for file_info in missing_files:
                logger.info(f"Would fix: {file_info['filename']}")
            return {'total': len(missing_files), 'fixed': 0, 'failed': 0}

        # Fix each file
        stats = {'total': len(missing_files), 'fixed': 0, 'failed': 0}

        for i, file_info in enumerate(missing_files, 1):
            logger.info(f"Processing {i}/{len(missing_files)}: {file_info['filename']}")

            success = await self.fix_single_metadata(file_info, force)
            if success:
                stats['fixed'] += 1
            else:
                stats['failed'] += 1

        return stats

    def generate_report(self) -> Dict[str, Any]:
        """Generate a report of cache metadata status"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'cache_dir': str(self.cache_dir.absolute()),
            'subdirectories': [],
            'totals': {'audio_files': 0, 'metadata_files': 0, 'missing_metadata': 0}
        }

        if not self.cache_dir.exists():
            return report

        for subdir in self.cache_dir.iterdir():
            if not subdir.is_dir():
                continue

            # Count files
            audio_extensions = ['*.m4a', '*.wav', '*.mp3', '*.flac']
            audio_files = []
            for ext in audio_extensions:
                audio_files.extend(subdir.glob(ext))

            metadata_files = list(subdir.glob("*_metadata.json"))

            subdir_info = {
                'name': subdir.name,
                'audio_files': len(audio_files),
                'metadata_files': len(metadata_files),
                'missing_metadata': len(audio_files) - len(metadata_files)
            }

            report['subdirectories'].append(subdir_info)
            report['totals']['audio_files'] += subdir_info['audio_files']
            report['totals']['metadata_files'] += subdir_info['metadata_files']
            report['totals']['missing_metadata'] += subdir_info['missing_metadata']

        return report

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Fix missing metadata in cache')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    parser.add_argument('--force', action='store_true', help='Overwrite existing metadata files')
    parser.add_argument('--report-only', action='store_true', help='Only generate a report')
    parser.add_argument('--config', default='configs/default_config.yaml', help='Config file path')

    args = parser.parse_args()

    # Change to script directory
    os.chdir(Path(__file__).parent)

    print("🔧 Cache Metadata Fixer")
    print("=" * 50)

    # Initialize fixer
    fixer = CacheMetadataFixer(args.config)

    # Generate report
    print("\n📊 Generating cache report...")
    report = fixer.generate_report()

    print(f"Cache Directory: {report['cache_dir']}")
    print(f"Total Audio Files: {report['totals']['audio_files']}")
    print(f"Total Metadata Files: {report['totals']['metadata_files']}")
    print(f"Missing Metadata: {report['totals']['missing_metadata']}")

    if report['totals']['audio_files'] > 0:
        coverage = (report['totals']['metadata_files'] / report['totals']['audio_files']) * 100
        print(f"Metadata Coverage: {coverage:.1f}%")

    # Show per-directory breakdown
    if report['subdirectories']:
        print("\n📁 Per-directory breakdown:")
        for subdir in report['subdirectories']:
            if subdir['audio_files'] > 0:
                print(f"  {subdir['name']}: {subdir['metadata_files']}/{subdir['audio_files']} files have metadata")

    if args.report_only:
        return

    if report['totals']['missing_metadata'] == 0:
        print("\n🎉 All cache files already have metadata!")
        return

    # Initialize data manager if we need to fix files
    print(f"\n🔧 Initializing data manager...")
    success = await fixer.initialize()
    if not success:
        print("❌ Failed to initialize. Some fixes may be limited.")

    # Fix metadata
    print(f"\n🔨 Fixing missing metadata...")
    stats = await fixer.fix_all_metadata(dry_run=args.dry_run, force=args.force)

    # Results
    print("\n" + "=" * 50)
    print("📋 Results:")
    print(f"  Total files processed: {stats['total']}")
    print(f"  Successfully fixed: {stats['fixed']}")
    print(f"  Failed: {stats['failed']}")

    if stats['fixed'] > 0:
        print(f"\n✅ Successfully added metadata to {stats['fixed']} files!")

    if stats['failed'] > 0:
        print(f"\n⚠️  Failed to process {stats['failed']} files. Check logs for details.")

if __name__ == "__main__":
    asyncio.run(main())
