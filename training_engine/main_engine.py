#!/usr/bin/env python3
"""
Main Orchestrator for the Superior Twi Speech Training Engine
============================================================

This script provides a centralized, interactive command-line interface (CLI)
to manage the entire speech recognition pipeline, from data recording to model inference.
"""

import os
import sys
import json
import yaml
import torch
import argparse
from pathlib import Path
from datetime import datetime

# --- System Path Setup ---
# Ensures that the script can find the necessary source files
try:
    current_dir = Path(__file__).parent
    src_dir = current_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
except NameError:
    # This handles the case where the script is run in an environment where __file__ is not defined
    current_dir = Path.cwd()
    src_dir = current_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

# --- Imports ---
# These are imported after the path setup to ensure they are found
try:
    from src.models.advanced_speech_model import SuperiorTwiSpeechModel
    from src.features.advanced_feature_extractor import SuperiorFeatureExtractor
    from src.features.dataset_utils import DatasetManager
    from src.trainers.superior_trainer import SuperiorTrainer
    from src.data.audio_recorder import interactive_recording_session
    from src.data.data_manager import TwiDataManager
    from src.config.config_loader import load_config, ConfigLoader
    from src.utils.env_loader import load_env
    from src.utils.async_utils import run_async
    import asyncio
except ImportError as e:
    print(f"❌ Error: Could not import necessary modules.")
    print(f"   Please ensure you are running this script from the 'training_engine' directory.")
    print(f"   Details: {e}")
    sys.exit(1)


class TwiSpeechPipeline:
    """
    An orchestrator class to manage the Twi speech recognition pipeline.
    """

    def __init__(self, config_path: str = None):
        """
        Initializes the pipeline manager.

        Args:
            config_path (str, optional): Path to a custom configuration file (JSON or YAML).
        """
        # Load environment variables first
        load_env()

        self.config = self._load_config(config_path)
        self.pipeline_state = {}
        self.check_pipeline_state()

    def _load_config(self, config_path: str) -> dict:
        """Loads configuration from a file or uses defaults."""
        # Try to load from config file if provided
        if config_path and os.path.exists(config_path):
            print(f"📋 Loading configuration from {config_path}...")
            try:
                # Use ConfigLoader which supports both JSON and YAML
                config = load_config(config_path, use_defaults=True)
                print(f"✓ Successfully loaded configuration from {config_path}")
            except Exception as e:
                print(f"⚠️ Failed to load config from {config_path}: {e}")
                print("📋 Falling back to default configuration.")
                config = self._get_default_config()
        else:
            # Try to find default config file
            default_configs = [
                Path(__file__).parent / "configs" / "default_config.yaml",
                Path(__file__).parent / "configs" / "config.yaml",
                Path(__file__).parent / "config.json",
            ]

            config_loaded = False
            for default_path in default_configs:
                if default_path.exists():
                    print(f"📋 Loading default configuration from {default_path}...")
                    try:
                        config = load_config(default_path, use_defaults=True)
                        print(f"✓ Successfully loaded configuration from {default_path}")
                        config_loaded = True
                        break
                    except Exception as e:
                        print(f"⚠️ Failed to load config from {default_path}: {e}")

            if not config_loaded:
                print("📋 Using built-in default configuration.")
                config = self._get_default_config()

        # Ensure essential directories exist
        essential_dirs = ['recordings_dir', 'features_dir', 'models_dir', 'analysis_dir',
                         'cache_dir', 'output_dir']
        for key in essential_dirs:
            if key in config:
                os.makedirs(config[key], exist_ok=True)

        # Also check nested directories
        if 'training' in config and 'output_dir' in config['training']:
            os.makedirs(config['training']['output_dir'], exist_ok=True)

        return config

    def _get_default_config(self) -> dict:
        """Returns the default configuration."""
        return {
            'recordings_dir': 'data/recordings',
            'features_dir': 'data/processed_superior',
            'models_dir': 'models/superior',
            'analysis_dir': 'analysis/superior',
            'cache_dir': 'data/cache',
            'batch_size': 32,
            'learning_rate': 0.001,
            'num_epochs': 100,
            'num_workers': 4,
            'script_file': 'script_actual.ts',
            'audio': {
                'sample_rate': 16000,
                'channels': 1,
                'format': 'wav'
            },
            'text': {
                'max_length': 500,
                'min_length': 1,
                'language': 'tw'
            },
            'model': {
                'type': 'wav2vec2',
                'pretrained_model': 'facebook/wav2vec2-base'
            },
            'training': {
                'batch_size': 8,
                'learning_rate': 1e-4,
                'num_epochs': 10,
                'output_dir': 'outputs/models'
            }
        }

    def check_pipeline_state(self):
        """Checks the current state of the pipeline and updates the state dictionary."""
        # Handle both old format (direct keys) and new format (nested under directories)
        recordings_dir = self.config.get('recordings_dir', 'data/recordings')
        features_dir = self.config.get('features_dir', 'data/processed_superior')
        models_dir = self.config.get('models_dir', 'models/superior')

        # Create directories if they don't exist
        os.makedirs(recordings_dir, exist_ok=True)
        os.makedirs(features_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

        self.pipeline_state['recordings_exist'] = any(f.endswith(('.wav', '.m4a', '.mp3')) for r, d, f_list in os.walk(recordings_dir) for f in f_list)

        # Check for features in multiple possible locations
        features_exist = (
            os.path.exists(os.path.join(features_dir, 'features.npy')) or
            os.path.exists('data/features') and any(f.endswith('.pkl') for f in os.listdir('data/features') if os.path.isfile(os.path.join('data/features', f))) or
            os.path.exists('data/processed') and any(f.endswith('.pkl') for f in os.listdir('data/processed') if os.path.isfile(os.path.join('data/processed', f)))
        )
        self.pipeline_state['features_exist'] = features_exist

        self.pipeline_state['model_trained'] = any(f.endswith('.pt') for f in os.listdir(models_dir) if os.path.isfile(os.path.join(models_dir, f)))

        # Check for processed audio
        self.pipeline_state['audio_processed'] = any('_aug' in f or '_processed' in f for r, d, f_list in os.walk(recordings_dir) for f in f_list)

        # Check for training-ready data
        self.pipeline_state['training_data_ready'] = os.path.exists('data/processed') and any(f.startswith(('train_', 'validation_', 'test_')) for f in os.listdir('data/processed') if f.endswith('.pkl'))

        # Check for organized data (metadata files in recordings directory)
        self.pipeline_state['data_organized'] = any(f.endswith('_metadata.json') for r, d, f_list in os.walk(recordings_dir) for f in f_list)

    def show_status(self):
        """Displays the current status of the pipeline."""
        self.check_pipeline_state()
        print("\n" + "="*60)
        print("📊 Twi Speech Engine Pipeline Status")
        print("="*60)
        print(f"  [1] Audio Data Available:     {'✅ Yes' if self.pipeline_state['recordings_exist'] else '❌ No'}")
        print(f"  [2] Cloud Data Downloaded:    {'✅ Yes' if self.pipeline_state.get('cloud_data_downloaded') else '❌ No'}")
        print(f"  [3] Data Organized:           {'✅ Yes' if self.pipeline_state.get('data_organized') else '❌ No'}")
        print(f"  [4] Audio Processed:          {'✅ Yes' if self.pipeline_state.get('audio_processed') else '❌ No'}")
        print(f"  [5] Features Extracted:       {'✅ Yes' if self.pipeline_state['features_exist'] else '❌ No'}")
        print(f"  [6] Training Data Ready:      {'✅ Yes' if self.pipeline_state.get('training_data_ready') else '❌ No'}")
        print(f"  [7] Model Trained:            {'✅ Yes' if self.pipeline_state['model_trained'] else '❌ No'}")
        print("="*60)
        print("Next recommended step is marked with '-->'.")
        if not self.pipeline_state['recordings_exist'] and not self.pipeline_state.get('cloud_data_downloaded'):
            print("--> Download cloud data or record new audio to get started.")
        elif self.pipeline_state.get('cloud_data_downloaded') and not self.pipeline_state.get('data_organized'):
            print("--> Organize downloaded data from cache to recordings directory.")
        elif self.pipeline_state['recordings_exist'] and not self.pipeline_state.get('audio_processed'):
            print("--> Process and augment your audio data.")
        elif self.pipeline_state.get('audio_processed') and not self.pipeline_state['features_exist']:
            print("--> Extract features from processed audio.")
        elif self.pipeline_state['features_exist'] and not self.pipeline_state.get('training_data_ready'):
            print("--> Run complete processing pipeline to prepare training data.")
        elif self.pipeline_state.get('training_data_ready') and not self.pipeline_state['model_trained']:
            print("--> Train your model with the prepared data.")
        elif not self.pipeline_state['features_exist']:
            print("--> Run feature extraction to prepare data for training.")
        elif not self.pipeline_state['model_trained']:
            print("--> Train a model with the extracted features.")
        else:
            print("--> Your pipeline is ready! You can test your model or run inference.")

    def run_recording(self):
        """Launches the interactive audio recording session."""
        print("\n" + "="*50)
        print("🎤 Launching Audio Recording Session")
        print("="*50)
        interactive_recording_session(self.config)
        self.check_pipeline_state()

    def run_cloud_download(self):
        """Downloads data from R2/MongoDB and prepares it for use."""
        print("\n" + "="*50)
        print("☁️ Downloading Cloud Data")
        print("="*50)

        print("This will:")
        print("1. 📥 Download audio files from R2 storage to cache")
        print("2. 📊 Fetch metadata from MongoDB")
        print("3. 🗂️ Organize files into recordings directory")
        print("4. 📋 Create metadata files and data index")

        try:
            manager = TwiDataManager(self.config)
            # This triggers download + organization
            print("\nINFO: Attempting to fetch data from backend. This may take a while...")
            samples = manager.fetch_data_from_backend_sync()

            if samples:
                print(f"\n✅ Successfully downloaded and organized {len(samples)} samples!")

                # Show where files are located
                recordings_dir = self.config.get('recordings_dir', 'data/recordings')
                cache_dir = self.config.get('cache_dir', 'data/cache')

                print(f"\n📂 File Locations:")
                print(f"   Cache (temporary): {cache_dir}")
                print(f"   Recordings (organized): {recordings_dir}")

                # Mark states as completed
                self.pipeline_state['cloud_data_downloaded'] = True
                self.pipeline_state['data_organized'] = True
                self.pipeline_state['recordings_exist'] = True

                print(f"\n📋 Next Steps:")
                print(f"   - Files are ready for processing (option 4)")
                print(f"   - Or run complete pipeline (option 6)")
            else:
                print("\n⚠️ No data downloaded. This could be because:")
                print("   - No data is available in the backend")
                print("   - Cloud credentials are not configured in .env file")
                print("   - Connection issues with MongoDB/R2")
                print(f"\n🔧 To configure:")
                print(f"   1. Set MONGODB_URI in .env file")
                print(f"   2. Set R2 credentials (R2_BUCKET_NAME, R2_ACCESS_KEY_ID, etc.)")
        except Exception as e:
            print(f"\n❌ Cloud data download failed: {e}")
            print("Check your .env file configuration and network connection.")
            import traceback
            traceback.print_exc()

    def run_feature_extraction(self):
        """Processes raw audio recordings into features for training."""
        if not self.pipeline_state['recordings_exist'] and not self.pipeline_state.get('cloud_data_downloaded'):
            print("❌ No local or cloud data found. Please record or download data first.")
            return

        print("\n" + "="*50)
        print("✨ Running Feature Extraction")
        print("="*50)
        try:
            extractor = SuperiorFeatureExtractor(output_dir=self.config['features_dir'])
            extractor.extract_features_from_recordings(self.config['recordings_dir'])
            print("\n✅ Feature extraction completed successfully!")
            self.check_pipeline_state()
        except Exception as e:
            print(f"\n❌ Feature extraction failed: {e}")

    def run_training(self):
        """Starts the model training process."""
        if not self.pipeline_state['features_exist']:
            print("❌ Features not found. Please run feature extraction first.")
            return

        print("\n" + "="*50)
        print("🧠 Starting Model Training")
        print("="*50)
        try:
            # 1. Dataset Preparation
            dataset_manager = DatasetManager(self.config)
            features_dir = self.config.get('features_dir', 'data/processed_superior')
            full_dataset = dataset_manager.load_dataset_from_directory(features_dir)
            train_ds, val_ds, test_ds = dataset_manager.create_stratified_splits(full_dataset)
            batch_size = self.config.get('batch_size', 32)
            if 'training' in self.config and 'batch_size' in self.config['training']:
                batch_size = self.config['training']['batch_size']

            num_workers = self.config.get('num_workers', 4)
            if 'training' in self.config and 'dataloader_num_workers' in self.config['training']:
                num_workers = self.config['training']['dataloader_num_workers']

            dataloaders = dataset_manager.create_dataloaders(
                train_dataset=train_ds, val_dataset=val_ds, test_dataset=test_ds,
                batch_size=batch_size, num_workers=num_workers
            )

            # 2. Model Creation
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            feature_info_path = os.path.join(features_dir, 'feature_info.json')
            with open(feature_info_path, 'r') as f:
                feature_info = json.load(f)

            model = SuperiorTwiSpeechModel(
                input_dim=feature_info['total'],
                num_classes=full_dataset.get_num_classes()
            )

            # 3. Training
            trainer = SuperiorTrainer(model, device, config=self.config)
            trainer.train(dataloaders['train'], dataloaders['val'], num_epochs=self.config['num_epochs'])

            print("\n✅ Model training completed successfully!")
            self.check_pipeline_state()
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()

    def run_testing(self):
        """Evaluates a trained model on the test set."""
        if not self.pipeline_state['model_trained']:
            print("❌ No trained model found. Please train a model first.")
            return

        print("\n" + "="*50)
        print("🧪 Evaluating Model Performance")
        print("="*50)
        # This functionality is part of the trainer's final steps.
        # For a standalone test, we would load the best model and run it on the test set.
        print("INFO: The training process automatically runs evaluation on the validation set.")
        print("A full, standalone test on the held-out test set would require loading the best checkpoint and running a final validation loop.")
        print("This can be added as a future enhancement.")

    def run_audio_processing(self):
        """Process and augment audio files without feature extraction"""
        print("\n" + "="*50)
        print("🎵 Processing and Augmenting Audio")
        print("="*50)

        try:
            # Check if we have audio data
            recordings_dir = self.config.get('recordings_dir', 'data/recordings')
            if not os.path.exists(recordings_dir) or not any(f.endswith('.wav') or f.endswith('.m4a') for r, d, files in os.walk(recordings_dir) for f in files):
                print("❌ No audio files found. Please record or download audio first.")
                return

            # Initialize data manager
            from src.data.data_manager import TwiDataManager
            manager = TwiDataManager(self.config)

            # Load samples from recordings directory
            samples = manager.load_local_data()
            if not samples:
                print("❌ No valid audio samples found.")
                return

            print(f"📁 Found {len(samples)} audio samples")

            # Ask user for augmentation preferences
            print("\nAugmentation Options:")
            print("1. Process only (normalize, trim silence)")
            print("2. Process + 2x augmentation")
            print("3. Process + 3x augmentation")
            print("4. Process + 5x augmentation")

            aug_choice = input("Choose augmentation level [1-4]: ").strip()

            augmentation_map = {'1': 1, '2': 2, '3': 3, '4': 5}
            augmentation_factor = augmentation_map.get(aug_choice, 1)
            apply_augmentation = augmentation_factor > 1

            # Process audio
            if apply_augmentation:
                print(f"\n🔄 Processing audio with {augmentation_factor}x augmentation...")
                processed_samples = manager.augment_dataset(samples, augmentation_factor=augmentation_factor)
            else:
                print("\n🔄 Processing audio without augmentation...")
                processed_samples = manager.process_audio_batch(samples, apply_augmentation=False)

            print(f"✅ Audio processing completed!")
            print(f"   Original samples: {len(samples)}")
            print(f"   Processed samples: {len(processed_samples)}")

            # Update pipeline state
            self.pipeline_state['audio_processed'] = True

        except Exception as e:
            print(f"❌ Audio processing failed: {e}")
            import traceback
            traceback.print_exc()

    def run_feature_extraction(self):
        """Extract features from processed audio"""
        print("\n" + "="*50)
        print("🧮 Extracting Features from Audio")
        print("="*50)

        try:
            # Initialize data manager
            from src.data.data_manager import TwiDataManager
            manager = TwiDataManager(self.config)

            # Load samples
            samples = manager.load_local_data()
            if not samples:
                print("❌ No audio samples found. Please process audio first.")
                return

            print(f"📁 Found {len(samples)} audio samples")

            # Ask user for feature type
            print("\nFeature Extraction Options:")
            print("1. MFCC (Traditional audio features)")
            print("2. Mel Spectrogram (Deep learning spectrograms)")
            print("3. Wav2Vec2 (Raw audio for transformers)")

            feature_choice = input("Choose feature type [1-3]: ").strip()

            feature_map = {'1': 'mfcc', '2': 'mel_spectrogram', '3': 'wav2vec2'}
            feature_type = feature_map.get(feature_choice, 'wav2vec2')

            print(f"\n🔄 Extracting {feature_type} features...")

            # Extract features
            features_data = manager.extract_features_batch(samples, feature_type=feature_type)

            if features_data:
                print(f"✅ Feature extraction completed!")
                print(f"   Extracted features from {len(features_data)} samples")
                print(f"   Feature type: {feature_type}")

                # Save features
                output_dir = "data/features"
                os.makedirs(output_dir, exist_ok=True)

                import pickle
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                feature_file = f"{output_dir}/features_{feature_type}_{timestamp}.pkl"

                with open(feature_file, 'wb') as f:
                    pickle.dump(features_data, f)

                print(f"   Features saved to: {feature_file}")

                # Update pipeline state
                self.pipeline_state['features_exist'] = True

            else:
                print("❌ No features extracted.")

        except Exception as e:
            print(f"❌ Feature extraction failed: {e}")
            import traceback
            traceback.print_exc()

    def run_complete_processing_pipeline(self):
        """Run the complete audio processing and feature extraction pipeline"""
        print("\n" + "="*60)
        print("🚀 Complete Audio Processing Pipeline")
        print("="*60)

        try:
            # Initialize data manager
            from src.data.data_manager import TwiDataManager
            manager = TwiDataManager(self.config)

            print("📋 Pipeline Configuration:")
            print("This will run the complete pipeline:")
            print("1. Load/download audio data")
            print("2. Process and augment audio")
            print("3. Prepare training datasets")
            print("4. Create train/validation/test splits")
            print("5. Save training-ready data")

            # Ask for configuration
            print("\n⚙️ Configuration Options:")

            # Training type
            print("\nTraining Type:")
            print("1. Speech-to-Text (Twi Audio → Twi Text)")
            print("2. Speech Translation (Twi Audio → English Text)")
            print("3. Multilingual (Twi Audio → Both Twi/English)")
            print("4. Cross-lingual (Twi Audio + Text → English)")
            training_choice = input("Choose training type [1-4]: ").strip()
            training_map = {
                '1': 'speech_to_text',
                '2': 'translation',
                '3': 'multilingual',
                '4': 'cross_lingual'
            }
            training_type = training_map.get(training_choice, 'speech_to_text')

            # Feature type
            print("\nFeature Type:")
            print("1. MFCC (Traditional)")
            print("2. Mel Spectrogram (CNN-friendly)")
            print("3. Wav2Vec2 (Transformer-friendly)")
            feature_choice = input("Choose feature type [1-3]: ").strip()
            feature_map = {'1': 'mfcc', '2': 'mel_spectrogram', '3': 'wav2vec2'}
            feature_type = feature_map.get(feature_choice, 'wav2vec2')

            # Augmentation
            print("\nAugmentation:")
            print("1. No augmentation")
            print("2. 2x augmentation")
            print("3. 3x augmentation")
            print("4. 5x augmentation")
            aug_choice = input("Choose augmentation level [1-4]: ").strip()
            augmentation_map = {'1': 1, '2': 2, '3': 3, '4': 5}
            augmentation_factor = augmentation_map.get(aug_choice, 2)
            apply_augmentation = augmentation_factor > 1

            # Parallel processing
            max_workers = min(4, os.cpu_count() or 1)
            worker_input = input(f"Number of parallel workers [1-{max_workers}] (default {max_workers}): ").strip()
            if worker_input.isdigit() and 1 <= int(worker_input) <= max_workers:
                max_workers = int(worker_input)

            # Additional config for multilingual training
            dataset_kwargs = {}
            if training_type == 'multilingual':
                print("\nMultilingual Configuration:")
                ratio_input = input("Task mixing ratio (0.0=all Twi, 1.0=all English) [default 0.5]: ").strip()
                try:
                    ratio = float(ratio_input) if ratio_input else 0.5
                    dataset_kwargs['task_mixing_ratio'] = max(0.0, min(1.0, ratio))
                except ValueError:
                    dataset_kwargs['task_mixing_ratio'] = 0.5

            print(f"\n🔄 Starting pipeline with:")
            print(f"   Training type: {training_type}")
            print(f"   Feature type: {feature_type}")
            print(f"   Augmentation: {augmentation_factor}x" if apply_augmentation else "   Augmentation: None")
            print(f"   Workers: {max_workers}")
            if dataset_kwargs:
                print(f"   Additional config: {dataset_kwargs}")

            # Run complete pipeline
            training_data = manager.prepare_complete_training_pipeline(
                feature_type=feature_type,
                training_type=training_type,
                apply_augmentation=apply_augmentation,
                augmentation_factor=augmentation_factor,
                save_processed_data=True,
                max_workers=max_workers,
                **dataset_kwargs
            )

            print("\n✅ Complete pipeline finished successfully!")
            print(f"\nResults:")
            print(f"   Training type: {training_data['training_type']}")
            print(f"   Total samples: {training_data['total_samples']}")
            print(f"   Train samples: {len(training_data['train_samples'])}")
            print(f"   Validation samples: {len(training_data['validation_samples'])}")
            print(f"   Test samples: {len(training_data['test_samples'])}")
            print(f"   Feature type: {training_data['feature_type']}")
            print(f"   Data saved to: {training_data.get('saved_to', 'Not saved')}")

            # Show training scenario info
            scenario_info = training_data['metadata'].get('training_scenarios', {})
            if scenario_info:
                print(f"\nTraining Scenario:")
                print(f"   Input: {scenario_info.get('input', 'N/A')}")
                print(f"   Output: {scenario_info.get('output', 'N/A')}")
                print(f"   Description: {scenario_info.get('description', 'N/A')}")

            # Update pipeline state
            self.pipeline_state['audio_processed'] = True
            self.pipeline_state['features_exist'] = True
            self.pipeline_state['training_data_ready'] = True

        except Exception as e:
            print(f"❌ Complete pipeline failed: {e}")
            import traceback
            traceback.print_exc()

    def run_organize_data(self):
        """Organize cached data into recordings directory with metadata"""
        print("\n" + "="*50)
        print("🗂️ Organizing Data from Cache to Recordings")
        print("="*50)

        try:
            # Initialize data manager
            from src.data.data_manager import TwiDataManager
            manager = TwiDataManager(self.config)

            cache_dir = Path(self.config.get('cache_dir', 'data/cache'))
            recordings_dir = Path(self.config.get('recordings_dir', 'data/recordings'))

            # Check if there are cached files to organize
            if not cache_dir.exists():
                print("❌ No cache directory found. Download cloud data first.")
                return

            cached_files = list(cache_dir.rglob('*.wav')) + list(cache_dir.rglob('*.m4a'))
            if not cached_files:
                print("❌ No audio files found in cache directory.")
                print(f"   Cache directory: {cache_dir.absolute()}")
                print("   Try downloading cloud data first (option 2).")
                return

            print(f"📁 Found {len(cached_files)} cached audio files")
            print(f"   Cache: {cache_dir.absolute()}")
            print(f"   Target: {recordings_dir.absolute()}")

            # Load samples from cache (this will create AudioSample objects from cached files)
            print("\n🔄 Loading cached samples...")
            samples = []

            # Create AudioSample objects from cached files
            from src.data.structures import AudioSample
            for cached_file in cached_files:
                # Extract info from filename/path
                filename = cached_file.name
                # Try to extract speaker and prompt info from filename patterns
                if '_' in filename:
                    parts = filename.replace('.wav', '').replace('.m4a', '').split('_')
                    if len(parts) >= 3:
                        speaker_id = parts[0] if parts[0].startswith('TWI_') else 'unknown_speaker'
                        prompt_id = '_'.join(parts[1:3]) if len(parts) >= 3 else 'unknown_prompt'
                    else:
                        speaker_id = 'unknown_speaker'
                        prompt_id = 'unknown_prompt'
                else:
                    speaker_id = 'unknown_speaker'
                    prompt_id = filename.replace('.wav', '').replace('.m4a', '')

                sample = AudioSample(
                    audio_path=str(cached_file),
                    transcription="Unknown transcription",
                    duration=0.0,  # Will be updated during organization
                    speaker_id=speaker_id,
                    dialect='unknown',
                    prompt_id=prompt_id,
                    prompt_text="Unknown prompt",
                    meaning="Unknown meaning",
                    original_filename=filename
                )
                samples.append(sample)

            print(f"✓ Created {len(samples)} sample objects")

            # Organize the data
            print("\n🗂️ Organizing data into recordings directory...")
            organized_samples = manager.organize_downloaded_data(samples)

            if organized_samples:
                print(f"\n✅ Successfully organized {len(organized_samples)} samples!")

                # Show organization results
                speakers = set(s.speaker_id for s in organized_samples)
                print(f"   Speakers: {len(speakers)} ({', '.join(speakers)})")
                print(f"   Files organized to: {recordings_dir.absolute()}")

                # Update pipeline state
                self.pipeline_state['data_organized'] = True
                self.pipeline_state['recordings_exist'] = True

                # Show next steps
                print(f"\n📋 Next Steps:")
                print(f"   1. Files are now in {recordings_dir}")
                print(f"   2. Each audio file has a metadata JSON file")
                print(f"   3. Ready for audio processing (option 4)")
            else:
                print("❌ No samples were organized")

        except Exception as e:
            print(f"❌ Data organization failed: {e}")
            import traceback
            traceback.print_exc()

    def run_inference(self):
        """Runs inference on a single audio file using a trained model."""
        if not self.pipeline_state['model_trained']:
            print("❌ No trained model found. Please train a model first.")
            return

        print("\n" + "="*50)
        print("🗣️ Running Inference")
        print("="*50)
        # Placeholder for inference logic
        print("Inference mode is not yet fully implemented in this orchestrator but can be run via example scripts.")

    def run_full_pipeline(self):
        """Runs the entire pipeline automatically."""
        print("\n" + "="*60)
        print("🚀🚀🚀 RUNNING FULL AUTOMATED PIPELINE 🚀🚀🚀")
        print("="*60)

        # Step 1: Check for data, if not, ask to record
        self.check_pipeline_state()
        if not self.pipeline_state['recordings_exist']:
            print("\nPipeline stopped: No audio recordings found.")
            self.run_recording()

        # Re-check state after recording
        self.check_pipeline_state()
        if not self.pipeline_state['recordings_exist']:
            print("\nPipeline aborted: Still no recordings available after prompt.")
            return

        # Step 2: Check if data needs organization
        if not self.pipeline_state.get('data_organized'):
            print("\n🗂️ Organizing data from cache to recordings directory...")

            # Initialize data manager
            from src.data.data_manager import TwiDataManager
            manager = TwiDataManager(self.config)

            try:
                # Load samples from cache and organize them
                cache_dir = Path(self.config.get('cache_dir', 'data/cache'))
                if cache_dir.exists():
                    cached_files = list(cache_dir.rglob('*.wav')) + list(cache_dir.rglob('*.m4a'))
                    if cached_files:
                        print(f"   Found {len(cached_files)} cached files to organize")
                        # This will be handled by the normal data flow
                    else:
                        print("   No cached files found - data already organized")

            except Exception as e:
                print(f"\n⚠️ Organization check failed: {e}")

        # Step 3: Complete Audio Processing Pipeline
        if not self.pipeline_state.get('training_data_ready'):
            print("\n🎵 Running complete audio processing pipeline...")
            print("This includes: audio processing, augmentation, feature extraction, and data splitting")

            # Initialize data manager
            from src.data.data_manager import TwiDataManager
            manager = TwiDataManager(self.config)

            try:
                # Run complete pipeline with default settings for automation
                training_data = manager.prepare_complete_training_pipeline(
                    feature_type="wav2vec2",  # Default to wav2vec2 for best results
                    training_type="speech_to_text",  # Default to standard ASR
                    apply_augmentation=True,
                    augmentation_factor=2,    # 2x augmentation for automation
                    save_processed_data=True,
                    max_workers=2            # Conservative for stability
                )

                print("✅ Complete processing pipeline finished successfully!")
                print(f"   Total samples: {training_data['total_samples']}")
                print(f"   Training data saved to: {training_data.get('saved_to', 'data/processed')}")

            except Exception as e:
                print(f"\n❌ Processing pipeline failed: {e}")
                print("Pipeline aborted.")
                return
        else:
            print("\n✅ Training data already prepared, skipping processing.")

        # Re-check state
        self.check_pipeline_state()
        if not self.pipeline_state.get('training_data_ready'):
            print("\nPipeline aborted: Data processing failed.")
            return

        # Step 4: Training
        if not self.pipeline_state['model_trained']:
            print("\n🤖 Starting model training...")
            self.run_training()
        else:
            print("\n✅ Model already trained.")

    def main_menu(self):
        """Displays the main interactive menu."""
        while True:
            self.show_status()
            print("\nPlease choose an option:")
            print("  1. Record New Audio Data")
            print("  2. Download Cloud Data (R2/MongoDB)")
            print("  3. Organize Data (Cache → Recordings)")
            print("  4. Process & Augment Audio")
            print("  5. Extract Features")
            print("  6. Complete Processing Pipeline")
            print("  7. Train a New Model")
            print("  8. Test a Model (Info)")
            print("  --------------------")
            print("  9. Run Full Pipeline (Download -> Process -> Train)")
            print("  10. Exit")

            try:
                choice = input("Enter your choice [1-10]: ").strip()
                if choice == '1':
                    self.run_recording()
                elif choice == '2':
                    self.run_cloud_download()
                elif choice == '3':
                    self.run_organize_data()
                elif choice == '4':
                    self.run_audio_processing()
                elif choice == '5':
                    self.run_feature_extraction()
                elif choice == '6':
                    self.run_complete_processing_pipeline()
                elif choice == '7':
                    self.run_training()
                elif choice == '8':
                    self.run_testing()
                elif choice == '9':
                    self.run_full_pipeline()
                elif choice == '10':
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please try again.")
            except KeyboardInterrupt:
                print("\n\n👋 Exiting...")
                break
            except Exception as e:
                print(f"\nAn unexpected error occurred: {e}")


def main():
    """Main function to run the pipeline orchestrator."""
    parser = argparse.ArgumentParser(description="Main Orchestrator for the Twi Speech Training Engine")
    parser.add_argument('--mode', type=str, choices=['interactive', 'record', 'download', 'features', 'train', 'full'],
                        help="Run a specific mode directly without the interactive menu.")
    parser.add_argument('--config', type=str, help="Path to a custom configuration file (JSON or YAML).")
    parser.add_argument('--env', type=str, help="Path to .env file for environment variables.")

    args = parser.parse_args()

    # Load environment variables if specified
    if args.env:
        load_env(args.env)

    pipeline = TwiSpeechPipeline(config_path=args.config)

    if args.mode:
        if args.mode == 'record':
            pipeline.run_recording()
        elif args.mode == 'download':
            pipeline.run_cloud_download()
        elif args.mode == 'features':
            pipeline.run_feature_extraction()
        elif args.mode == 'train':
            pipeline.run_training()
        elif args.mode == 'full':
            pipeline.run_full_pipeline()
        else: # interactive
            pipeline.main_menu()
    else:
        pipeline.main_menu()


if __name__ == "__main__":
    main()
