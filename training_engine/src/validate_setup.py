import os
import logging
import pandas as pd
import torch
import torchaudio
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Import our modules
try:
    from .data_loader import load_and_prepare_data, get_mongo_client, get_s3_client, load_backend_env, BACKEND_ENV_PATH
    from .model import ECommerceCommandModel
    from .train_hf import load_and_prepare_dataset
except ImportError:
    from data_loader import load_and_prepare_data, get_mongo_client, get_s3_client, load_backend_env, BACKEND_ENV_PATH
    from model import ECommerceCommandModel
    from train_hf import load_and_prepare_dataset

# Configuration
AUDIO_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'e_commerce_audio')
METADATA_CSV = os.path.join(AUDIO_DIR, 'metadata.csv')
VALIDATION_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'validation_results')
os.makedirs(VALIDATION_OUTPUT_DIR, exist_ok=True)

class TwiSpeechValidator:
    """Comprehensive validator for the Twi speech training pipeline."""

    def __init__(self):
        self.results = {}
        self.issues = []
        self.recommendations = []
        # Load environment variables at initialization
        self._load_environment()

    def _load_environment(self):
        """Load environment variables from .env file."""
        try:
            # Try to load from the backend .env path
            load_backend_env(BACKEND_ENV_PATH)
            logging.info("✓ Environment variables loaded from .env file")
        except Exception as e:
            logging.warning(f"⚠ Could not load .env file: {e}")
            logging.info("Using system environment variables")

    def validate_environment(self):
        """Check if all required packages and environment variables are available."""
        logging.info("=== Validating Environment ===")

        required_packages = [
            'torch', 'torchaudio', 'transformers', 'datasets',
            'pandas', 'numpy', 'librosa', 'sounddevice',
            'audiomentations', 'wandb', 'sklearn'
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                logging.info(f"✓ {package} is available")
            except ImportError:
                missing_packages.append(package)
                logging.error(f"✗ {package} is missing")

        # Check environment variables
        required_env_vars = [
            'MONGODB_URI', 'MONGO_DB_NAME', 'CLOUDFLARE_ACCOUNT_ID',
            'CLOUDFLARE_ACCESS_KEY_ID', 'CLOUDFLARE_SECRET_ACCESS_KEY',
            'R2_BUCKET_NAME'
        ]

        missing_env_vars = []
        for var in required_env_vars:
            if not os.getenv(var):
                missing_env_vars.append(var)
                logging.warning(f"⚠ Environment variable {var} is not set")
            else:
                logging.info(f"✓ {var} is set")

        # Check CUDA availability
        if torch.cuda.is_available():
            logging.info(f"✓ CUDA is available: {torch.cuda.get_device_name()}")
            logging.info(f"✓ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logging.warning("⚠ CUDA is not available, training will use CPU")

        self.results['environment'] = {
            'missing_packages': missing_packages,
            'missing_env_vars': missing_env_vars,
            'cuda_available': torch.cuda.is_available()
        }

        return len(missing_packages) == 0

    def validate_data_connectivity(self):
        """Test connections to MongoDB and Cloudflare R2."""
        logging.info("=== Validating Data Connectivity ===")

        # Test MongoDB connection
        mongo_client = get_mongo_client()
        mongodb_ok = mongo_client is not None
        if mongodb_ok:
            logging.info("✓ MongoDB connection successful")
            try:
                mongo_client.close()
            except:
                pass
        else:
            logging.error("✗ MongoDB connection failed")
            self.issues.append("MongoDB connection failed - check credentials")

        # Test S3/R2 connection
        s3_client = get_s3_client()
        s3_ok = s3_client is not None
        if s3_ok:
            logging.info("✓ Cloudflare R2 connection successful")
        else:
            logging.error("✗ Cloudflare R2 connection failed")
            self.issues.append("Cloudflare R2 connection failed - check credentials")

        self.results['connectivity'] = {
            'mongodb': mongodb_ok,
            'cloudflare_r2': s3_ok
        }

        return mongodb_ok and s3_ok

    def validate_dataset(self, metadata_csv_path=None):
        """Analyze the dataset for quality and distribution."""
        logging.info("=== Validating Dataset ===")

        if metadata_csv_path is None:
            metadata_csv_path = METADATA_CSV

        if not os.path.exists(metadata_csv_path):
            logging.error(f"✗ Metadata CSV not found at {metadata_csv_path}")
            self.issues.append(f"Metadata CSV missing at {metadata_csv_path}")
            return False

        # Load metadata
        df = pd.read_csv(metadata_csv_path)
        logging.info(f"✓ Loaded metadata with {len(df)} records")

        # Check required columns
        required_columns = ['local_path', 'prompt_text']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"✗ Missing required columns: {missing_columns}")
            self.issues.append(f"Missing columns in metadata: {missing_columns}")
            return False

        # Analyze class distribution
        class_counts = df['prompt_text'].value_counts()
        logging.info(f"✓ Found {len(class_counts)} unique classes")

        # Identify problematic classes
        very_few_samples = class_counts[class_counts < 3]
        few_samples = class_counts[(class_counts >= 3) & (class_counts < 8)]

        if len(very_few_samples) > 0:
            logging.warning(f"⚠ {len(very_few_samples)} classes have < 3 samples")
            self.recommendations.append("Consider collecting more data for underrepresented classes")

        if len(few_samples) > 0:
            logging.info(f"ℹ {len(few_samples)} classes have 3-7 samples (will benefit from augmentation)")

        # Check audio files
        missing_files = 0
        corrupted_files = 0
        total_duration = 0

        for idx, row in df.head(min(50, len(df))).iterrows():  # Check first 50 files
            audio_path = row['local_path']

            # Handle relative paths by making them absolute
            if not os.path.isabs(audio_path):
                # If path starts with 'training_engine/', remove the prefix and make it relative to project root
                if audio_path.startswith('training_engine/'):
                    # Get the project root (parent of parent of src directory)
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                    # Remove 'training_engine/' prefix and join with project root
                    relative_path = audio_path[len('training_engine/'):]
                    audio_path = os.path.join(project_root, 'training_engine', relative_path)
                else:
                    # Otherwise, assume it's relative to current working directory
                    audio_path = os.path.abspath(audio_path)

            if not os.path.exists(audio_path):
                missing_files += 1
                logging.warning(f"Missing audio file: {audio_path}")
                continue

            try:
                waveform, sr = torchaudio.load(audio_path)
                duration = waveform.shape[1] / sr
                total_duration += duration

                if duration < 0.5:
                    logging.warning(f"⚠ Very short audio file: {audio_path} ({duration:.2f}s)")
                elif duration > 10:
                    logging.warning(f"⚠ Very long audio file: {audio_path} ({duration:.2f}s)")

            except Exception as e:
                corrupted_files += 1
                logging.error(f"✗ Corrupted audio file: {audio_path} - {e}")

        avg_duration = total_duration / max(1, len(df.head(50)) - missing_files - corrupted_files)

        logging.info(f"✓ Audio validation complete:")
        logging.info(f"  - Missing files: {missing_files}")
        logging.info(f"  - Corrupted files: {corrupted_files}")
        logging.info(f"  - Average duration: {avg_duration:.2f}s")

        # Save distribution plot
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        class_counts.head(20).plot(kind='bar')
        plt.title('Top 20 Classes Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plt.subplot(2, 1, 2)
        plt.hist(class_counts.values, bins=20, alpha=0.7)
        plt.title('Distribution of Samples per Class')
        plt.xlabel('Number of Samples')
        plt.ylabel('Number of Classes')
        plt.tight_layout()

        plot_path = os.path.join(VALIDATION_OUTPUT_DIR, 'class_distribution.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"✓ Class distribution plot saved to {plot_path}")

        self.results['dataset'] = {
            'total_samples': len(df),
            'num_classes': len(class_counts),
            'missing_files': missing_files,
            'corrupted_files': corrupted_files,
            'avg_duration': avg_duration,
            'very_few_samples_classes': len(very_few_samples),
            'few_samples_classes': len(few_samples),
            'class_distribution': class_counts.to_dict()
        }

        if missing_files > len(df) * 0.1:  # More than 10% missing
            self.issues.append(f"Too many missing audio files: {missing_files}/{len(df)}")

        if corrupted_files > 0:
            self.issues.append(f"Found {corrupted_files} corrupted audio files")

        return missing_files == 0 and corrupted_files == 0

    def validate_model_architecture(self):
        """Test model architecture with dummy data."""
        logging.info("=== Validating Model Architecture ===")

        try:
            # Test traditional CNN-RNN model
            n_mels = 80
            n_classes = 10
            batch_size = 4
            time_steps = 100

            model = ECommerceCommandModel(n_input_mels=n_mels, n_output_classes=n_classes)
            dummy_input = torch.randn(batch_size, 1, n_mels, time_steps)

            # Test forward pass
            with torch.no_grad():
                output = model(dummy_input)

            expected_shape = (batch_size, n_classes)
            if output.shape == expected_shape:
                logging.info(f"✓ Model architecture test passed: {output.shape}")
            else:
                logging.error(f"✗ Model output shape mismatch: expected {expected_shape}, got {output.shape}")
                self.issues.append("Model architecture produces wrong output shape")
                return False

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            logging.info(f"✓ Model parameters: {total_params:,} total, {trainable_params:,} trainable")

            # Test gradient flow
            model.train()
            criterion = torch.nn.CrossEntropyLoss()
            dummy_labels = torch.randint(0, n_classes, (batch_size,))

            output = model(dummy_input)
            loss = criterion(output, dummy_labels)
            loss.backward()

            # Check if gradients are computed
            has_gradients = any(p.grad is not None for p in model.parameters())
            if has_gradients:
                logging.info("✓ Gradient computation test passed")
            else:
                logging.error("✗ No gradients computed")
                self.issues.append("Model gradient computation failed")
                return False

            self.results['model'] = {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'output_shape_correct': True,
                'gradients_computed': has_gradients
            }

            return True

        except Exception as e:
            logging.error(f"✗ Model architecture test failed: {e}")
            self.issues.append(f"Model architecture error: {e}")
            return False

    def validate_audio_processing(self, metadata_csv_path=None):
        """Test audio preprocessing pipeline."""
        logging.info("=== Validating Audio Processing ===")

        if metadata_csv_path is None:
            metadata_csv_path = METADATA_CSV

        if not os.path.exists(metadata_csv_path):
            logging.warning("⚠ No metadata CSV found, skipping audio processing validation")
            return True

        try:
            df = pd.read_csv(metadata_csv_path)
            if len(df) == 0:
                logging.warning("⚠ Empty metadata CSV")
                return True

            # Test with first available audio file
            test_audio_path = None
            for _, row in df.head(10).iterrows():
                    audio_path = row['local_path']

                    # Handle relative paths by making them absolute
                    if not os.path.isabs(audio_path):
                        if audio_path.startswith('training_engine/'):
                            # Get the project root (parent of parent of src directory)
                            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                            # Remove 'training_engine/' prefix and join with project root
                            relative_path = audio_path[len('training_engine/'):]
                            audio_path = os.path.join(project_root, 'training_engine', relative_path)
                        else:
                            audio_path = os.path.abspath(audio_path)

                    if os.path.exists(audio_path):
                        test_audio_path = audio_path
                        break

            if test_audio_path is None:
                logging.warning("⚠ No valid audio files found for processing test")
                return True

            # Test audio loading and preprocessing
            waveform, sr = torchaudio.load(test_audio_path)
            logging.info(f"✓ Loaded test audio: shape={waveform.shape}, sr={sr}")

            # Test resampling
            target_sr = 16000
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)
                logging.info(f"✓ Resampled to {target_sr}Hz")

            # Test mel spectrogram
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=target_sr,
                n_fft=512,
                n_mels=80,
                hop_length=256
            )

            spectrogram = mel_transform(waveform)
            logging.info(f"✓ Generated mel spectrogram: shape={spectrogram.shape}")

            # Test normalization
            normalized = spectrogram / (spectrogram.max() + 1e-8)
            logging.info(f"✓ Normalized spectrogram: range=[{normalized.min():.3f}, {normalized.max():.3f}]")

            self.results['audio_processing'] = {
                'loading_successful': True,
                'resampling_successful': True,
                'spectrogram_shape': list(spectrogram.shape),
                'normalization_successful': True
            }

            return True

        except Exception as e:
            logging.error(f"✗ Audio processing test failed: {e}")
            self.issues.append(f"Audio processing error: {e}")
            return False

    def generate_recommendations(self):
        """Generate recommendations based on validation results."""
        logging.info("=== Generating Recommendations ===")

        dataset_results = self.results.get('dataset', {})

        # Dataset recommendations
        if dataset_results.get('total_samples', 0) < 100:
            self.recommendations.append("Very small dataset (<100 samples). Consider collecting more data.")
        elif dataset_results.get('total_samples', 0) < 500:
            self.recommendations.append("Small dataset (<500 samples). Use data augmentation and transfer learning.")

        if dataset_results.get('very_few_samples_classes', 0) > 0:
            self.recommendations.append("Use targeted data augmentation for classes with <3 samples.")

        if dataset_results.get('num_classes', 0) > 50:
            self.recommendations.append("Large number of classes. Consider hierarchical classification or class grouping.")

        # Model recommendations
        model_results = self.results.get('model', {})
        if model_results.get('total_params', 0) > 10_000_000:  # 10M parameters
            self.recommendations.append("Large model for small dataset. Consider reducing model size or using pre-trained models.")

        # Training recommendations
        if not self.results.get('connectivity', {}).get('mongodb', False):
            self.recommendations.append("Use local CSV data loading instead of MongoDB for development.")

        if not torch.cuda.is_available():
            self.recommendations.append("Training on CPU will be slow. Consider using Google Colab or cloud GPU.")

        # Print all recommendations
        if self.recommendations:
            logging.info("📋 Recommendations:")
            for i, rec in enumerate(self.recommendations, 1):
                logging.info(f"  {i}. {rec}")
        else:
            logging.info("✓ No specific recommendations - setup looks good!")

    def save_report(self):
        """Save validation report to file."""
        report_path = os.path.join(VALIDATION_OUTPUT_DIR, 'validation_report.txt')

        with open(report_path, 'w') as f:
            f.write("TWI SPEECH TRAINING PIPELINE VALIDATION REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write("ENVIRONMENT:\n")
            env_results = self.results.get('environment', {})
            f.write(f"  - Missing packages: {env_results.get('missing_packages', [])}\n")
            f.write(f"  - Missing env vars: {env_results.get('missing_env_vars', [])}\n")
            f.write(f"  - CUDA available: {env_results.get('cuda_available', False)}\n\n")

            f.write("DATASET:\n")
            dataset_results = self.results.get('dataset', {})
            f.write(f"  - Total samples: {dataset_results.get('total_samples', 0)}\n")
            f.write(f"  - Number of classes: {dataset_results.get('num_classes', 0)}\n")
            f.write(f"  - Missing files: {dataset_results.get('missing_files', 0)}\n")
            f.write(f"  - Corrupted files: {dataset_results.get('corrupted_files', 0)}\n")
            f.write(f"  - Avg duration: {dataset_results.get('avg_duration', 0):.2f}s\n")
            f.write(f"  - Classes with <3 samples: {dataset_results.get('very_few_samples_classes', 0)}\n\n")

            f.write("MODEL:\n")
            model_results = self.results.get('model', {})
            f.write(f"  - Total parameters: {model_results.get('total_params', 0):,}\n")
            f.write(f"  - Trainable parameters: {model_results.get('trainable_params', 0):,}\n\n")

            if self.issues:
                f.write("ISSUES:\n")
                for issue in self.issues:
                    f.write(f"  - {issue}\n")
                f.write("\n")

            if self.recommendations:
                f.write("RECOMMENDATIONS:\n")
                for rec in self.recommendations:
                    f.write(f"  - {rec}\n")

        logging.info(f"✓ Validation report saved to {report_path}")

    def run_full_validation(self, metadata_csv_path=None):
        """Run complete validation suite."""
        logging.info("🚀 Starting Twi Speech Training Pipeline Validation")
        logging.info("=" * 60)

        success = True

        # For training purposes, we can skip connectivity validation if metadata exists
        skip_connectivity = metadata_csv_path and os.path.exists(metadata_csv_path)

        # Run all validations
        success &= self.validate_environment()

        # Only check connectivity if we don't have local data
        if not skip_connectivity:
            success &= self.validate_data_connectivity()
        else:
            logging.info("=== Skipping Data Connectivity (using local CSV) ===")
            logging.info("✓ Using local CSV data, skipping MongoDB/R2 validation")

        success &= self.validate_dataset(metadata_csv_path)
        success &= self.validate_model_architecture()
        success &= self.validate_audio_processing(metadata_csv_path)

        # Generate recommendations
        self.generate_recommendations()

        # Save report
        self.save_report()

        # Final summary
        logging.info("=" * 60)

        # Filter out connectivity issues if we're using local CSV
        critical_issues = []
        if skip_connectivity:
            critical_issues = [issue for issue in self.issues
                             if not ("MongoDB" in issue or "Cloudflare" in issue)]
        else:
            critical_issues = self.issues

        if success and len(critical_issues) == 0:
            logging.info("🎉 VALIDATION PASSED - Ready for training!")
        elif len(critical_issues) == 0:
            logging.info("⚠️  VALIDATION COMPLETED with warnings - Should work but may need attention")
        else:
            logging.info("❌ VALIDATION FAILED - Please fix issues before training")
            logging.info("Critical issues found:")
            for issue in critical_issues:
                logging.info(f"  - {issue}")

        return success and len(critical_issues) == 0


def main():
    """Main validation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate Twi Speech Training Pipeline")
    parser.add_argument('--metadata_csv', type=str, help="Path to metadata CSV file")
    parser.add_argument('--quick', action='store_true', help="Run quick validation (skip dataset analysis)")

    args = parser.parse_args()

    validator = TwiSpeechValidator()

    if args.quick:
        logging.info("Running quick validation...")
        success = validator.validate_environment()
        success &= validator.validate_model_architecture()
    else:
        success = validator.run_full_validation(args.metadata_csv)

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
