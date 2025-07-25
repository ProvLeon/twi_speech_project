#!/usr/bin/env python3
"""
Twi Speech Training Pipeline Setup Script

This script helps you set up the training environment quickly and correctly.
It installs dependencies, validates the setup, and provides guidance for getting started.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TwiSpeechSetup:
    """Setup manager for the Twi Speech training pipeline."""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.src_dir = self.project_root / "src"
        self.env_file = self.src_dir / ".env"

    def check_python_version(self):
        """Check if Python version is compatible."""
        logging.info("Checking Python version...")

        if sys.version_info < (3, 8):
            logging.error("Python 3.8 or higher is required!")
            logging.error(f"Current version: {sys.version}")
            return False

        logging.info(f"✓ Python version: {sys.version.split()[0]}")
        return True

    def install_dependencies(self):
        """Install required Python packages."""
        logging.info("Installing Python dependencies...")

        # Core ML packages
        core_packages = [
            "torch>=1.12.0",
            "torchaudio>=0.12.0",
            "transformers>=4.20.0",
            "datasets>=2.0.0",
        ]

        # Data processing packages
        data_packages = [
            "pandas>=1.4.0",
            "numpy>=1.21.0",
            "scikit-learn>=1.1.0",
        ]

        # Audio processing packages
        audio_packages = [
            "librosa>=0.9.0",
            "sounddevice>=0.4.0",
            "audiomentations>=1.1.0",
            "pydub>=0.25.0",
        ]

        # Database and cloud packages
        cloud_packages = [
            "pymongo>=4.0.0",
            "boto3>=1.24.0",
            "python-dotenv>=0.19.0",
        ]

        # Visualization and monitoring
        viz_packages = [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "wandb>=0.13.0",
            "tqdm>=4.64.0",
        ]

        all_packages = core_packages + data_packages + audio_packages + cloud_packages + viz_packages

        for package in all_packages:
            try:
                logging.info(f"Installing {package}...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package, "--upgrade"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                logging.info(f"✓ {package.split('>=')[0]} installed")
            except subprocess.CalledProcessError as e:
                logging.error(f"✗ Failed to install {package}: {e}")
                return False

        return True

    def create_env_template(self):
        """Create a template .env file if it doesn't exist."""
        logging.info("Setting up environment variables...")

        if self.env_file.exists():
            logging.info("✓ .env file already exists")
            return True

        env_template = """
# MongoDB Configuration
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net
MONGO_DB_NAME=twi_speech_data

# Cloudflare R2 Configuration
CLOUDFLARE_ACCOUNT_ID=your_account_id_here
CLOUDFLARE_ACCESS_KEY_ID=your_access_key_here
CLOUDFLARE_SECRET_ACCESS_KEY=your_secret_key_here
R2_BUCKET_NAME=your_bucket_name_here

# Weights & Biases (Optional)
WANDB_API_KEY=your_wandb_api_key_here

# CORS Origins (Optional)
FRONTEND_ORIGIN=*
"""

        try:
            with open(self.env_file, 'w') as f:
                f.write(env_template.strip())
            logging.info(f"✓ Created template .env file at {self.env_file}")
            logging.info("⚠️  Please edit the .env file with your actual credentials!")
            return True
        except Exception as e:
            logging.error(f"✗ Failed to create .env file: {e}")
            return False

    def create_directories(self):
        """Create necessary directories."""
        logging.info("Creating directory structure...")

        directories = [
            self.project_root / "data" / "e_commerce_audio",
            self.project_root / "models" / "e_commerce_model",
            self.project_root / "models" / "e_commerce_model_hf_optimized",
            self.project_root / "validation_results",
            self.project_root / "logs",
        ]

        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logging.info(f"✓ Created directory: {directory.relative_to(self.project_root)}")
            except Exception as e:
                logging.error(f"✗ Failed to create directory {directory}: {e}")
                return False

        return True

    def validate_setup(self):
        """Run basic validation to ensure setup is working."""
        logging.info("Validating setup...")

        try:
            # Test imports
            import torch
            import torchaudio
            import transformers
            import pandas as pd
            import numpy as np

            logging.info("✓ Core packages imported successfully")

            # Test CUDA
            if torch.cuda.is_available():
                logging.info(f"✓ CUDA available: {torch.cuda.get_device_name()}")
            else:
                logging.info("⚠️  CUDA not available - will use CPU")

            # Test model creation
            from src.model import ECommerceCommandModel
            model = ECommerceCommandModel(n_input_mels=80, n_output_classes=10)
            dummy_input = torch.randn(1, 1, 80, 100)

            with torch.no_grad():
                output = model(dummy_input)

            if output.shape == (1, 10):
                logging.info("✓ Model architecture test passed")
            else:
                logging.error(f"✗ Model output shape incorrect: {output.shape}")
                return False

            return True

        except ImportError as e:
            logging.error(f"✗ Import error: {e}")
            return False
        except Exception as e:
            logging.error(f"✗ Validation failed: {e}")
            return False

    def print_next_steps(self):
        """Print next steps for the user."""
        logging.info("\n" + "="*60)
        logging.info("🎉 SETUP COMPLETE!")
        logging.info("="*60)

        steps = [
            "1. Edit the .env file with your actual credentials:",
            f"   {self.env_file}",
            "",
            "2. Validate your complete setup:",
            "   python -m src.pipeline --mode validate",
            "",
            "3. Fetch your data:",
            "   python -m src.pipeline --mode fetch",
            "",
            "4. Train your model (recommended approach):",
            "   python -m src.pipeline --mode train --use-hf",
            "",
            "5. Test your trained model:",
            "   python -m src.pipeline --mode test",
            "",
            "For detailed documentation, see:",
            "   TRAINING_IMPROVEMENTS.md",
            "",
            "For issues or questions:",
            "   - Run validation first: python -m src.pipeline --mode validate",
            "   - Check the validation report in validation_results/",
            "   - Review console logs for specific errors",
        ]

        for step in steps:
            if step:
                logging.info(f"  {step}")
            else:
                logging.info("")

    def run_setup(self):
        """Run the complete setup process."""
        logging.info("🚀 Starting Twi Speech Training Pipeline Setup")
        logging.info("="*60)

        success = True

        # Check Python version
        success &= self.check_python_version()

        # Install dependencies
        if success:
            success &= self.install_dependencies()

        # Create directories
        if success:
            success &= self.create_directories()

        # Create .env template
        if success:
            success &= self.create_env_template()

        # Validate setup
        if success:
            success &= self.validate_setup()

        if success:
            self.print_next_steps()
        else:
            logging.error("\n❌ Setup failed! Please check the errors above.")
            return False

        return True


def main():
    """Main setup function."""
    import argparse

    parser = argparse.ArgumentParser(description="Set up Twi Speech Training Pipeline")
    parser.add_argument('--skip-install', action='store_true',
                       help="Skip package installation (useful if packages are already installed)")
    parser.add_argument('--validate-only', action='store_true',
                       help="Only run validation, skip setup")

    args = parser.parse_args()

    setup = TwiSpeechSetup()

    if args.validate_only:
        success = setup.validate_setup()
        return 0 if success else 1

    if args.skip_install:
        logging.info("Skipping package installation...")
        success = True
        success &= setup.create_directories()
        success &= setup.create_env_template()
        success &= setup.validate_setup()

        if success:
            setup.print_next_steps()

        return 0 if success else 1

    # Full setup
    success = setup.run_setup()
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
