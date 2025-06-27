"""
Twi Speech Recognition Training Engine

A professional training engine for Twi speech-to-text models designed to achieve
Word Error Rate (WER) < 30% using state-of-the-art deep learning techniques.

This package provides:
- Data loading and preprocessing pipelines
- Model architectures optimized for Twi language
- Training loops with advanced optimization techniques
- Evaluation metrics and monitoring
- Experiment tracking and management
- Model deployment utilities

Key Features:
- Support for Wav2Vec2, Whisper, and other SOTA models
- Twi-specific text preprocessing and phoneme handling
- Multi-dialect support (Asante, Akuapem, Fante)
- Advanced data augmentation techniques
- Distributed training capabilities
- Comprehensive evaluation suite
- Integration with existing data collection backend

Author: Twi Speech Recognition Team
Version: 1.0.0
License: MIT
"""

import logging
import sys
from pathlib import Path

# Package metadata
__version__ = "1.0.0"
__author__ = "Twi Speech Recognition Team"
__email__ = "contact@twispeech.ai"
__license__ = "MIT"

# Package root directory
PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent

# Ensure logs directory exists before configuring logging
try:
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
    log_file = PROJECT_ROOT / "logs" / "training_engine.log"

    # Configure logging for the package
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode='a')
        ]
    )
except (OSError, PermissionError) as e:
    # Fallback to console-only logging if file logging fails
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    print(f"Warning: Could not create log file. Using console logging only. Error: {e}")

logger = logging.getLogger(__name__)

# Package imports
try:
    from .trainer import TwiSpeechTrainer
    from .models import *
    from .data import *
    from .utils import *
    from .evaluation import *

    logger.info("Successfully imported all training engine components")

except ImportError as e:
    logger.warning(f"Some components could not be imported: {e}")
    logger.info("This is normal if running setup for the first time")

# Package configuration
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Ensure directories exist
for dir_path in [DATA_DIR, MODELS_DIR, EXPERIMENTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Global configuration
GLOBAL_CONFIG = {
    "project_name": "twi_speech_recognition",
    "target_wer": 0.30,
    "supported_languages": ["tw"],
    "supported_dialects": ["Asante", "Akuapem", "Fante"],
    "default_sample_rate": 16000,
    "default_model": "wav2vec2_base",
}

def get_version():
    """Get package version."""
    return __version__

def get_config_path(config_name: str) -> Path:
    """Get path to configuration file."""
    return CONFIG_DIR / f"{config_name}.yaml"

def get_model_path(model_name: str) -> Path:
    """Get path to model directory."""
    return MODELS_DIR / model_name

def get_experiment_path(experiment_name: str) -> Path:
    """Get path to experiment directory."""
    return EXPERIMENTS_DIR / experiment_name

def setup_environment():
    """Setup the training environment."""
    logger.info("Setting up Twi Speech Training Engine environment...")

    # Check Python version
    if sys.version_info < (3, 8):
        raise RuntimeError("Python 3.8 or higher is required")

    # Check for required dependencies
    required_packages = [
        "torch",
        "transformers",
        "datasets",
        "librosa",
        "jiwer",
        "wandb",
        "hydra-core"
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info("Please install missing packages using: pip install -r requirements.txt")
        raise ImportError(f"Missing required packages: {', '.join(missing_packages)}")

    logger.info("Environment setup completed successfully")

# Initialize the environment when package is imported
try:
    setup_environment()
except Exception as e:
    logger.warning(f"Environment setup failed: {e}")
    logger.info("Some features may not be available")

# Export public API
__all__ = [
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",

    # Core components
    "TwiSpeechTrainer",

    # Utility functions
    "get_version",
    "get_config_path",
    "get_model_path",
    "get_experiment_path",
    "setup_environment",

    # Constants
    "GLOBAL_CONFIG",
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "DATA_DIR",
    "MODELS_DIR",
    "EXPERIMENTS_DIR",
]

logger.info(f"Twi Speech Training Engine v{__version__} initialized")
logger.info(f"Target WER: {GLOBAL_CONFIG['target_wer']*100:.1f}%")
logger.info(f"Supported dialects: {', '.join(GLOBAL_CONFIG['supported_dialects'])}")
