import argparse
import logging
import os
import sys

from .data_loader import load_and_prepare_data, load_backend_env, BACKEND_ENV_PATH
from .train import run_training, run_testing
from .train_hf import run_hf_training
from .validate_setup import TwiSpeechValidator

AUDIO_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'e_commerce_audio')
METADATA_CSV = os.path.join(AUDIO_DIR, 'metadata.csv')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch():
    logging.info("=== [FETCH MODE] ===")

    # Load environment variables first
    try:
        load_backend_env(BACKEND_ENV_PATH)
        logging.info("✓ Environment variables loaded")
    except Exception as e:
        logging.error(f"Failed to load environment variables: {e}")
        return

    df = load_and_prepare_data()
    if df is not None and not df.empty:
        logging.info(f"Fetched and saved {len(df)} audio files and metadata.")

        # Run validation after fetching data
        logging.info("Running post-fetch validation...")
        validator = TwiSpeechValidator()
        validator.validate_dataset(METADATA_CSV)
        validator.generate_recommendations()
    else:
        logging.warning("No data fetched.")

def train(use_huggingface=False, augment=True):
    """Train using either traditional CNN-RNN or HuggingFace Wav2Vec2 approach."""
    if use_huggingface:
        logging.info("=== [HUGGINGFACE TRAIN MODE] ===")
    else:
        logging.info("=== [TRADITIONAL TRAIN MODE] ===")

    if not os.path.exists(METADATA_CSV):
        logging.error(f"Metadata CSV not found. Run fetch mode first. {METADATA_CSV}")
        return

    # Load environment variables for training
    try:
        load_backend_env(BACKEND_ENV_PATH)
        logging.info("✓ Environment variables loaded for training")
    except Exception as e:
        logging.warning(f"Could not load .env file: {e}")
        logging.info("Continuing with system environment variables")

    # Run pre-training validation with local data focus
    logging.info("Running pre-training validation...")
    validator = TwiSpeechValidator()

    # Since we have local CSV, focus on essential validations
    validation_passed = True
    validation_passed &= validator.validate_environment()
    validation_passed &= validator.validate_dataset(METADATA_CSV)
    validation_passed &= validator.validate_model_architecture()
    validation_passed &= validator.validate_audio_processing(METADATA_CSV)

    validator.generate_recommendations()
    validator.save_report()

    if not validation_passed:
        logging.error("Critical validation failed. Please fix issues before training.")
        return
    else:
        logging.info("✓ Essential validations passed. Proceeding with training...")

    if use_huggingface:
        # Use improved HuggingFace training
        run_hf_training(metadata_csv=METADATA_CSV, augment_data=augment)
    else:
        # Use traditional training
        run_training(metadata_csv=METADATA_CSV)

def test():
    logging.info("=== [TEST MODE] ===")
    if not os.path.exists(METADATA_CSV):
        logging.error("Metadata CSV not found. Run fetch mode first.")
        return
    run_testing(metadata_csv=METADATA_CSV)

def validate(quick=False):
    """Run comprehensive validation of the training pipeline."""
    logging.info("=== [VALIDATE MODE] ===")

    # Load environment variables first
    try:
        load_backend_env(BACKEND_ENV_PATH)
        logging.info("✓ Environment variables loaded for validation")
    except Exception as e:
        logging.warning(f"Could not load .env file: {e}")
        logging.info("Using system environment variables")

    validator = TwiSpeechValidator()

    if quick:
        success = validator.validate_environment()
        success &= validator.validate_model_architecture()
    else:
        success = validator.run_full_validation(METADATA_CSV)

    if success:
        logging.info("✅ All validations passed! Ready for training.")
        return 0
    else:
        logging.error("❌ Validation failed. Check the issues above.")
        return 1

def main():
    parser = argparse.ArgumentParser(description="Enhanced Twi Speech E-commerce Command Recognition Pipeline")
    parser.add_argument('--mode', choices=['fetch', 'train', 'test', 'validate'], required=True,
                       help="Pipeline mode to run")
    parser.add_argument('--use-hf', action='store_true',
                       help="Use HuggingFace Wav2Vec2 model instead of traditional CNN-RNN (recommended)")
    parser.add_argument('--no-augment', action='store_true',
                       help="Disable data augmentation during training")
    parser.add_argument('--quick-validate', action='store_true',
                       help="Run quick validation (environment and model only)")
    args = parser.parse_args()

    try:
        if args.mode == 'fetch':
            fetch()
        elif args.mode == 'train':
            train(use_huggingface=args.use_hf, augment=not args.no_augment)
        elif args.mode == 'test':
            test()
        elif args.mode == 'validate':
            sys.exit(validate(quick=args.quick_validate))

        logging.info("Pipeline execution completed successfully!")

    except KeyboardInterrupt:
        logging.info("Pipeline execution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
