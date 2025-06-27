#!/usr/bin/env python3
"""
Main Training Script for Twi Speech Recognition Engine

This script provides the main entry point for training Twi speech-to-text models.
It supports multiple model architectures, comprehensive evaluation metrics, and
professional ML practices to achieve Word Error Rate (WER) < 30%.

Usage:
    python train.py --config configs/config.yaml
    python train.py --config configs/wav2vec2_base.yaml --experiment-name my_experiment
    python train.py --help

Key Features:
- Support for multiple model architectures (Wav2Vec2, Whisper, Custom)
- Comprehensive training pipeline with advanced optimization
- Integration with Weights & Biases and TensorBoard
- Automatic hyperparameter optimization
- Multi-GPU and distributed training support
- Comprehensive evaluation and monitoring
- Model deployment preparation

Author: Twi Speech Recognition Team
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Optional, Dict, Any
import argparse
import yaml

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import compose, initialize_config_dir

# Import training engine components
from src.trainer import TwiSpeechTrainer
from src.utils.reproducibility import set_seed
from src.utils.logging_utils import setup_logging, log_system_info
from src.utils.model_utils import analyze_model_architecture

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train Twi Speech Recognition Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Configuration
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Directory containing configuration files"
    )

    # Experiment settings
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for this experiment (overrides config)"
    )

    parser.add_argument(
        "--tags",
        nargs="+",
        default=None,
        help="Tags for experiment tracking"
    )

    # Training overrides
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size per device (overrides config)"
    )

    parser.add_argument(
        "--learning-rate", "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
    )

    # Model settings
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["wav2vec2_base", "wav2vec2_large", "whisper_base", "custom"],
        help="Model architecture to use"
    )

    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained model or model name"
    )

    # Data settings
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing training data"
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching processed data"
    )

    # Hardware settings
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for training"
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of data loader workers"
    )

    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed precision training"
    )

    # Logging and monitoring
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name"
    )

    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity (username/team)"
    )

    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default=None,
        help="TensorBoard log directory"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    # Evaluation and checkpointing
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="Evaluate every N steps"
    )

    parser.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="Save checkpoint every N steps"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments",
        help="Output directory for experiments"
    )

    # Debugging and development
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (smaller dataset, more logging)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without actual training"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    # Hyperparameter optimization
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run hyperparameter optimization"
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of optimization trials"
    )

    return parser.parse_args()


def load_config(config_path: str, config_dir: Optional[str] = None) -> DictConfig:
    """Load configuration from file"""
    config_path = Path(config_path)

    if config_dir is None:
        config_dir = str(config_path.parent.absolute())

    config_name = config_path.stem

    try:
        # Initialize Hydra with config directory
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name=f"{config_name}.yaml")

        logger.info(f"Loaded configuration from {config_path}")
        return cfg

    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")

        # Fallback to direct YAML loading
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            cfg = OmegaConf.create(config_dict)
            logger.info(f"Loaded configuration using fallback method")
            return cfg
        except Exception as e2:
            logger.error(f"Fallback configuration loading also failed: {e2}")
            raise


def apply_overrides(config: DictConfig, args: argparse.Namespace) -> DictConfig:
    """Apply command line overrides to configuration"""

    # Experiment settings
    if args.experiment_name:
        config.experiment.name = args.experiment_name

    if args.tags:
        config.experiment.tags = args.tags

    # Training overrides
    if args.epochs:
        config.training.num_epochs = args.epochs

    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
        config.training.per_device_eval_batch_size = args.batch_size

    if args.learning_rate:
        config.training.learning_rate = args.learning_rate

    # Model overrides
    if args.model:
        config.model = f"configs/model/{args.model}.yaml"

    if args.pretrained:
        config.model.pretrained_model_name = args.pretrained

    # Data overrides
    if args.data_dir:
        config.paths.data_dir = args.data_dir

    if args.cache_dir:
        config.paths.cache_dir = args.cache_dir

    # Hardware overrides
    if args.device != "auto":
        config.hardware.device = args.device

    if args.num_workers:
        config.hardware.num_workers = args.num_workers

    if args.mixed_precision:
        config.hardware.mixed_precision = True

    # Logging overrides
    if args.wandb_project:
        config.logging.wandb.project = args.wandb_project

    if args.wandb_entity:
        config.logging.wandb.entity = args.wandb_entity

    if args.tensorboard_dir:
        config.logging.tensorboard.log_dir = args.tensorboard_dir

    if args.log_level:
        config.logging.level = args.log_level

    # Evaluation overrides
    if args.eval_steps:
        config.training.eval_steps = args.eval_steps

    if args.save_steps:
        config.training.save_steps = args.save_steps

    # Output directory
    config.paths.experiment_dir = args.output_dir

    # Debug mode
    if args.debug:
        config.debug.dev_mode = True
        config.debug.use_sample_data = True
        config.debug.sample_size = 100
        config.logging.level = "DEBUG"

    # Seed
    if args.seed:
        config.seed.random = args.seed
        config.seed.numpy = args.seed
        config.seed.torch = args.seed
        config.seed.transformers = args.seed

    return config


def setup_environment(config: DictConfig):
    """Setup training environment"""

    # Setup logging
    setup_logging(config.logging)

    # Set random seeds for reproducibility
    seed_value = config.seed.random if hasattr(config.seed, 'random') else config.seed
    deterministic_mode = getattr(config, 'deterministic', False)
    set_seed(seed_value, deterministic_mode)

    # Create necessary directories
    for path_key in ['data_dir', 'model_dir', 'experiment_dir', 'cache_dir', 'checkpoint_dir', 'log_dir']:
        if hasattr(config.paths, path_key):
            path = Path(getattr(config.paths, path_key))
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {path}")

    # Log system information
    log_system_info()

    # Log configuration
    logger.info("Training Configuration:")
    logger.info(f"  Project: {config.project.name}")
    logger.info(f"  Target WER: {config.project.target_wer:.1%}")
    logger.info(f"  Experiment: {config.experiment.name}")
    logger.info(f"  Model: {config.model._target_}")
    logger.info(f"  Epochs: {config.training.num_epochs}")
    logger.info(f"  Batch size: {config.training.per_device_train_batch_size}")
    logger.info(f"  Learning rate: {config.training.learning_rate}")

    # Check hardware availability
    if torch.cuda.is_available():
        logger.info(f"CUDA devices available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    else:
        logger.info("CUDA not available, using CPU")


def validate_config(config: DictConfig) -> bool:
    """Validate configuration for common issues"""
    errors = []
    warnings = []

    # Check required fields
    required_fields = [
        'project.name',
        'project.target_wer',
        'training.num_epochs',
        'training.learning_rate',
        'model._target_'
    ]

    for field in required_fields:
        if not OmegaConf.select(config, field):
            errors.append(f"Missing required field: {field}")

    # Check reasonable values
    if config.training.learning_rate <= 0 or config.training.learning_rate > 1.0:
        warnings.append(f"Learning rate {config.training.learning_rate} seems unreasonable")

    if config.training.num_epochs <= 0 or config.training.num_epochs > 1000:
        warnings.append(f"Number of epochs {config.training.num_epochs} seems unreasonable")

    if config.project.target_wer <= 0 or config.project.target_wer > 1.0:
        errors.append(f"Target WER {config.project.target_wer} must be between 0 and 1")

    # Log warnings
    for warning in warnings:
        logger.warning(warning)

    # Log errors and return validation result
    for error in errors:
        logger.error(error)

    return len(errors) == 0


def run_hyperparameter_optimization(config: DictConfig, n_trials: int = 50):
    """Run hyperparameter optimization using Optuna"""
    try:
        import optuna
    except ImportError:
        logger.error("Optuna not available for hyperparameter optimization")
        return None

    def objective(trial):
        # Create modified config for this trial
        trial_config = config.copy()

        # Suggest hyperparameters
        trial_config.training.learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        trial_config.training.per_device_train_batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
        trial_config.training.weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
        trial_config.training.warmup_steps = trial.suggest_int('warmup_steps', 100, 2000)

        # Run training
        trainer = TwiSpeechTrainer(trial_config)
        trainer.train()

        # Return metric to optimize (WER - lower is better)
        return trainer.state.best_wer

    # Create study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    logger.info("Hyperparameter optimization completed")
    logger.info(f"Best trial: {study.best_trial.value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")

    return study.best_params


def main():
    """Main training function"""
    # Parse arguments
    args = parse_arguments()

    try:
        # Load configuration
        config = load_config(args.config, args.config_dir)

        # Apply command line overrides
        config = apply_overrides(config, args)

        # Validate configuration
        if not validate_config(config):
            logger.error("Configuration validation failed")
            sys.exit(1)

        # Setup environment
        setup_environment(config)

        # Handle dry run
        if args.dry_run:
            logger.info("Dry run mode - configuration loaded successfully")
            logger.info("Configuration summary:")
            print(OmegaConf.to_yaml(config))
            return

        # Handle hyperparameter optimization
        if args.optimize:
            logger.info("Running hyperparameter optimization...")
            best_params = run_hyperparameter_optimization(config, args.n_trials)
            if best_params:
                logger.info("Optimization completed successfully")
            return

        # Create trainer
        logger.info("Creating trainer...")
        trainer = TwiSpeechTrainer(config)

        # Setup model for analysis (without data loaders yet)
        logger.info("Setting up model...")
        trainer.setup_model()

        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming training from {args.resume}")
            trainer._load_checkpoint(args.resume)

        # Log model information
        model_analysis = analyze_model_architecture(trainer.model)
        logger.info("Model Analysis:")
        logger.info(f"  Total parameters: {model_analysis['total_parameters']:,}")
        logger.info(f"  Trainable parameters: {model_analysis['trainable_parameters']:,}")
        logger.info(f"  Model size: {model_analysis['model_size_mb']:.2f} MB")

        # Start training
        logger.info("Starting training...")
        trainer.train()

        # Training completed
        logger.info("Training completed successfully!")
        logger.info(f"Best WER achieved: {trainer.state.best_wer:.4f}")

        # Check if target was achieved
        if trainer.state.best_wer < config.project.target_wer:
            logger.info(f"🎉 Target WER of {config.project.target_wer:.1%} achieved!")
        else:
            improvement_needed = trainer.state.best_wer - config.project.target_wer
            logger.info(f"Target WER not achieved. Need {improvement_needed:.1%} improvement.")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
