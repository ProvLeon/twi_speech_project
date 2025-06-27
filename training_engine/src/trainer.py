"""
Main Trainer Class for Twi Speech Recognition

This module implements the TwiSpeechTrainer class, which orchestrates the entire
training pipeline for Twi speech-to-text models. It's designed to achieve
Word Error Rate (WER) < 30% through professional ML practices.

Key Features:
- Support for multiple model architectures (Wav2Vec2, Whisper, etc.)
- Advanced training techniques (curriculum learning, mixed precision, etc.)
- Comprehensive evaluation and monitoring
- Integration with experiment tracking (Weights & Biases, TensorBoard)
- Automatic hyperparameter optimization
- Distributed training support
- Model deployment preparation

Author: Twi Speech Recognition Team
"""

import os
import sys
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    get_scheduler
)

from .data.data_manager import TwiDataManager, TwiSpeechDataset
from .models.model_factory import ModelFactory
from .evaluation.evaluator import TwiEvaluator
from .utils.metrics import WERMetric, CERMetric
from .utils.callbacks import TwiTrainingCallbacks
from .utils.logging_utils import setup_logging, log_system_info
from .utils.reproducibility import set_seed
from .utils.model_utils import count_parameters, get_model_size

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Training state tracking"""
    epoch: int = 0
    step: int = 0
    best_wer: float = float('inf')
    best_cer: float = float('inf')
    best_model_path: Optional[str] = None
    training_loss: float = 0.0
    validation_loss: float = 0.0
    learning_rate: float = 0.0
    grad_norm: float = 0.0

    # Performance tracking
    samples_per_second: float = 0.0
    tokens_per_second: float = 0.0
    gpu_memory_used: float = 0.0

    # Early stopping
    patience_counter: int = 0
    should_stop: bool = False


class TwiSpeechTrainer:
    """
    Main trainer class for Twi speech recognition models.

    This class handles the complete training pipeline from data loading
    to model deployment preparation, with focus on achieving WER < 30%.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the trainer.

        Args:
            config: Hydra configuration object containing all training parameters
        """
        self.config = config
        self.device = self._setup_device()
        self.is_distributed = self._setup_distributed_training()

        # Initialize state
        self.state = TrainingState()
        self.training_stats = defaultdict(list)

        # Setup tracking flags
        self._model_setup_done = False
        self._data_setup_done = False

        # Setup logging
        setup_logging(config.logging)
        logger.info(f"Initializing TwiSpeechTrainer with target WER < {config.project.target_wer:.1%}")

        # Log system information
        log_system_info()

        # Set reproducibility
        set_seed(config.seed)

        # Initialize components
        self.data_manager = None
        self.model = None
        self.processor = None
        self.optimizer = None
        self.scheduler = None
        self.evaluator = None
        self.scaler = GradScaler() if config.hardware.mixed_precision else None

        # Initialize experiment tracking
        self._setup_experiment_tracking()

        # Training callbacks
        self.callbacks = TwiTrainingCallbacks(config)

    def _setup_device(self) -> torch.device:
        """Setup training device (CPU/CUDA/MPS)."""
        if self.config.hardware.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
                logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("Using Apple MPS device")
            else:
                device = torch.device("cpu")
                logger.warning("No GPU available, using CPU")
        else:
            device = torch.device(self.config.hardware.device)

        return device

    def _setup_distributed_training(self) -> bool:
        """Setup distributed training if available."""
        if torch.cuda.device_count() > 1 and dist.is_available():
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')
            return True
        return False

    def _setup_experiment_tracking(self):
        """Setup experiment tracking with W&B, TensorBoard, etc."""
        if self.config.logging.wandb.project:
            try:
                wandb.init(
                    project=self.config.logging.wandb.project,
                    entity=self.config.logging.wandb.get("entity"),
                    name=self.config.experiment.name,
                    tags=self.config.experiment.tags,
                    notes=self.config.experiment.notes,
                    config=OmegaConf.to_container(self.config, resolve=True)
                )
                logger.info("Weights & Biases tracking initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize W&B: {e}")

    def setup_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Setup data loading pipeline.

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if self._data_setup_done:
            logger.debug("Data already set up, skipping...")
            # Return cached data loaders if available
            return getattr(self, '_train_loader', None), getattr(self, '_val_loader', None), getattr(self, '_test_loader', None)

        logger.info("Setting up data pipeline...")

        self.data_manager = TwiDataManager(self.config.data)

        # Load and preprocess data samples
        train_samples, val_samples, test_samples = self.data_manager.prepare_datasets()

        # Store samples for later dataset creation
        self._train_samples = train_samples
        self._val_samples = val_samples
        self._test_samples = test_samples

        logger.info(f"Data samples prepared:")
        logger.info(f"  Train samples: {len(train_samples)}")
        logger.info(f"  Validation samples: {len(val_samples)}")
        logger.info(f"  Test samples: {len(test_samples)}")

        # Datasets and loaders will be created after model setup
        train_loader = None
        val_loader = None
        test_loader = None

        # Cache data loaders and mark setup as complete
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._test_loader = test_loader
        self._data_setup_done = True

        return train_loader, val_loader, test_loader

    def setup_model(self):
        """Setup model architecture and processor."""
        if self._model_setup_done:
            logger.debug("Model already set up, skipping...")
            return

        logger.info("Setting up model architecture...")

        # Create model using factory
        model_factory = ModelFactory(self.config.model)
        self.model, self.processor = model_factory.create_model()

        # Validate model creation
        if self.model is None:
            raise RuntimeError("Model creation failed: model_factory.create_model() returned None")

        if self.processor is None:
            raise RuntimeError("Processor creation failed: model_factory.create_model() returned None processor")

        # Ensure model has parameters method
        if not hasattr(self.model, 'parameters'):
            raise RuntimeError(f"Invalid model object: {type(self.model)} does not have parameters() method")

        # Move model to device
        self.model = self.model.to(self.device)

        # Setup distributed training
        if self.is_distributed:
            if hasattr(self.model, 'parameters'):
                self.model = DDP(self.model, find_unused_parameters=False)
            else:
                logger.warning("Model does not have parameters method, skipping DDP")

        # Log model information
        total_params = count_parameters(self.model)
        model_size_mb = get_model_size(self.model)

        logger.info(f"Model architecture: {self.config.model._target_.split('.')[-1]}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Model size: {model_size_mb:.2f} MB")

        # Compile model for PyTorch 2.0+ if enabled
        if self.config.training.model_compilation.enabled and hasattr(torch, 'compile'):
            logger.info("Compiling model with PyTorch 2.0...")
            self.model = torch.compile(
                self.model,
                backend=self.config.training.model_compilation.backend,
                mode=self.config.training.model_compilation.mode
            )

        # Mark setup as complete
        self._model_setup_done = True

    def _create_datasets_and_loaders(self):
        """Create datasets and data loaders after both model and data are ready"""
        if not hasattr(self, '_train_samples') or self.processor is None:
            raise RuntimeError("Cannot create datasets: samples or processor not available")

        # Create datasets using the processor

        train_dataset = TwiSpeechDataset(
            self._train_samples, self.processor,
            self.data_manager.audio_processor, self.data_manager.text_processor,
            is_training=True
        )

        val_dataset = TwiSpeechDataset(
            self._val_samples, self.processor,
            self.data_manager.audio_processor, self.data_manager.text_processor,
            is_training=False
        )

        test_dataset = TwiSpeechDataset(
            self._test_samples, self.processor,
            self.data_manager.audio_processor, self.data_manager.text_processor,
            is_training=False
        )

        # Create data loaders
        train_loader = self.data_manager.create_dataloader(
            train_dataset,
            batch_size=self.config.training.per_device_train_batch_size,
            shuffle=True,
            is_training=True
        )

        val_loader = self.data_manager.create_dataloader(
            val_dataset,
            batch_size=self.config.training.per_device_eval_batch_size,
            shuffle=False,
            is_training=False
        )

        test_loader = self.data_manager.create_dataloader(
            test_dataset,
            batch_size=self.config.training.per_device_eval_batch_size,
            shuffle=False,
            is_training=False
        )

        logger.info(f"Datasets and data loaders created:")
        logger.info(f"  Train dataset: {len(train_dataset)} samples")
        logger.info(f"  Validation dataset: {len(val_dataset)} samples")
        logger.info(f"  Test dataset: {len(test_dataset)} samples")

        return train_loader, val_loader, test_loader

    def setup_optimizer_and_scheduler(self, num_training_steps: int):
        """Setup optimizer and learning rate scheduler."""
        logger.info("Setting up optimizer and scheduler...")

        # Setup optimizer
        if self.config.training.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.config.training.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.training.optimizer}")

        # Setup learning rate scheduler
        scheduler_config = self.config.training.lr_scheduler
        warmup_steps = int(num_training_steps * scheduler_config.warmup_ratio) if scheduler_config.warmup_ratio else scheduler_config.warmup_steps

        self.scheduler = get_scheduler(
            name=scheduler_config.type,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )

        logger.info(f"Optimizer: {self.config.training.optimizer}")
        logger.info(f"Learning rate: {self.config.training.learning_rate}")
        logger.info(f"Scheduler: {scheduler_config.type}")
        logger.info(f"Warmup steps: {warmup_steps}")

    def setup_evaluator(self):
        """Setup evaluation pipeline."""
        self.evaluator = TwiEvaluator(
            processor=self.processor,
            config=self.config.evaluation
        )
        logger.info("Evaluation pipeline ready")

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        total_samples = 0
        start_time = time.time()

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{self.config.training.num_epochs}",
            disable=not self._is_main_process()
        )

        for step, batch in enumerate(progress_bar):
            batch_start_time = time.time()

            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass with mixed precision
            with autocast(enabled=self.config.hardware.mixed_precision):
                outputs = self.model(**batch)
                loss = outputs.loss

                # Scale loss for gradient accumulation
                loss = loss / self.config.training.gradient_accumulation_steps

            # Backward pass
            if self.config.hardware.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights
            if (step + 1) % self.config.training.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.training.max_grad_norm > 0:
                    if self.config.hardware.mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                    if (hasattr(self.config.training, 'max_grad_norm') and
                        self.config.training.max_grad_norm > 0 and
                        self.model is not None and
                        hasattr(self.model, 'parameters')):
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.training.max_grad_norm
                        )
                    self.state.grad_norm = grad_norm.item()

                # Optimizer step
                if self.config.hardware.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # Scheduler step
                self.scheduler.step()
                self.optimizer.zero_grad()

                # Update state
                self.state.step += 1
                self.state.learning_rate = self.scheduler.get_last_lr()[0]

            # Update metrics
            batch_loss = loss.item() * self.config.training.gradient_accumulation_steps
            total_loss += batch_loss
            batch_size = batch['input_values'].size(0)
            total_samples += batch_size

            # Calculate performance metrics
            batch_time = time.time() - batch_start_time
            samples_per_second = batch_size / batch_time

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'lr': f'{self.state.learning_rate:.2e}',
                'samples/s': f'{samples_per_second:.1f}'
            })

            # Log metrics
            if self.state.step % self.config.training.logging_steps == 0:
                self._log_training_step(batch_loss, samples_per_second)

            # Evaluation during training
            if (self.config.training.evaluation_strategy == "steps" and
                self.state.step % self.config.training.eval_steps == 0):
                eval_metrics = self.evaluate(train_loader, prefix="eval")
                self._log_evaluation_metrics(eval_metrics)

                # Check for early stopping
                if self._check_early_stopping(eval_metrics):
                    logger.info("Early stopping triggered")
                    break

            # Save checkpoint
            if (self.config.training.save_strategy == "steps" and
                self.state.step % self.config.training.save_steps == 0):
                self._save_checkpoint(f"checkpoint-step-{self.state.step}")

        # Calculate epoch metrics
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        samples_per_second = total_samples / epoch_time

        epoch_metrics = {
            'train_loss': avg_loss,
            'train_samples_per_second': samples_per_second,
            'train_epoch_time': epoch_time,
            'learning_rate': self.state.learning_rate,
            'grad_norm': self.state.grad_norm
        }

        return epoch_metrics

    def evaluate(self, data_loader: DataLoader, prefix: str = "eval") -> Dict[str, float]:
        """
        Evaluate model on validation/test data.

        Args:
            data_loader: Data loader for evaluation
            prefix: Prefix for metric names

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Running {prefix} evaluation...")

        self.model.eval()

        total_loss = 0.0
        predictions = []
        references = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"{prefix.title()} evaluation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Forward pass
                with autocast(enabled=self.config.hardware.mixed_precision):
                    outputs = self.model(**batch)
                    loss = outputs.loss

                total_loss += loss.item()

                # Decode predictions
                logits = outputs.logits
                predicted_ids = torch.argmax(logits, dim=-1)

                # Process predictions and references
                batch_predictions = self.processor.batch_decode(predicted_ids)
                batch_references = self.processor.batch_decode(batch['labels'])

                predictions.extend(batch_predictions)
                references.extend(batch_references)

        # Calculate metrics
        avg_loss = total_loss / len(data_loader)
        metrics = self.evaluator.compute_metrics(predictions, references)

        # Add loss to metrics
        metrics[f"{prefix}_loss"] = avg_loss

        # Log metrics
        logger.info(f"{prefix.title()} metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")

        self.model.train()
        return metrics

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Target WER: {self.config.project.target_wer:.1%}")

        # Setup components
        self.setup_data()  # Prepares samples
        self.setup_model()  # Creates model and processor

        # Now create datasets and loaders with both samples and processor available
        train_loader, val_loader, test_loader = self._create_datasets_and_loaders()

        # Calculate training steps
        num_training_steps = (
            len(train_loader) * self.config.training.num_epochs //
            self.config.training.gradient_accumulation_steps
        )

        self.setup_optimizer_and_scheduler(num_training_steps)
        self.setup_evaluator()

        logger.info(f"Training setup complete:")
        logger.info(f"  Total training steps: {num_training_steps}")
        logger.info(f"  Epochs: {self.config.training.num_epochs}")
        logger.info(f"  Batch size per device: {self.config.training.per_device_train_batch_size}")
        logger.info(f"  Gradient accumulation steps: {self.config.training.gradient_accumulation_steps}")

        # Training loop
        try:
            for epoch in range(self.config.training.num_epochs):
                self.state.epoch = epoch

                # Train epoch
                train_metrics = self.train_epoch(train_loader, epoch)

                # Evaluate
                if self.config.training.evaluation_strategy == "epoch":
                    eval_metrics = self.evaluate(val_loader, prefix="eval")
                    train_metrics.update(eval_metrics)

                    # Check WER target
                    current_wer = eval_metrics.get('eval_wer', float('inf'))
                    if current_wer < self.config.project.target_wer:
                        logger.info(f"🎉 Target WER achieved! Current WER: {current_wer:.1%}")
                        if self.config.training.early_stopping.enabled:
                            logger.info("Stopping training as target WER is achieved")
                            break

                    # Update best metrics
                    if current_wer < self.state.best_wer:
                        self.state.best_wer = current_wer
                        self.state.best_model_path = self._save_checkpoint(f"best-model-epoch-{epoch}")
                        logger.info(f"New best WER: {self.state.best_wer:.4f}")

                # Log epoch metrics
                self._log_epoch_metrics(train_metrics, epoch)

                # Save checkpoint
                if self.config.training.save_strategy == "epoch":
                    self._save_checkpoint(f"checkpoint-epoch-{epoch}")

                # Check early stopping
                if self.config.training.early_stopping.enabled:
                    if self._check_early_stopping(train_metrics):
                        logger.info("Early stopping triggered")
                        break

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise

        # Final evaluation on test set
        if test_loader:
            logger.info("Running final evaluation on test set...")

            # Load best model if available
            if self.state.best_model_path:
                self._load_checkpoint(self.state.best_model_path)

            test_metrics = self.evaluate(test_loader, prefix="test")
            self._log_test_metrics(test_metrics)

        # Training summary
        self._log_training_summary()

        # Export model for deployment
        if self.config.training.export.export_best_model:
            self._export_model()

        logger.info("Training completed!")

    def _check_early_stopping(self, metrics: Dict[str, float]) -> bool:
        """Check if early stopping should be triggered."""
        if not self.config.training.early_stopping.enabled:
            return False

        metric_name = self.config.training.early_stopping.metric
        current_value = metrics.get(metric_name)

        if current_value is None:
            return False

        threshold = self.config.training.early_stopping.threshold
        patience = self.config.training.early_stopping.patience
        greater_is_better = self.config.training.early_stopping.greater_is_better

        # Check if metric improved
        if greater_is_better:
            improved = current_value > (self.state.best_wer + threshold)
        else:
            improved = current_value < (self.state.best_wer - threshold)

        if improved:
            self.state.patience_counter = 0
            if not greater_is_better:
                self.state.best_wer = current_value
        else:
            self.state.patience_counter += 1

        return self.state.patience_counter >= patience

    def _save_checkpoint(self, checkpoint_name: str) -> str:
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.paths.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"{checkpoint_name}.pt"

        checkpoint = {
            'epoch': self.state.epoch,
            'step': self.state.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_wer': self.state.best_wer,
            'config': OmegaConf.to_container(self.config, resolve=True)
        }

        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        return str(checkpoint_path)

    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.state.epoch = checkpoint['epoch']
        self.state.step = checkpoint['step']
        self.state.best_wer = checkpoint['best_wer']

        logger.info(f"Checkpoint loaded: {checkpoint_path}")

    def _export_model(self):
        """Export model for deployment."""
        export_dir = Path(self.config.paths.model_dir) / "exported"
        export_dir.mkdir(parents=True, exist_ok=True)

        # Export PyTorch model
        if "pytorch" in self.config.training.export.export_formats:
            torch.save(self.model.state_dict(), export_dir / "pytorch_model.bin")
            self.processor.save_pretrained(export_dir)
            logger.info(f"PyTorch model exported to {export_dir}")

        # Export ONNX model
        if "onnx" in self.config.training.export.export_formats:
            try:
                self._export_onnx(export_dir)
            except Exception as e:
                logger.warning(f"ONNX export failed: {e}")

    def _export_onnx(self, export_dir: Path):
        """Export model to ONNX format."""
        import torch.onnx

        self.model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, 16000, device=self.device)

        # Export
        torch.onnx.export(
            self.model,
            dummy_input,
            export_dir / "model.onnx",
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size', 1: 'sequence_length'}
            }
        )

        logger.info(f"ONNX model exported to {export_dir / 'model.onnx'}")

    def _log_training_step(self, loss: float, samples_per_second: float):
        """Log training step metrics."""
        if wandb.run:
            wandb.log({
                'train_step_loss': loss,
                'learning_rate': self.state.learning_rate,
                'grad_norm': self.state.grad_norm,
                'samples_per_second': samples_per_second,
                'step': self.state.step
            })

    def _log_epoch_metrics(self, metrics: Dict[str, float], epoch: int):
        """Log epoch metrics."""
        logger.info(f"Epoch {epoch + 1} metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")

        if wandb.run:
            wandb.log({**metrics, 'epoch': epoch})

    def _log_evaluation_metrics(self, metrics: Dict[str, float]):
        """Log evaluation metrics."""
        if wandb.run:
            wandb.log(metrics)

    def _log_test_metrics(self, metrics: Dict[str, float]):
        """Log final test metrics."""
        logger.info("Final test metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")

        if wandb.run:
            wandb.log(metrics)

    def _log_training_summary(self):
        """Log training summary."""
        logger.info("Training Summary:")
        logger.info(f"  Best WER: {self.state.best_wer:.4f}")
        logger.info(f"  Target WER: {self.config.project.target_wer:.4f}")
        logger.info(f"  Target achieved: {'✅' if self.state.best_wer < self.config.project.target_wer else '❌'}")
        logger.info(f"  Total epochs: {self.state.epoch + 1}")
        logger.info(f"  Total steps: {self.state.step}")

    def _is_main_process(self) -> bool:
        """Check if this is the main process in distributed training."""
        return not self.is_distributed or dist.get_rank() == 0


# Utility functions for standalone usage
def create_trainer(config_path: str) -> TwiSpeechTrainer:
    """Create trainer from config file."""
    with hydra.initialize(config_path="../configs"):
        cfg = hydra.compose(config_name="config")
    return TwiSpeechTrainer(cfg)


def train_model(config_path: str):
    """Train model from config file."""
    trainer = create_trainer(config_path)
    trainer.train()


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Train Twi Speech Recognition Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    args = parser.parse_args()

    train_model(args.config)
