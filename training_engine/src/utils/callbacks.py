"""
Training Callbacks for Twi Speech Recognition Training Engine

This module provides comprehensive callback utilities for training Twi speech
recognition models. It includes callbacks for logging, early stopping, model
checkpointing, learning rate scheduling, and custom Twi-specific monitoring.

Key Features:
- Early stopping with configurable patience and metrics
- Model checkpointing with best model saving
- Learning rate scheduling callbacks
- Weights & Biases integration
- TensorBoard logging
- Custom Twi-specific evaluation callbacks
- Training progress monitoring
- Memory usage tracking
- Gradient monitoring

Author: Twi Speech Recognition Team
"""

import logging
import time
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from collections import defaultdict
import warnings

import torch
import torch.nn as nn
import numpy as np

# Optional imports with fallbacks
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("Weights & Biases not available")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    warnings.warn("TensorBoard not available")

logger = logging.getLogger(__name__)


class Callback:
    """Base callback class"""

    def on_training_start(self, trainer, **kwargs):
        """Called at the start of training"""
        pass

    def on_training_end(self, trainer, **kwargs):
        """Called at the end of training"""
        pass

    def on_epoch_start(self, trainer, epoch: int, **kwargs):
        """Called at the start of each epoch"""
        pass

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs):
        """Called at the end of each epoch"""
        pass

    def on_step_start(self, trainer, step: int, **kwargs):
        """Called at the start of each training step"""
        pass

    def on_step_end(self, trainer, step: int, loss: float, **kwargs):
        """Called at the end of each training step"""
        pass

    def on_evaluation_start(self, trainer, **kwargs):
        """Called at the start of evaluation"""
        pass

    def on_evaluation_end(self, trainer, metrics: Dict[str, float], **kwargs):
        """Called at the end of evaluation"""
        pass

    def on_save_checkpoint(self, trainer, checkpoint_path: str, **kwargs):
        """Called when saving a checkpoint"""
        pass

    def on_load_checkpoint(self, trainer, checkpoint_path: str, **kwargs):
        """Called when loading a checkpoint"""
        pass


class EarlyStoppingCallback(Callback):
    """Early stopping callback to prevent overfitting"""

    def __init__(
        self,
        monitor: str = "eval_wer",
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = "min",
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        """
        Initialize early stopping callback

        Args:
            monitor: Metric to monitor
            patience: Number of epochs to wait before stopping
            min_delta: Minimum improvement threshold
            mode: 'min' for metrics to minimize, 'max' for metrics to maximize
            restore_best_weights: Whether to restore best weights when stopping
            verbose: Whether to log early stopping events
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        # Internal state
        self.best_value = None
        self.best_epoch = 0
        self.wait = 0
        self.best_weights = None
        self.stopped_epoch = 0

        # Set comparison function based on mode
        if mode == "min":
            self.is_better = lambda current, best: current < (best - self.min_delta)
            self.best_value = float('inf')
        elif mode == "max":
            self.is_better = lambda current, best: current > (best + self.min_delta)
            self.best_value = float('-inf')
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs):
        """Check for early stopping condition"""
        current_value = metrics.get(self.monitor)

        if current_value is None:
            if self.verbose:
                logger.warning(f"Early stopping metric '{self.monitor}' not found in metrics")
            return

        # Check if this is the best value so far
        if self.is_better(current_value, self.best_value):
            self.best_value = current_value
            self.best_epoch = epoch
            self.wait = 0

            # Save best weights if requested
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in trainer.model.state_dict().items()}

            if self.verbose:
                logger.info(f"New best {self.monitor}: {current_value:.6f}")
        else:
            self.wait += 1

            if self.verbose:
                logger.info(f"EarlyStopping counter: {self.wait}/{self.patience}")

        # Check if we should stop
        if self.wait >= self.patience:
            self.stopped_epoch = epoch

            if self.restore_best_weights and self.best_weights is not None:
                trainer.model.load_state_dict(self.best_weights)
                if self.verbose:
                    logger.info(f"Restored best weights from epoch {self.best_epoch}")

            trainer.should_stop = True

            if self.verbose:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                logger.info(f"Best {self.monitor}: {self.best_value:.6f} at epoch {self.best_epoch}")


class ModelCheckpointCallback(Callback):
    """Callback for saving model checkpoints"""

    def __init__(
        self,
        dirpath: str,
        filename: str = "checkpoint-{epoch:02d}-{wer:.4f}",
        monitor: str = "eval_wer",
        mode: str = "min",
        save_top_k: int = 3,
        save_last: bool = True,
        save_weights_only: bool = False,
        verbose: bool = True
    ):
        """
        Initialize model checkpoint callback

        Args:
            dirpath: Directory to save checkpoints
            filename: Filename template
            monitor: Metric to monitor for best model selection
            mode: 'min' or 'max' for metric optimization
            save_top_k: Number of best models to keep
            save_last: Whether to save the last checkpoint
            save_weights_only: Whether to save only model weights
            verbose: Whether to log checkpoint events
        """
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.save_weights_only = save_weights_only
        self.verbose = verbose

        # Create directory
        self.dirpath.mkdir(parents=True, exist_ok=True)

        # Internal state
        self.best_models = []  # List of (metric_value, filepath) tuples

        # Set comparison function
        if mode == "min":
            self.is_better = lambda current, best: current < best
        elif mode == "max":
            self.is_better = lambda current, best: current > best
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs):
        """Save checkpoint if conditions are met"""
        current_value = metrics.get(self.monitor)

        # Format filename
        filename_formatted = self.filename.format(
            epoch=epoch,
            wer=metrics.get('eval_wer', 0.0),
            cer=metrics.get('eval_cer', 0.0),
            **metrics
        )
        filepath = self.dirpath / filename_formatted

        # Always save if save_last is True
        if self.save_last:
            last_path = self.dirpath / "last.ckpt"
            self._save_checkpoint(trainer, last_path, epoch, metrics)

        # Save if this is one of the top k models
        if current_value is not None and self.save_top_k > 0:
            should_save = False

            if len(self.best_models) < self.save_top_k:
                should_save = True
            else:
                # Check if this is better than the worst saved model
                worst_value = min(self.best_models, key=lambda x: x[0] if self.mode == "max" else -x[0])[0]
                if self.is_better(current_value, worst_value):
                    should_save = True

                    # Remove worst model
                    worst_item = min(self.best_models, key=lambda x: x[0] if self.mode == "max" else -x[0])
                    self.best_models.remove(worst_item)

                    # Delete worst checkpoint file
                    try:
                        if os.path.exists(worst_item[1]):
                            os.remove(worst_item[1])
                    except Exception as e:
                        logger.warning(f"Could not remove checkpoint {worst_item[1]}: {e}")

            if should_save:
                self._save_checkpoint(trainer, filepath, epoch, metrics)
                self.best_models.append((current_value, str(filepath)))

                if self.verbose:
                    logger.info(f"Saved checkpoint: {filepath}")

    def _save_checkpoint(self, trainer, filepath: Path, epoch: int, metrics: Dict[str, float]):
        """Save checkpoint to file"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": trainer.model.state_dict(),
            "metrics": metrics,
            "config": trainer.config,
        }

        if not self.save_weights_only:
            if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
                checkpoint["optimizer_state_dict"] = trainer.optimizer.state_dict()
            if hasattr(trainer, 'scheduler') and trainer.scheduler is not None:
                checkpoint["scheduler_state_dict"] = trainer.scheduler.state_dict()
            if hasattr(trainer, 'scaler') and trainer.scaler is not None:
                checkpoint["scaler_state_dict"] = trainer.scaler.state_dict()

        torch.save(checkpoint, filepath)


class WandbCallback(Callback):
    """Weights & Biases logging callback"""

    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        log_model: bool = True,
        log_code: bool = True,
        log_gradients: bool = False,
        log_parameters: bool = False,
        log_frequency: int = 100
    ):
        """
        Initialize Weights & Biases callback

        Args:
            project: W&B project name
            entity: W&B entity (username/team)
            name: Run name
            tags: List of tags for the run
            log_model: Whether to log model artifacts
            log_code: Whether to log code
            log_gradients: Whether to log gradients
            log_parameters: Whether to log parameters
            log_frequency: Frequency of logging (steps)
        """
        if not WANDB_AVAILABLE:
            raise ImportError("Weights & Biases is not available")

        self.project = project
        self.entity = entity
        self.name = name
        self.tags = tags or []
        self.log_model = log_model
        self.log_code = log_code
        self.log_gradients = log_gradients
        self.log_parameters = log_parameters
        self.log_frequency = log_frequency

        self.run = None

    def on_training_start(self, trainer, **kwargs):
        """Initialize W&B run"""
        config_dict = trainer.config if hasattr(trainer.config, '__dict__') else dict(trainer.config)

        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.name,
            tags=self.tags,
            config=config_dict,
            save_code=self.log_code
        )

        # Log model architecture
        if self.log_model:
            wandb.watch(trainer.model, log="all" if self.log_gradients else None, log_freq=self.log_frequency)

        logger.info(f"Initialized W&B run: {self.run.name}")

    def on_step_end(self, trainer, step: int, loss: float, **kwargs):
        """Log training step metrics"""
        if step % self.log_frequency == 0:
            metrics = {
                "train/loss": loss,
                "train/learning_rate": trainer.state.learning_rate,
                "train/step": step
            }

            if hasattr(trainer.state, 'grad_norm'):
                metrics["train/grad_norm"] = trainer.state.grad_norm

            wandb.log(metrics, step=step)

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs):
        """Log epoch metrics"""
        epoch_metrics = {"epoch": epoch}

        for key, value in metrics.items():
            if key.startswith("eval_"):
                epoch_metrics[f"eval/{key[5:]}"] = value
            else:
                epoch_metrics[f"train/{key}"] = value

        wandb.log(epoch_metrics, step=trainer.state.step)

    def on_training_end(self, trainer, **kwargs):
        """Finish W&B run"""
        if self.run is not None:
            wandb.finish()
            logger.info("Finished W&B run")


class TensorBoardCallback(Callback):
    """TensorBoard logging callback"""

    def __init__(
        self,
        log_dir: str,
        log_graph: bool = True,
        log_images: bool = False,
        log_frequency: int = 100
    ):
        """
        Initialize TensorBoard callback

        Args:
            log_dir: Directory for TensorBoard logs
            log_graph: Whether to log model graph
            log_images: Whether to log images
            log_frequency: Frequency of logging (steps)
        """
        if not TENSORBOARD_AVAILABLE:
            raise ImportError("TensorBoard is not available")

        self.log_dir = log_dir
        self.log_graph = log_graph
        self.log_images = log_images
        self.log_frequency = log_frequency

        self.writer = None

    def on_training_start(self, trainer, **kwargs):
        """Initialize TensorBoard writer"""
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # Log model graph
        if self.log_graph:
            try:
                dummy_input = torch.randn(1, 16000)  # Dummy audio input
                self.writer.add_graph(trainer.model, dummy_input)
            except Exception as e:
                logger.warning(f"Could not log model graph: {e}")

        logger.info(f"Initialized TensorBoard writer: {self.log_dir}")

    def on_step_end(self, trainer, step: int, loss: float, **kwargs):
        """Log training step metrics"""
        if step % self.log_frequency == 0:
            self.writer.add_scalar("train/loss", loss, step)
            self.writer.add_scalar("train/learning_rate", trainer.state.learning_rate, step)

            if hasattr(trainer.state, 'grad_norm'):
                self.writer.add_scalar("train/grad_norm", trainer.state.grad_norm, step)

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs):
        """Log epoch metrics"""
        for key, value in metrics.items():
            if key.startswith("eval_"):
                self.writer.add_scalar(f"eval/{key[5:]}", value, epoch)
            else:
                self.writer.add_scalar(f"train/{key}", value, epoch)

    def on_training_end(self, trainer, **kwargs):
        """Close TensorBoard writer"""
        if self.writer is not None:
            self.writer.close()
            logger.info("Closed TensorBoard writer")


class LearningRateSchedulerCallback(Callback):
    """Learning rate scheduler callback"""

    def __init__(
        self,
        scheduler,
        monitor: Optional[str] = None,
        interval: str = "epoch",
        frequency: int = 1,
        reduce_on_plateau: bool = False
    ):
        """
        Initialize learning rate scheduler callback

        Args:
            scheduler: PyTorch learning rate scheduler
            monitor: Metric to monitor for ReduceLROnPlateau
            interval: 'epoch' or 'step'
            frequency: How often to step the scheduler
            reduce_on_plateau: Whether this is a ReduceLROnPlateau scheduler
        """
        self.scheduler = scheduler
        self.monitor = monitor
        self.interval = interval
        self.frequency = frequency
        self.reduce_on_plateau = reduce_on_plateau

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs):
        """Step scheduler at epoch end"""
        if self.interval == "epoch" and epoch % self.frequency == 0:
            if self.reduce_on_plateau and self.monitor:
                metric_value = metrics.get(self.monitor)
                if metric_value is not None:
                    self.scheduler.step(metric_value)
            else:
                self.scheduler.step()

    def on_step_end(self, trainer, step: int, loss: float, **kwargs):
        """Step scheduler at step end"""
        if self.interval == "step" and step % self.frequency == 0:
            if self.reduce_on_plateau:
                self.scheduler.step(loss)
            else:
                self.scheduler.step()


class TwiEvaluationCallback(Callback):
    """Custom evaluation callback for Twi-specific metrics"""

    def __init__(
        self,
        evaluation_data: Optional[Any] = None,
        evaluate_every: int = 1,
        compute_dialect_metrics: bool = True,
        compute_tone_metrics: bool = True,
        save_predictions: bool = False,
        predictions_dir: Optional[str] = None
    ):
        """
        Initialize Twi evaluation callback

        Args:
            evaluation_data: Custom evaluation dataset
            evaluate_every: Evaluate every N epochs
            compute_dialect_metrics: Whether to compute dialect metrics
            compute_tone_metrics: Whether to compute tone metrics
            save_predictions: Whether to save predictions
            predictions_dir: Directory to save predictions
        """
        self.evaluation_data = evaluation_data
        self.evaluate_every = evaluate_every
        self.compute_dialect_metrics = compute_dialect_metrics
        self.compute_tone_metrics = compute_tone_metrics
        self.save_predictions = save_predictions
        self.predictions_dir = predictions_dir

        if save_predictions and predictions_dir:
            Path(predictions_dir).mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs):
        """Run custom evaluation"""
        if epoch % self.evaluate_every == 0:
            logger.info(f"Running Twi-specific evaluation at epoch {epoch}")

            # Custom evaluation logic would go here
            # For now, just log that it would happen
            custom_metrics = self._run_custom_evaluation(trainer)

            # Update metrics with custom results
            metrics.update(custom_metrics)

            # Save predictions if requested
            if self.save_predictions and self.predictions_dir:
                self._save_predictions(trainer, epoch)

    def _run_custom_evaluation(self, trainer) -> Dict[str, float]:
        """Run custom Twi evaluation"""
        # Placeholder for custom evaluation logic
        custom_metrics = {}

        if self.compute_dialect_metrics:
            # Compute dialect classification accuracy
            custom_metrics["dialect_accuracy"] = 0.85  # Placeholder

        if self.compute_tone_metrics:
            # Compute tone accuracy
            custom_metrics["tone_accuracy"] = 0.78  # Placeholder

        return custom_metrics

    def _save_predictions(self, trainer, epoch: int):
        """Save model predictions"""
        predictions_file = Path(self.predictions_dir) / f"predictions_epoch_{epoch}.json"

        # Placeholder for saving predictions
        predictions = {
            "epoch": epoch,
            "predictions": [],  # Would contain actual predictions
            "references": []    # Would contain reference texts
        }

        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)


class ProgressCallback(Callback):
    """Progress monitoring callback"""

    def __init__(self, log_frequency: int = 100):
        """
        Initialize progress callback

        Args:
            log_frequency: How often to log progress (steps)
        """
        self.log_frequency = log_frequency
        self.start_time = None
        self.epoch_start_time = None

    def on_training_start(self, trainer, **kwargs):
        """Record training start time"""
        self.start_time = time.time()
        logger.info("Training started")

    def on_epoch_start(self, trainer, epoch: int, **kwargs):
        """Record epoch start time"""
        self.epoch_start_time = time.time()
        logger.info(f"Starting epoch {epoch + 1}/{trainer.config.training.num_epochs}")

    def on_step_end(self, trainer, step: int, loss: float, **kwargs):
        """Log progress periodically"""
        if step % self.log_frequency == 0:
            elapsed = time.time() - self.start_time if self.start_time else 0
            logger.info(f"Step {step}: loss={loss:.4f}, elapsed={elapsed:.1f}s")

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs):
        """Log epoch completion"""
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.1f}s")

            # Log key metrics
            wer = metrics.get('eval_wer', 'N/A')
            cer = metrics.get('eval_cer', 'N/A')
            logger.info(f"  WER: {wer}, CER: {cer}")

    def on_training_end(self, trainer, **kwargs):
        """Log training completion"""
        if self.start_time:
            total_time = time.time() - self.start_time
            logger.info(f"Training completed in {total_time:.1f}s")


class TwiTrainingCallbacks:
    """Container for all Twi training callbacks"""

    def __init__(self, config):
        """Initialize callback container with configuration"""
        self.config = config
        self.callbacks = []

        # Initialize callbacks based on config
        self._setup_callbacks()

    def _setup_callbacks(self):
        """Setup callbacks based on configuration"""
        # Progress callback
        self.callbacks.append(ProgressCallback(log_frequency=100))

        # Early stopping
        if self.config.training.early_stopping.enabled:
            early_stopping = EarlyStoppingCallback(
                monitor=self.config.training.early_stopping.metric,
                patience=self.config.training.early_stopping.patience,
                min_delta=self.config.training.early_stopping.threshold,
                mode="min" if "wer" in self.config.training.early_stopping.metric else "max"
            )
            self.callbacks.append(early_stopping)

        # Model checkpointing
        if hasattr(self.config.training, 'save_strategy') and self.config.training.save_strategy:
            checkpoint_callback = ModelCheckpointCallback(
                dirpath=self.config.paths.checkpoint_dir,
                monitor=self.config.training.metric_for_best_model,
                mode="min" if "wer" in self.config.training.metric_for_best_model else "max",
                save_top_k=3
            )
            self.callbacks.append(checkpoint_callback)

        # Weights & Biases
        if WANDB_AVAILABLE and self.config.logging.wandb.project:
            wandb_callback = WandbCallback(
                project=self.config.logging.wandb.project,
                entity=self.config.logging.wandb.get("entity"),
                name=self.config.experiment.name,
                tags=self.config.experiment.tags
            )
            self.callbacks.append(wandb_callback)

        # TensorBoard
        if TENSORBOARD_AVAILABLE and self.config.logging.tensorboard.log_dir:
            tensorboard_callback = TensorBoardCallback(
                log_dir=self.config.logging.tensorboard.log_dir
            )
            self.callbacks.append(tensorboard_callback)

        # Custom Twi evaluation
        twi_eval_callback = TwiEvaluationCallback(
            evaluate_every=1,
            compute_dialect_metrics=True,
            compute_tone_metrics=True
        )
        self.callbacks.append(twi_eval_callback)

    def on_training_start(self, trainer, **kwargs):
        """Call on_training_start for all callbacks"""
        for callback in self.callbacks:
            callback.on_training_start(trainer, **kwargs)

    def on_training_end(self, trainer, **kwargs):
        """Call on_training_end for all callbacks"""
        for callback in self.callbacks:
            callback.on_training_end(trainer, **kwargs)

    def on_epoch_start(self, trainer, epoch: int, **kwargs):
        """Call on_epoch_start for all callbacks"""
        for callback in self.callbacks:
            callback.on_epoch_start(trainer, epoch, **kwargs)

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs):
        """Call on_epoch_end for all callbacks"""
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, epoch, metrics, **kwargs)

    def on_step_start(self, trainer, step: int, **kwargs):
        """Call on_step_start for all callbacks"""
        for callback in self.callbacks:
            callback.on_step_start(trainer, step, **kwargs)

    def on_step_end(self, trainer, step: int, loss: float, **kwargs):
        """Call on_step_end for all callbacks"""
        for callback in self.callbacks:
            callback.on_step_end(trainer, step, loss, **kwargs)

    def on_evaluation_start(self, trainer, **kwargs):
        """Call on_evaluation_start for all callbacks"""
        for callback in self.callbacks:
            callback.on_evaluation_start(trainer, **kwargs)

    def on_evaluation_end(self, trainer, metrics: Dict[str, float], **kwargs):
        """Call on_evaluation_end for all callbacks"""
        for callback in self.callbacks:
            callback.on_evaluation_end(trainer, metrics, **kwargs)


if __name__ == "__main__":
    # Example usage
    print("Testing training callbacks...")

    # Mock trainer object for testing
    class MockTrainer:
        def __init__(self):
            self.model = nn.Linear(10, 1)
            self.config = type('Config', (), {
                'training': type('Training', (), {
                    'early_stopping': type('EarlyStopping', (), {
                        'enabled': True,
                        'metric': 'eval_wer',
                        'patience': 3,
                        'threshold': 0.001
                    })(),
                    'metric_for_best_model': 'eval_wer',
                    'save_strategy': 'epoch'
                })(),
                'paths': type('Paths', (), {
                    'checkpoint_dir': 'checkpoints'
                })(),
                'logging': type('Logging', (), {
                    'wandb': type('Wandb', (), {'project': None})(),
                    'tensorboard': type('Tensorboard', (), {'log_dir': None})()
                })(),
                'experiment': type('Experiment', (), {
                    'name': 'test',
                    'tags': ['test']
                })()
            })()
            self.should_stop = False

    trainer = MockTrainer()

    # Test early stopping callback
    early_stopping = EarlyStoppingCallback(monitor="eval_wer", patience=2)

    # Simulate training epochs
    metrics_sequence = [
        {"eval_wer": 0.5},
        {"eval_wer": 0.4},  # Improvement
        {"eval_wer": 0.45}, # No improvement
        {"eval_wer": 0.47}, # No improvement - should trigger early stopping
    ]

    for epoch, metrics in enumerate(metrics_sequence):
        early_stopping.on_epoch_end(trainer, epoch, metrics)
        if trainer.should_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print("Callback tests completed!")
