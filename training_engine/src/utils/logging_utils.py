"""
Logging Utilities for Twi Speech Recognition Training Engine

This module provides comprehensive logging utilities for training Twi speech
recognition models. It includes structured logging, performance monitoring,
system information logging, and integration with external logging services.

Key Features:
- Structured logging with configurable levels and formats
- Performance and memory monitoring
- System information logging
- Integration with Weights & Biases and TensorBoard
- Log file rotation and management
- Debug utilities for model training
- Custom formatters for different output targets

Author: Twi Speech Recognition Team
"""

import logging
import sys
import os
import time
import psutil
import platform
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
import json
import threading
from collections import defaultdict

import torch
import numpy as np

# Rich logging for better console output
try:
    from rich.logging import RichHandler
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

logger = logging.getLogger(__name__)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']

        # Format timestamp
        record.asctime = self.formatTime(record, self.datefmt)

        # Color the level name
        record.colored_levelname = f"{log_color}{record.levelname}{reset_color}"

        # Create the formatted message
        formatter = logging.Formatter(
            '%(asctime)s - %(colored_levelname)s - %(name)s - %(message)s'
        )

        return formatter.format(record)


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class PerformanceLogger:
    """Logger for performance metrics during training"""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_times = {}
        self.lock = threading.Lock()

    def start_timer(self, name: str):
        """Start timing an operation"""
        with self.lock:
            self.start_times[name] = time.time()

    def end_timer(self, name: str) -> float:
        """End timing an operation and log the duration"""
        with self.lock:
            if name in self.start_times:
                duration = time.time() - self.start_times[name]
                self.metrics[f'{name}_duration'].append(duration)
                del self.start_times[name]
                return duration
            return 0.0

    def log_metric(self, name: str, value: float):
        """Log a performance metric"""
        with self.lock:
            self.metrics[name].append(value)

    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        with self.lock:
            values = self.metrics.get(name, [])
            if not values:
                return {}

            return {
                'count': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'total': np.sum(values)
            }

    def reset_metrics(self):
        """Reset all metrics"""
        with self.lock:
            self.metrics.clear()
            self.start_times.clear()


class SystemMonitor:
    """Monitor system resources during training"""

    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB

    def get_system_info(self) -> Dict[str, Any]:
        """Get current system information"""
        info = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'disk_usage_percent': psutil.disk_usage('/').percent,
            'process_memory_mb': self.process.memory_info().rss / 1024 / 1024,
            'process_cpu_percent': self.process.cpu_percent(),
        }

        # GPU information if available
        if torch.cuda.is_available():
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            info['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
            info['gpu_memory_percent'] = (
                torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                if torch.cuda.max_memory_allocated() > 0 else 0
            )

        return info

    def log_system_status(self):
        """Log current system status"""
        info = self.get_system_info()

        logger.info("System Status:")
        logger.info(f"  CPU Usage: {info['cpu_percent']:.1f}%")
        logger.info(f"  Memory Usage: {info['memory_percent']:.1f}%")
        logger.info(f"  Available Memory: {info['memory_available_gb']:.1f} GB")
        logger.info(f"  Process Memory: {info['process_memory_mb']:.1f} MB")

        if 'gpu_count' in info:
            logger.info(f"  GPU Count: {info['gpu_count']}")
            logger.info(f"  GPU Memory Allocated: {info['gpu_memory_allocated_mb']:.1f} MB")
            logger.info(f"  GPU Memory Reserved: {info['gpu_memory_reserved_mb']:.1f} MB")


class TrainingLogger:
    """Specialized logger for training progress"""

    def __init__(self):
        self.epoch_start_time = None
        self.step_start_time = None
        self.performance_logger = PerformanceLogger()
        self.system_monitor = SystemMonitor()

    def log_training_start(self, config: Dict[str, Any]):
        """Log training start information"""
        logger.info("=" * 80)
        logger.info("TRAINING STARTED")
        logger.info("=" * 80)

        logger.info(f"Project: {config.get('project', {}).get('name', 'Twi Speech Recognition')}")
        logger.info(f"Experiment: {config.get('experiment', {}).get('name', 'default')}")
        logger.info(f"Target WER: {config.get('project', {}).get('target_wer', 0.3):.1%}")

        # Log configuration summary
        if 'training' in config:
            training_config = config['training']
            logger.info(f"Epochs: {training_config.get('num_epochs', 'N/A')}")
            logger.info(f"Batch Size: {training_config.get('per_device_train_batch_size', 'N/A')}")
            logger.info(f"Learning Rate: {training_config.get('learning_rate', 'N/A')}")

    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start"""
        self.epoch_start_time = time.time()
        logger.info(f"Epoch {epoch + 1}/{total_epochs} started")

    def log_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch end with metrics"""
        if self.epoch_start_time:
            epoch_duration = time.time() - self.epoch_start_time
            self.performance_logger.log_metric('epoch_duration', epoch_duration)

            logger.info(f"Epoch {epoch + 1} completed in {epoch_duration:.2f}s")

            # Log key metrics
            if 'eval_wer' in metrics:
                logger.info(f"  WER: {metrics['eval_wer']:.4f}")
            if 'eval_cer' in metrics:
                logger.info(f"  CER: {metrics['eval_cer']:.4f}")
            if 'train_loss' in metrics:
                logger.info(f"  Train Loss: {metrics['train_loss']:.4f}")

    def log_step(self, step: int, loss: float, lr: float, grad_norm: Optional[float] = None):
        """Log training step"""
        if step % 100 == 0:  # Log every 100 steps
            log_msg = f"Step {step}: loss={loss:.4f}, lr={lr:.2e}"
            if grad_norm is not None:
                log_msg += f", grad_norm={grad_norm:.4f}"
            logger.info(log_msg)

    def log_training_end(self, best_wer: float, target_wer: float):
        """Log training completion"""
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 80)

        logger.info(f"Best WER: {best_wer:.4f}")
        logger.info(f"Target WER: {target_wer:.4f}")

        if best_wer < target_wer:
            logger.info("🎉 TARGET ACHIEVED!")
        else:
            improvement_needed = best_wer - target_wer
            logger.info(f"❌ Target not met. Need {improvement_needed:.1%} improvement.")

    def log_performance_summary(self):
        """Log performance summary"""
        logger.info("Performance Summary:")

        # Epoch duration stats
        epoch_stats = self.performance_logger.get_stats('epoch_duration')
        if epoch_stats:
            logger.info(f"  Average epoch time: {epoch_stats['mean']:.2f}s")
            logger.info(f"  Total training time: {epoch_stats['total']:.2f}s")

        # System resources
        self.system_monitor.log_system_status()


def setup_logging(config: Dict[str, Any]):
    """
    Setup logging configuration based on config

    Args:
        config: Logging configuration dictionary
    """
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set logging level
    level_str = config.get('level', 'INFO').upper()
    level = getattr(logging, level_str, logging.INFO)

    # Create formatters
    handlers = []

    # Console handler
    if RICH_AVAILABLE and config.get('use_rich', True):
        # Use Rich handler for beautiful console output
        console_handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            show_time=True
        )
        console_handler.setLevel(level)
        handlers.append(console_handler)
    else:
        # Use colored formatter for regular console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        if config.get('use_colors', True) and sys.stdout.isatty():
            console_handler.setFormatter(ColoredFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
            )
        handlers.append(console_handler)

    # File handler
    if 'log_dir' in config and config['log_dir']:
        log_dir = Path(config['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)

        # Main log file
        log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)

        if config.get('structured_logging', False):
            file_handler.setFormatter(StructuredFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
            )
        handlers.append(file_handler)

        # Error log file
        error_file = log_dir / f"errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        error_handler = logging.FileHandler(error_file, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        )
        handlers.append(error_handler)

    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True
    )

    # Set specific logger levels
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('datasets').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

    logger.info(f"Logging configured with level {level_str}")
    if 'log_dir' in config and config['log_dir']:
        logger.info(f"Log files will be saved to: {config['log_dir']}")


def log_system_info():
    """Log comprehensive system information"""
    logger.info("System Information:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python Version: {platform.python_version()}")
    logger.info(f"  CPU Count: {psutil.cpu_count()}")
    logger.info(f"  Total Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")

    # PyTorch information
    logger.info(f"  PyTorch Version: {torch.__version__}")
    logger.info(f"  CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"  CUDA Version: {torch.version.cuda}")
        logger.info(f"  GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"    GPU {i}: {props.name} ({props.total_memory / (1024**3):.1f} GB)")

    # Environment variables
    relevant_env_vars = ['CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']
    for var in relevant_env_vars:
        value = os.environ.get(var, 'not set')
        logger.info(f"  {var}: {value}")


def create_progress_bar(description: str = "Processing", total: Optional[int] = None):
    """Create a progress bar for long-running operations"""
    if RICH_AVAILABLE:
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        )
    else:
        # Fallback to simple logging
        class SimpleProgress:
            def __init__(self, description):
                self.description = description

            def add_task(self, description, total=None):
                logger.info(f"Starting: {description}")
                return 0

            def update(self, task_id, advance=1):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                logger.info(f"Completed: {self.description}")

        return SimpleProgress(description)


def log_model_info(model: torch.nn.Module, model_name: str = "Model"):
    """Log detailed model information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"{model_name} Information:")
    logger.info(f"  Architecture: {model.__class__.__name__}")
    logger.info(f"  Total Parameters: {total_params:,}")
    logger.info(f"  Trainable Parameters: {trainable_params:,}")
    logger.info(f"  Frozen Parameters: {total_params - trainable_params:,}")

    # Model size estimation
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)
    logger.info(f"  Estimated Size: {model_size_mb:.2f} MB")


def log_config_summary(config: Dict[str, Any]):
    """Log a summary of the training configuration"""
    logger.info("Configuration Summary:")

    # Project info
    if 'project' in config:
        project = config['project']
        logger.info(f"  Project: {project.get('name', 'N/A')}")
        logger.info(f"  Version: {project.get('version', 'N/A')}")
        logger.info(f"  Target WER: {project.get('target_wer', 'N/A')}")

    # Training parameters
    if 'training' in config:
        training = config['training']
        logger.info(f"  Epochs: {training.get('num_epochs', 'N/A')}")
        logger.info(f"  Batch Size: {training.get('per_device_train_batch_size', 'N/A')}")
        logger.info(f"  Learning Rate: {training.get('learning_rate', 'N/A')}")
        logger.info(f"  Optimizer: {training.get('optimizer', 'N/A')}")

    # Model info
    if 'model' in config:
        model = config['model']
        logger.info(f"  Model: {model.get('_target_', 'N/A')}")
        if 'pretrained_model_name' in model:
            logger.info(f"  Pretrained: {model['pretrained_model_name']}")


class TimingContext:
    """Context manager for timing operations"""

    def __init__(self, operation_name: str, performance_logger: Optional[PerformanceLogger] = None):
        self.operation_name = operation_name
        self.performance_logger = performance_logger
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        logger.debug(f"Starting: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            logger.debug(f"Completed: {self.operation_name} in {duration:.2f}s")

            if self.performance_logger:
                self.performance_logger.log_metric(f"{self.operation_name}_duration", duration)


# Global instances
performance_logger = PerformanceLogger()
system_monitor = SystemMonitor()
training_logger = TrainingLogger()


# Convenience functions
def time_operation(operation_name: str):
    """Decorator to time function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with TimingContext(operation_name, performance_logger):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def log_exception(exc_info=None):
    """Log exception with full traceback"""
    logger.exception("An error occurred:", exc_info=exc_info)


def log_warning_once(message: str, category: str = "general"):
    """Log a warning message only once per category"""
    if not hasattr(log_warning_once, 'logged_warnings'):
        log_warning_once.logged_warnings = set()

    warning_key = f"{category}:{message}"
    if warning_key not in log_warning_once.logged_warnings:
        logger.warning(message)
        log_warning_once.logged_warnings.add(warning_key)


if __name__ == "__main__":
    # Example usage
    config = {
        'level': 'INFO',
        'log_dir': 'logs',
        'use_rich': True,
        'use_colors': True,
        'structured_logging': False
    }

    setup_logging(config)
    log_system_info()

    logger.info("Logging utilities test completed!")
