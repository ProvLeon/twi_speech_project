"""
Model Utilities for Twi Speech Recognition Training

This module provides utility functions for model management, initialization,
optimization, and analysis during training. It includes tools for model
inspection, parameter counting, memory analysis, and optimization techniques
specific to speech recognition models.

Key Features:
- Model parameter analysis and counting
- Memory usage optimization
- Model initialization strategies
- Gradient analysis and clipping utilities
- Model compression and quantization tools
- Checkpoint management utilities
- Model architecture visualization

Author: Twi Speech Recognition Team
"""

import logging
import math
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import OrderedDict
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import numpy as np

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count the number of parameters in a model

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_model_size(model: nn.Module, unit: str = "MB") -> float:
    """
    Calculate model size in memory

    Args:
        model: PyTorch model
        unit: Unit for size (MB, KB, GB)

    Returns:
        Model size in specified unit
    """
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size = param_size + buffer_size

    if unit == "KB":
        return total_size / 1024
    elif unit == "MB":
        return total_size / (1024 ** 2)
    elif unit == "GB":
        return total_size / (1024 ** 3)
    else:
        return total_size


def analyze_model_architecture(model: nn.Module) -> Dict[str, Any]:
    """
    Analyze model architecture and provide detailed statistics

    Args:
        model: PyTorch model

    Returns:
        Dictionary with architecture analysis
    """
    analysis = {
        "total_parameters": count_parameters(model),
        "trainable_parameters": count_parameters(model, trainable_only=True),
        "model_size_mb": get_model_size(model, "MB"),
        "layer_count": 0,
        "layer_types": {},
        "parameter_distribution": {},
    }

    # Analyze layers
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            analysis["layer_count"] += 1

            layer_type = type(module).__name__
            if layer_type not in analysis["layer_types"]:
                analysis["layer_types"][layer_type] = 0
            analysis["layer_types"][layer_type] += 1

            # Parameter distribution by layer type
            layer_params = count_parameters(module)
            if layer_type not in analysis["parameter_distribution"]:
                analysis["parameter_distribution"][layer_type] = 0
            analysis["parameter_distribution"][layer_type] += layer_params

    # Calculate frozen parameters
    analysis["frozen_parameters"] = analysis["total_parameters"] - analysis["trainable_parameters"]
    analysis["trainable_ratio"] = analysis["trainable_parameters"] / max(analysis["total_parameters"], 1)

    return analysis


def init_weights(model: nn.Module, init_type: str = "xavier_uniform"):
    """
    Initialize model weights using specified strategy

    Args:
        model: PyTorch model
        init_type: Initialization strategy
    """
    def init_func(m):
        if isinstance(m, nn.Linear):
            if init_type == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight)
            elif init_type == "xavier_normal":
                nn.init.xavier_normal_(m.weight)
            elif init_type == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            elif init_type == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif init_type == "normal":
                nn.init.normal_(m.weight, std=0.02)

            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Conv1d):
            if init_type == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            elif init_type == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif init_type == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight)
            elif init_type == "xavier_normal":
                nn.init.xavier_normal_(m.weight)

            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    model.apply(init_func)
    logger.info(f"Initialized model weights using {init_type}")


def freeze_layers(model: nn.Module, layer_names: List[str]):
    """
    Freeze specified layers in the model

    Args:
        model: PyTorch model
        layer_names: List of layer names to freeze
    """
    frozen_count = 0
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = False
                frozen_count += 1
                break

    logger.info(f"Frozen {frozen_count} parameters in layers: {layer_names}")


def unfreeze_layers(model: nn.Module, layer_names: List[str]):
    """
    Unfreeze specified layers in the model

    Args:
        model: PyTorch model
        layer_names: List of layer names to unfreeze
    """
    unfrozen_count = 0
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = True
                unfrozen_count += 1
                break

    logger.info(f"Unfrozen {unfrozen_count} parameters in layers: {layer_names}")


def gradual_unfreezing(model: nn.Module, current_epoch: int, unfreeze_schedule: Dict[int, List[str]]):
    """
    Gradually unfreeze layers based on training epoch

    Args:
        model: PyTorch model
        current_epoch: Current training epoch
        unfreeze_schedule: Dictionary mapping epochs to layer names to unfreeze
    """
    if current_epoch in unfreeze_schedule:
        layers_to_unfreeze = unfreeze_schedule[current_epoch]
        unfreeze_layers(model, layers_to_unfreeze)
        logger.info(f"Epoch {current_epoch}: Unfroze layers {layers_to_unfreeze}")


def analyze_gradients(model: nn.Module) -> Dict[str, float]:
    """
    Analyze gradient statistics for debugging

    Args:
        model: PyTorch model

    Returns:
        Dictionary with gradient statistics
    """
    grad_stats = {
        "total_norm": 0.0,
        "max_grad": 0.0,
        "min_grad": float('inf'),
        "num_zero_grads": 0,
        "num_none_grads": 0,
        "layer_grad_norms": {},
    }

    total_norm = 0.0
    param_count = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            total_norm += grad_norm ** 2

            grad_stats["max_grad"] = max(grad_stats["max_grad"], param.grad.data.abs().max().item())
            grad_stats["min_grad"] = min(grad_stats["min_grad"], param.grad.data.abs().min().item())
            grad_stats["layer_grad_norms"][name] = grad_norm

            if torch.all(param.grad.data == 0):
                grad_stats["num_zero_grads"] += 1

            param_count += 1
        else:
            grad_stats["num_none_grads"] += 1

    grad_stats["total_norm"] = math.sqrt(total_norm)
    grad_stats["avg_grad_norm"] = grad_stats["total_norm"] / max(param_count, 1)

    if grad_stats["min_grad"] == float('inf'):
        grad_stats["min_grad"] = 0.0

    return grad_stats


def clip_gradients(model: nn.Module, max_norm: float = 1.0, clip_type: str = "norm") -> float:
    """
    Clip gradients to prevent exploding gradients

    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm or value
        clip_type: Type of clipping ('norm' or 'value')

    Returns:
        Gradient norm before clipping
    """
    if clip_type == "norm":
        grad_norm = clip_grad_norm_(model.parameters(), max_norm)
        return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
    elif clip_type == "value":
        clip_grad_value_(model.parameters(), max_norm)
        # Calculate norm for logging
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return math.sqrt(total_norm)
    else:
        raise ValueError(f"Unknown clip_type: {clip_type}")


def get_parameter_groups(model: nn.Module, lr_multipliers: Optional[Dict[str, float]] = None) -> List[Dict]:
    """
    Create parameter groups with different learning rates

    Args:
        model: PyTorch model
        lr_multipliers: Dictionary mapping layer patterns to LR multipliers

    Returns:
        List of parameter groups for optimizer
    """
    if lr_multipliers is None:
        return [{"params": model.parameters()}]

    param_groups = []
    assigned_params = set()

    # Create groups for specified patterns
    for pattern, multiplier in lr_multipliers.items():
        group_params = []
        for name, param in model.named_parameters():
            if pattern in name and param not in assigned_params:
                group_params.append(param)
                assigned_params.add(param)

        if group_params:
            param_groups.append({
                "params": group_params,
                "lr_multiplier": multiplier,
                "name": pattern
            })

    # Add remaining parameters to default group
    remaining_params = []
    for param in model.parameters():
        if param not in assigned_params:
            remaining_params.append(param)

    if remaining_params:
        param_groups.append({
            "params": remaining_params,
            "lr_multiplier": 1.0,
            "name": "default"
        })

    return param_groups


def enable_gradient_checkpointing(model: nn.Module):
    """
    Enable gradient checkpointing for memory efficiency

    Args:
        model: PyTorch model
    """
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing")
    else:
        logger.warning("Model does not support gradient checkpointing")


def model_summary(model: nn.Module, input_size: Optional[Tuple] = None,
                 device: str = "cpu") -> str:
    """
    Generate a detailed model summary

    Args:
        model: PyTorch model
        input_size: Input tensor size for forward pass analysis
        device: Device to run analysis on

    Returns:
        String summary of the model
    """
    analysis = analyze_model_architecture(model)

    summary = f"""
Model Summary
=============
Architecture: {type(model).__name__}
Total Parameters: {analysis['total_parameters']:,}
Trainable Parameters: {analysis['trainable_parameters']:,}
Frozen Parameters: {analysis['frozen_parameters']:,}
Trainable Ratio: {analysis['trainable_ratio']:.1%}
Model Size: {analysis['model_size_mb']:.2f} MB
Total Layers: {analysis['layer_count']}

Layer Types:
"""

    for layer_type, count in analysis['layer_types'].items():
        summary += f"  {layer_type}: {count}\n"

    summary += "\nParameter Distribution by Layer Type:\n"
    for layer_type, params in analysis['parameter_distribution'].items():
        percentage = (params / analysis['total_parameters']) * 100
        summary += f"  {layer_type}: {params:,} ({percentage:.1f}%)\n"

    # If input size provided, analyze memory usage
    if input_size is not None:
        try:
            model = model.to(device)
            dummy_input = torch.randn(1, *input_size).to(device)

            # Estimate memory usage
            with torch.no_grad():
                _ = model(dummy_input)

            if device == "cuda":
                memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                summary += f"\nEstimated GPU Memory Usage: {memory_allocated:.2f} MB\n"

        except Exception as e:
            logger.warning(f"Could not analyze memory usage: {e}")

    return summary


def save_model_checkpoint(model: nn.Module, optimizer: Optional[Optimizer],
                         epoch: int, loss: float, metrics: Dict[str, float],
                         filepath: Union[str, Path], metadata: Optional[Dict] = None):
    """
    Save model checkpoint with comprehensive information

    Args:
        model: PyTorch model
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        metrics: Dictionary of metrics
        filepath: Path to save checkpoint
        metadata: Additional metadata to save
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "loss": loss,
        "metrics": metrics,
        "model_architecture": type(model).__name__,
        "total_parameters": count_parameters(model),
        "trainable_parameters": count_parameters(model, trainable_only=True),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if metadata is not None:
        checkpoint["metadata"] = metadata

    # Save model architecture analysis
    checkpoint["architecture_analysis"] = analyze_model_architecture(model)

    torch.save(checkpoint, filepath)
    logger.info(f"Saved checkpoint to {filepath}")


def load_model_checkpoint(filepath: Union[str, Path], model: nn.Module,
                         optimizer: Optional[Optimizer] = None,
                         map_location: Optional[str] = None,
                         strict: bool = True) -> Dict[str, Any]:
    """
    Load model checkpoint

    Args:
        filepath: Path to checkpoint file
        model: PyTorch model to load state into
        optimizer: Optimizer to load state into
        map_location: Device mapping for loading
        strict: Whether to strictly enforce state dict matching

    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(filepath, map_location=map_location)

    # Load model state
    if strict:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint["model_state_dict"].items()
                          if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        logger.info(f"Loaded {len(pretrained_dict)}/{len(checkpoint['model_state_dict'])} parameters")

    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            logger.warning(f"Could not load optimizer state: {e}")

    logger.info(f"Loaded checkpoint from {filepath} (epoch {checkpoint.get('epoch', 'unknown')})")

    return checkpoint


def quantize_model(model: nn.Module, quantization_type: str = "dynamic") -> nn.Module:
    """
    Quantize model for inference optimization

    Args:
        model: PyTorch model
        quantization_type: Type of quantization ('dynamic', 'static', 'qat')

    Returns:
        Quantized model
    """
    try:
        if quantization_type == "dynamic":
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.Conv1d}, dtype=torch.qint8
            )
        elif quantization_type == "static":
            # This requires calibration data
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            # Note: Need to run calibration data through model here
            quantized_model = torch.quantization.convert(model, inplace=False)
        else:
            raise ValueError(f"Unsupported quantization type: {quantization_type}")

        logger.info(f"Applied {quantization_type} quantization")
        return quantized_model

    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        return model


def profile_model(model: nn.Module, input_tensor: torch.Tensor,
                 num_runs: int = 10) -> Dict[str, float]:
    """
    Profile model inference time and memory usage

    Args:
        model: PyTorch model
        input_tensor: Input tensor for profiling
        num_runs: Number of runs for averaging

    Returns:
        Dictionary with profiling results
    """
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Warm up
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_tensor)

    # Profile inference time
    if device.type == "cuda":
        torch.cuda.synchronize()

    import time
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()
    avg_inference_time = (end_time - start_time) / num_runs

    # Memory profiling
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(input_tensor)
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    else:
        peak_memory = 0.0

    return {
        "avg_inference_time_ms": avg_inference_time * 1000,
        "peak_memory_mb": peak_memory,
        "throughput_samples_per_sec": 1.0 / avg_inference_time,
    }


def convert_to_torchscript(model: nn.Module, example_input: torch.Tensor,
                          method: str = "trace") -> torch.jit.ScriptModule:
    """
    Convert model to TorchScript for deployment

    Args:
        model: PyTorch model
        example_input: Example input for tracing
        method: Conversion method ('trace' or 'script')

    Returns:
        TorchScript model
    """
    model.eval()

    try:
        if method == "trace":
            traced_model = torch.jit.trace(model, example_input)
        elif method == "script":
            traced_model = torch.jit.script(model)
        else:
            raise ValueError(f"Unknown method: {method}")

        logger.info(f"Converted model to TorchScript using {method}")
        return traced_model

    except Exception as e:
        logger.error(f"TorchScript conversion failed: {e}")
        raise


class ModelEMA:
    """Exponential Moving Average for model parameters"""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.module = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in self.module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA parameters"""
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA parameters to model"""
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """Restore original parameters"""
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


if __name__ == "__main__":
    # Example usage and testing
    print("Testing model utilities...")

    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(1, 64, 3)
            self.linear = nn.Linear(64, 32)
            self.output = nn.Linear(32, 10)

        def forward(self, x):
            x = self.conv(x)
            x = x.mean(dim=-1)  # Global average pooling
            x = self.linear(x)
            x = self.output(x)
            return x

    model = TestModel()

    # Test parameter counting
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    model_size = get_model_size(model)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {model_size:.2f} MB")

    # Test architecture analysis
    analysis = analyze_model_architecture(model)
    print(f"Layer types: {analysis['layer_types']}")

    # Test model summary
    summary = model_summary(model, input_size=(1, 100))
    print(summary)

    print("All tests passed!")
