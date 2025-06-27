"""
Reproducibility Utilities for Twi Speech Recognition Training

This module provides utilities to ensure reproducible training runs by
setting random seeds, managing randomness sources, and providing
deterministic behavior across different platforms and hardware configurations.

Key Features:
- Comprehensive seed setting for all random number generators
- Platform-specific reproducibility settings
- Deterministic behavior configuration
- Random state management and restoration
- Reproducibility verification tools

Author: Twi Speech Recognition Team
"""

import logging
import os
import random
import warnings
from typing import Optional, Dict, Any
import hashlib
import json

import numpy as np
import torch

try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


def set_seed(seed = 42, deterministic: bool = False):
    """
    Set random seeds for reproducible training

    Args:
        seed: Random seed value (int) or seed configuration object
        deterministic: Whether to use deterministic algorithms (may be slower)
    """
    # Handle different seed input types
    if isinstance(seed, dict):
        # If seed is a dictionary, extract the 'random' key
        seed_value = seed.get('random', 42)
    elif hasattr(seed, 'random'):
        # If seed is an object with 'random' attribute
        seed_value = seed.random
    elif isinstance(seed, (int, float)):
        # If seed is a number, use it directly
        seed_value = int(seed)
    else:
        # Fallback to default
        logger.warning(f"Unknown seed type {type(seed)}, using default seed 42")
        seed_value = 42

    logger.info(f"Setting random seed to {seed_value}")

    # Python random
    random.seed(seed_value)

    # NumPy random
    np.random.seed(seed_value)

    # PyTorch random
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    # Transformers random (if available)
    if TRANSFORMERS_AVAILABLE:
        transformers.set_seed(seed_value)

    # Additional environment variables for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # Configure deterministic behavior
    if deterministic:
        enable_deterministic_mode()
    else:
        # Set reasonable defaults for reproducibility vs performance
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        # Use deterministic algorithms where possible without major performance impact
        try:
            torch.use_deterministic_algorithms(False)
        except AttributeError:
            # Older PyTorch versions
            pass

    logger.info(f"Random seed {seed_value} set successfully (deterministic={deterministic})")


def enable_deterministic_mode():
    """Enable full deterministic mode for maximum reproducibility"""
    logger.info("Enabling deterministic mode for maximum reproducibility")

    # PyTorch deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Use deterministic algorithms
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        # Older PyTorch versions
        logger.warning("torch.use_deterministic_algorithms not available in this PyTorch version")

    # Additional environment variables
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # Warn about potential performance impact
    warnings.warn(
        "Deterministic mode is enabled. This may significantly impact performance. "
        "Use only when full reproducibility is required.",
        UserWarning
    )


def disable_deterministic_mode():
    """Disable deterministic mode for better performance"""
    logger.info("Disabling deterministic mode for better performance")

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    try:
        torch.use_deterministic_algorithms(False)
    except AttributeError:
        pass


def get_random_state() -> Dict[str, Any]:
    """
    Get current random state from all generators

    Returns:
        Dictionary containing random states
    """
    state = {
        'python_random': random.getstate(),
        'numpy_random': np.random.get_state(),
        'torch_random': torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state['torch_cuda_random'] = torch.cuda.get_rng_state()
        if torch.cuda.device_count() > 1:
            state['torch_cuda_random_all'] = torch.cuda.get_rng_state_all()

    return state


def set_random_state(state: Dict[str, Any]):
    """
    Restore random state from saved state

    Args:
        state: Dictionary containing random states
    """
    logger.info("Restoring random state")

    if 'python_random' in state:
        random.setstate(state['python_random'])

    if 'numpy_random' in state:
        np.random.set_state(state['numpy_random'])

    if 'torch_random' in state:
        torch.set_rng_state(state['torch_random'])

    if torch.cuda.is_available():
        if 'torch_cuda_random' in state:
            torch.cuda.set_rng_state(state['torch_cuda_random'])

        if 'torch_cuda_random_all' in state and torch.cuda.device_count() > 1:
            torch.cuda.set_rng_state_all(state['torch_cuda_random_all'])


def save_random_state(filepath: str):
    """
    Save current random state to file

    Args:
        filepath: Path to save random state
    """
    state = get_random_state()

    # Convert torch tensors to numpy for JSON serialization
    serializable_state = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            serializable_state[key] = value.cpu().numpy().tolist()
        elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], torch.Tensor):
            serializable_state[key] = [v.cpu().numpy().tolist() for v in value]
        else:
            serializable_state[key] = value

    with open(filepath, 'w') as f:
        json.dump(serializable_state, f, indent=2)

    logger.info(f"Random state saved to {filepath}")


def load_random_state(filepath: str):
    """
    Load random state from file

    Args:
        filepath: Path to load random state from
    """
    with open(filepath, 'r') as f:
        serializable_state = json.load(f)

    # Convert back to appropriate types
    state = {}
    for key, value in serializable_state.items():
        if 'torch' in key and isinstance(value, list):
            if key == 'torch_cuda_random_all':
                state[key] = [torch.tensor(v, dtype=torch.uint8) for v in value]
            else:
                state[key] = torch.tensor(value, dtype=torch.uint8)
        else:
            state[key] = value

    set_random_state(state)
    logger.info(f"Random state loaded from {filepath}")


def verify_reproducibility(model: torch.nn.Module, input_data: torch.Tensor,
                          seed: int = 42, num_runs: int = 3) -> bool:
    """
    Verify that model produces reproducible outputs

    Args:
        model: PyTorch model to test
        input_data: Sample input data
        seed: Random seed to use
        num_runs: Number of runs to compare

    Returns:
        True if outputs are identical across runs
    """
    logger.info(f"Verifying reproducibility with {num_runs} runs")

    outputs = []

    for run in range(num_runs):
        # Set seed before each run
        set_seed(seed, deterministic=True)

        # Run model
        model.eval()
        with torch.no_grad():
            output = model(input_data)

        # Store output
        if isinstance(output, torch.Tensor):
            outputs.append(output.cpu().numpy())
        elif isinstance(output, (tuple, list)):
            outputs.append([o.cpu().numpy() if isinstance(o, torch.Tensor) else o for o in output])
        else:
            outputs.append(output)

    # Check if all outputs are identical
    first_output = outputs[0]

    for i, output in enumerate(outputs[1:], 1):
        if isinstance(first_output, np.ndarray):
            if not np.allclose(first_output, output, rtol=1e-6, atol=1e-8):
                logger.error(f"Reproducibility check failed at run {i}")
                return False
        elif isinstance(first_output, list):
            for j, (first_item, output_item) in enumerate(zip(first_output, output)):
                if isinstance(first_item, np.ndarray):
                    if not np.allclose(first_item, output_item, rtol=1e-6, atol=1e-8):
                        logger.error(f"Reproducibility check failed at run {i}, item {j}")
                        return False
        else:
            if first_output != output:
                logger.error(f"Reproducibility check failed at run {i}")
                return False

    logger.info("Reproducibility check passed")
    return True


def create_experiment_hash(config: Dict[str, Any], exclude_keys: Optional[list] = None) -> str:
    """
    Create a hash for experiment configuration for reproducibility tracking

    Args:
        config: Configuration dictionary
        exclude_keys: Keys to exclude from hashing (e.g., paths, names)

    Returns:
        SHA256 hash of the configuration
    """
    if exclude_keys is None:
        exclude_keys = ['experiment_name', 'output_dir', 'log_dir', 'checkpoint_dir']

    # Create a copy of config without excluded keys
    config_copy = {}
    for key, value in config.items():
        if key not in exclude_keys:
            config_copy[key] = value

    # Convert to JSON string for hashing
    config_str = json.dumps(config_copy, sort_keys=True, default=str)

    # Create hash
    experiment_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

    return experiment_hash


def setup_reproducible_environment(seed: int = 42, deterministic: bool = False):
    """
    Setup complete reproducible environment

    Args:
        seed: Random seed
        deterministic: Whether to enable deterministic mode
    """
    logger.info("Setting up reproducible environment")

    # Set random seeds
    set_seed(seed, deterministic)

    # Additional environment setup
    if deterministic:
        # Disable multithreading in some libraries for deterministic behavior
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'

        logger.warning(
            "Deterministic mode may significantly reduce performance. "
            "Use only when full reproducibility is required."
        )

    # Log environment information
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"CUDNN version: {torch.backends.cudnn.version()}")
        logger.info(f"CUDNN deterministic: {torch.backends.cudnn.deterministic}")
        logger.info(f"CUDNN benchmark: {torch.backends.cudnn.benchmark}")


class ReproducibilityContext:
    """Context manager for reproducible code blocks"""

    def __init__(self, seed: int = 42, deterministic: bool = False):
        self.seed = seed
        self.deterministic = deterministic
        self.original_state = None

    def __enter__(self):
        # Save current state
        self.original_state = get_random_state()

        # Set reproducible state
        set_seed(self.seed, self.deterministic)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original state
        if self.original_state is not None:
            set_random_state(self.original_state)


def reproducible_function(seed: int = 42, deterministic: bool = False):
    """
    Decorator to make functions reproducible

    Args:
        seed: Random seed to use
        deterministic: Whether to use deterministic mode
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with ReproducibilityContext(seed, deterministic):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Utility functions for debugging reproducibility issues
def compare_model_states(model1: torch.nn.Module, model2: torch.nn.Module) -> Dict[str, bool]:
    """
    Compare states of two models to check for differences

    Args:
        model1: First model
        model2: Second model

    Returns:
        Dictionary indicating which parameters differ
    """
    state1 = model1.state_dict()
    state2 = model2.state_dict()

    differences = {}

    for key in state1.keys():
        if key in state2:
            are_equal = torch.allclose(state1[key], state2[key], rtol=1e-6, atol=1e-8)
            differences[key] = are_equal
        else:
            differences[key] = False

    return differences


def log_reproducibility_info():
    """Log information relevant to reproducibility"""
    logger.info("Reproducibility Information:")
    logger.info(f"  Python random seed: {random.getstate()[1][0]}")
    logger.info(f"  PyTorch version: {torch.__version__}")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  CUDNN version: {torch.backends.cudnn.version()}")
        logger.info(f"  CUDNN deterministic: {torch.backends.cudnn.deterministic}")
        logger.info(f"  CUDNN benchmark: {torch.backends.cudnn.benchmark}")

    # Check for deterministic algorithms
    try:
        det_algos = torch.are_deterministic_algorithms_enabled()
        logger.info(f"  Deterministic algorithms: {det_algos}")
    except AttributeError:
        logger.info("  Deterministic algorithms: not available")

    # Environment variables
    relevant_env_vars = [
        'PYTHONHASHSEED',
        'CUBLAS_WORKSPACE_CONFIG',
        'OMP_NUM_THREADS',
        'MKL_NUM_THREADS'
    ]

    for var in relevant_env_vars:
        value = os.environ.get(var, 'not set')
        logger.info(f"  {var}: {value}")


if __name__ == "__main__":
    # Example usage and testing
    print("Testing reproducibility utilities...")

    # Test basic seed setting
    set_seed(42)
    print(f"Random number after seed 42: {random.random()}")

    set_seed(42)
    print(f"Random number after seed 42 again: {random.random()}")

    # Test context manager
    with ReproducibilityContext(seed=123):
        print(f"Random number in context: {random.random()}")

    print(f"Random number after context: {random.random()}")

    # Test decorator
    @reproducible_function(seed=456)
    def test_function():
        return random.random()

    print(f"Decorated function result 1: {test_function()}")
    print(f"Decorated function result 2: {test_function()}")

    # Log reproducibility info
    log_reproducibility_info()

    print("Reproducibility utilities test completed!")
