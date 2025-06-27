"""
Configuration loader for the Twi Speech Training Engine
Handles loading and parsing YAML and JSON configuration files with environment variable support
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

# Import environment loader
try:
    from src.utils.env_loader import load_env, get_env
except ImportError:
    try:
        # Try relative import as fallback
        from ..utils.env_loader import load_env, get_env
    except ImportError:
        # Final fallback if env_loader is not available
        def load_env(*args, **kwargs):
            pass
        def get_env(key, default=None):
            return os.getenv(key, default)

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Handles configuration loading and management"""

    def __init__(self, config_dir: Optional[Path] = None, load_env_vars: bool = True):
        """
        Initialize the configuration loader

        Args:
            config_dir: Base directory for configuration files
            load_env_vars: Whether to automatically load environment variables
        """
        if config_dir is None:
            # Default to configs directory in training_engine
            self.config_dir = Path(__file__).parent.parent.parent / "configs"
        else:
            self.config_dir = Path(config_dir)

        # Load environment variables if requested
        if load_env_vars:
            load_env()

    def load(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from a YAML or JSON file

        Args:
            config_path: Path to the configuration file

        Returns:
            Dictionary containing the configuration
        """
        config_path = Path(config_path)

        # If path is not absolute, check relative to config_dir
        if not config_path.is_absolute() and not config_path.exists():
            config_path = self.config_dir / config_path

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, 'r') as f:
                # Determine file type by extension
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config = json.load(f)
                else:
                    # Try YAML first, then JSON
                    f.seek(0)
                    try:
                        config = yaml.safe_load(f)
                    except yaml.YAMLError:
                        f.seek(0)
                        config = json.load(f)

            # Handle empty files
            if config is None:
                config = {}

            # Replace environment variables in config
            config = self._replace_env_vars(config)

            # Post-process to handle empty strings and integrate env vars
            config = self._post_process_config(config)

            return config

        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ValueError(f"Error parsing configuration file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {e}")

    def load_with_defaults(self, config_path: Union[str, Path],
                          defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load configuration with default values

        Args:
            config_path: Path to the configuration file
            defaults: Dictionary of default values

        Returns:
            Merged configuration dictionary
        """
        if defaults is None:
            defaults = self._get_default_config()

        try:
            config = self.load(config_path)
        except FileNotFoundError:
            # If config file doesn't exist, use defaults
            config = {}

        # Deep merge with defaults
        merged = self._deep_merge(defaults, config)

        # Replace environment variables after merge
        merged = self._replace_env_vars(merged)

        # Post-process to handle empty strings and integrate env vars
        return self._post_process_config(merged)

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries

        Args:
            base: Base dictionary (defaults)
            override: Override dictionary

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            elif value is not None:  # Only override if value is not None
                result[key] = value

        return result

    def _replace_env_vars(self, config: Any) -> Any:
        """
        Recursively replace environment variable placeholders in config

        Args:
            config: Configuration object (dict, list, or scalar)

        Returns:
            Configuration with environment variables replaced
        """
        if isinstance(config, dict):
            return {k: self._replace_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._replace_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Replace ${VAR_NAME} or $VAR_NAME with environment variable
            if config.startswith('${') and config.endswith('}'):
                var_name = config[2:-1]
                env_value = get_env(var_name)
                # Return the env value if set, otherwise return the original string
                return env_value if env_value is not None else config
            elif config.startswith('$'):
                var_name = config[1:]
                env_value = get_env(var_name)
                # Return the env value if set, otherwise return the original string
                return env_value if env_value is not None else config
            return config
        else:
            return config

    def _post_process_config(self, config: Any) -> Any:
        """
        Post-process configuration to handle empty strings and environment variables

        Args:
            config: Configuration object

        Returns:
            Post-processed configuration
        """
        if isinstance(config, dict):
            result = {}
            for k, v in config.items():
                processed = self._post_process_config(v)
                # Special handling for certain keys
                if k in ['uri', 'connection_uri', 'endpoint_url', 'bucket_name',
                        'access_key_id', 'secret_access_key']:
                    # If it's an empty string or still has ${} pattern, try env var
                    if isinstance(processed, str):
                        if not processed or processed.startswith('${'):
                            # Try to get from environment
                            env_key = k.upper()
                            if k == 'uri':
                                env_key = 'MONGODB_URI'
                            elif k == 'endpoint_url':
                                env_key = 'R2_ENDPOINT_URL'
                            elif k == 'bucket_name':
                                env_key = 'R2_BUCKET_NAME'
                            elif k == 'access_key_id':
                                env_key = 'R2_ACCESS_KEY_ID'
                            elif k == 'secret_access_key':
                                env_key = 'R2_SECRET_ACCESS_KEY'

                            env_value = get_env(env_key)
                            if env_value:
                                processed = env_value
                            elif not processed:
                                # Keep empty string for optional configs
                                processed = ''
                result[k] = processed
            return result
        elif isinstance(config, list):
            return [self._post_process_config(item) for item in config]
        else:
            return config

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for the Twi Speech Training Engine

        Returns:
            Dictionary containing default configuration
        """
        return {
            'data_split': {
                'train_ratio': 0.8,
                'val_ratio': 0.1,
                'test_ratio': 0.1
            },
            'mongodb': {
                'uri': get_env('MONGODB_URI', 'mongodb://localhost:27017/'),
                'database': 'twi_speech',
                'collection': 'recordings'
            },
            'r2_storage': {
                'bucket_name': get_env('R2_BUCKET_NAME', 'twi-speech-data'),
                'access_key_id': get_env('R2_ACCESS_KEY_ID', ''),
                'secret_access_key': get_env('R2_SECRET_ACCESS_KEY', ''),
                'endpoint_url': get_env('R2_ENDPOINT_URL', '')
            },
            'audio': {
                'sample_rate': 16000,
                'channels': 1,
                'format': 'wav',
                'bit_depth': 16,
                'max_duration': 30,
                'min_duration': 0.5
            },
            'text': {
                'max_length': 500,
                'min_length': 1,
                'language': 'tw'
            },
            'cache_dir': 'data/cache',
            'recordings_dir': 'data/recordings',
            'script_file': 'script_actual.ts',
            'model': {
                'type': 'wav2vec2',
                'pretrained_model': 'facebook/wav2vec2-base',
                'hidden_size': 768,
                'num_attention_heads': 12,
                'num_hidden_layers': 12
            },
            'training': {
                'batch_size': 8,
                'learning_rate': 1e-4,
                'num_epochs': 10,
                'warmup_steps': 500,
                'gradient_accumulation_steps': 2,
                'fp16': True,
                'save_steps': 500,
                'eval_steps': 500,
                'logging_steps': 100,
                'save_total_limit': 3,
                'load_best_model_at_end': True,
                'metric_for_best_model': 'wer',
                'greater_is_better': False
            },
            'optimization': {
                'optimizer': 'adamw',
                'adam_beta1': 0.9,
                'adam_beta2': 0.999,
                'adam_epsilon': 1e-8,
                'weight_decay': 0.01,
                'max_grad_norm': 1.0
            },
            'augmentation': {
                'enable': True,
                'noise_prob': 0.1,
                'pitch_shift_prob': 0.1,
                'time_stretch_prob': 0.1,
                'volume_change_prob': 0.1
            }
        }


def load_config(config_path: Union[str, Path],
                config_dir: Optional[Path] = None,
                use_defaults: bool = True,
                load_env_vars: bool = True) -> Dict[str, Any]:
    """
    Convenience function to load configuration

    Args:
        config_path: Path to the configuration file (supports .yaml, .yml, .json)
        config_dir: Base directory for configuration files
        use_defaults: Whether to merge with default configuration
        load_env_vars: Whether to load environment variables

    Returns:
        Configuration dictionary
    """
    loader = ConfigLoader(config_dir, load_env_vars=load_env_vars)

    if use_defaults:
        return loader.load_with_defaults(config_path)
    else:
        return loader.load(config_path)


# For backward compatibility
def load_yaml_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML configuration file

    Args:
        path: Path to the YAML file

    Returns:
        Parsed configuration dictionary
    """
    return load_config(path, use_defaults=False)


def load_json_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a JSON configuration file

    Args:
        path: Path to the JSON file

    Returns:
        Parsed configuration dictionary
    """
    return load_config(path, use_defaults=False)
