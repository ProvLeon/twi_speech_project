"""
Environment Variable Loader for Twi Speech Training Engine

This module handles loading environment variables from .env files
and provides utilities for managing configuration settings.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class EnvLoader:
    """Handles loading environment variables from .env files"""

    def __init__(self, env_file: Optional[Union[str, Path]] = None):
        """
        Initialize the environment loader

        Args:
            env_file: Path to .env file. If None, searches for .env in common locations
        """
        self.env_file = self._find_env_file(env_file)
        self.loaded = False

    def _find_env_file(self, env_file: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """Find the .env file in common locations"""
        if env_file:
            env_path = Path(env_file)
            if env_path.exists():
                return env_path
            else:
                logger.warning(f"Specified .env file not found: {env_file}")
                return None

        # Search for .env file in common locations
        search_paths = [
            Path.cwd() / '.env',  # Current directory
            Path.cwd().parent / '.env',  # Parent directory
            Path.cwd().parent.parent / '.env',  # Grandparent directory
            Path(__file__).parent.parent.parent / '.env',  # Project root from this file
            Path(__file__).parent.parent.parent.parent / '.env',  # One level up
        ]

        for path in search_paths:
            if path.exists():
                logger.info(f"Found .env file at: {path}")
                return path

        logger.debug("No .env file found in common locations")
        return None

    def load(self, override: bool = False) -> Dict[str, str]:
        """
        Load environment variables from .env file

        Args:
            override: Whether to override existing environment variables

        Returns:
            Dictionary of loaded environment variables
        """
        if self.loaded and not override:
            logger.debug("Environment variables already loaded")
            return {}

        if not self.env_file:
            logger.debug("No .env file to load")
            return {}

        loaded_vars = {}

        try:
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()

                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue

                    # Parse key=value pairs
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()

                        # Remove quotes if present
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]

                        # Clean up the value - remove any comments after the value
                        if '#' in value:
                            # Only split on # if it's not part of a URL (e.g., anchor in URL)
                            if not ('://' in value and value.index('#') > value.index('://')):
                                value = value.split('#')[0].strip()

                        # Remove any escaped characters that might have been added
                        value = value.replace('\\032', ' ').replace('\\n', '\n').replace('\\t', '\t')

                        # Only set if not already in environment or if override is True
                        if override or key not in os.environ:
                            os.environ[key] = value
                            loaded_vars[key] = value

            self.loaded = True
            logger.info(f"Loaded {len(loaded_vars)} environment variables from {self.env_file}")

        except Exception as e:
            logger.error(f"Error loading .env file: {e}")

        return loaded_vars

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get an environment variable with optional default

        Args:
            key: Environment variable name
            default: Default value if not found

        Returns:
            Environment variable value or default
        """
        # Ensure environment is loaded
        if not self.loaded:
            self.load()

        value = os.getenv(key, default)

        # Clean up the value if it exists
        if value:
            # Remove any inline comments that might have leaked through
            if '#' in value and not ('://' in value and value.index('#') > value.index('://')):
                value = value.split('#')[0].strip()

            # Remove escaped characters
            value = value.replace('\\032', ' ').replace('\\n', '\n').replace('\\t', '\t')

            # Return None for empty strings
            if not value:
                return default

        return value

    def get_required(self, key: str) -> str:
        """
        Get a required environment variable

        Args:
            key: Environment variable name

        Returns:
            Environment variable value

        Raises:
            ValueError: If the environment variable is not set
        """
        value = self.get(key)
        if value is None:
            raise ValueError(f"Required environment variable '{key}' is not set")
        return value

    def get_bool(self, key: str, default: bool = False) -> bool:
        """
        Get an environment variable as a boolean

        Args:
            key: Environment variable name
            default: Default value if not found

        Returns:
            Boolean value
        """
        value = self.get(key)
        if value is None:
            return default

        return value.lower() in ('true', '1', 'yes', 'on')

    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """
        Get an environment variable as an integer

        Args:
            key: Environment variable name
            default: Default value if not found

        Returns:
            Integer value or default
        """
        value = self.get(key)
        if value is None:
            return default

        try:
            return int(value)
        except ValueError:
            logger.warning(f"Invalid integer value for {key}: {value}")
            return default

    def get_float(self, key: str, default: Optional[float] = None) -> Optional[float]:
        """
        Get an environment variable as a float

        Args:
            key: Environment variable name
            default: Default value if not found

        Returns:
            Float value or default
        """
        value = self.get(key)
        if value is None:
            return default

        try:
            return float(value)
        except ValueError:
            logger.warning(f"Invalid float value for {key}: {value}")
            return default

    def get_list(self, key: str, separator: str = ',', default: Optional[list] = None) -> list:
        """
        Get an environment variable as a list

        Args:
            key: Environment variable name
            separator: String separator for list items
            default: Default value if not found

        Returns:
            List of values
        """
        value = self.get(key)
        if value is None:
            return default or []

        return [item.strip() for item in value.split(separator) if item.strip()]


# Global instance for convenience
_env_loader = None


def load_env(env_file: Optional[Union[str, Path]] = None, override: bool = False) -> Dict[str, str]:
    """
    Load environment variables from .env file

    Args:
        env_file: Path to .env file
        override: Whether to override existing environment variables

    Returns:
        Dictionary of loaded environment variables
    """
    global _env_loader

    if _env_loader is None or env_file:
        _env_loader = EnvLoader(env_file)

    return _env_loader.load(override)


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get an environment variable with optional default

    Args:
        key: Environment variable name
        default: Default value if not found

    Returns:
        Environment variable value or default
    """
    global _env_loader

    if _env_loader is None:
        _env_loader = EnvLoader()

    return _env_loader.get(key, default)


def get_required_env(key: str) -> str:
    """
    Get a required environment variable

    Args:
        key: Environment variable name

    Returns:
        Environment variable value

    Raises:
        ValueError: If the environment variable is not set
    """
    global _env_loader

    if _env_loader is None:
        _env_loader = EnvLoader()

    return _env_loader.get_required(key)


# Convenience functions
def get_env_bool(key: str, default: bool = False) -> bool:
    """Get environment variable as boolean"""
    global _env_loader

    if _env_loader is None:
        _env_loader = EnvLoader()

    return _env_loader.get_bool(key, default)


def get_env_int(key: str, default: Optional[int] = None) -> Optional[int]:
    """Get environment variable as integer"""
    global _env_loader

    if _env_loader is None:
        _env_loader = EnvLoader()

    return _env_loader.get_int(key, default)


def get_env_float(key: str, default: Optional[float] = None) -> Optional[float]:
    """Get environment variable as float"""
    global _env_loader

    if _env_loader is None:
        _env_loader = EnvLoader()

    return _env_loader.get_float(key, default)


def get_env_list(key: str, separator: str = ',', default: Optional[list] = None) -> list:
    """Get environment variable as list"""
    global _env_loader

    if _env_loader is None:
        _env_loader = EnvLoader()

    return _env_loader.get_list(key, separator, default)


# Auto-load on import
load_env()
