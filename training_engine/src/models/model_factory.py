"""
Model Factory for Twi Speech Recognition Training Engine

This module provides a factory pattern for creating different model architectures
optimized for Twi speech recognition. It supports multiple state-of-the-art models
including Wav2Vec2, Whisper, and custom architectures.

Key Features:
- Support for multiple model architectures
- Twi-specific model adaptations
- Automatic processor creation
- Model configuration management
- Fine-tuning capabilities
- Multi-dialect support

Author: Twi Speech Recognition Team
"""

import logging
from typing import Dict, Any, Tuple, Optional, Union
from pathlib import Path
import torch
import torch.nn as nn
from omegaconf import DictConfig

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    AutoModel,
    AutoProcessor,
    AutoConfig
)

from .wav2vec2_model import Wav2Vec2ForTwiASR
from .whisper_model import WhisperForTwiASR
from .custom_models import TwiASRModel
from .fallback_model import create_fallback_model_and_processor

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory class for creating speech recognition models"""

    def __init__(self, config: DictConfig):
        """
        Initialize the model factory.

        Args:
            config: Model configuration
        """
        self.config = config
        self.supported_models = {
            'wav2vec2': self._create_wav2vec2_model,
            'whisper': self._create_whisper_model,
            'custom': self._create_custom_model,
            'fallback': self._create_fallback_model,
        }

    def create_model(self) -> Tuple[nn.Module, Any]:
        """
        Create model and processor based on configuration.

        Returns:
            Tuple of (model, processor)
        """
        try:
            model_type = self._get_model_type()
            logger.info(f"Detected model type: {model_type}")

            if model_type not in self.supported_models:
                raise ValueError(f"Unsupported model type: {model_type}")

            logger.info(f"Creating model of type: {model_type}")
            logger.debug(f"Model configuration: {self.config}")

            model, processor = self.supported_models[model_type]()

            if model is None:
                raise ValueError(f"Model creation returned None for type: {model_type}")

            if processor is None:
                raise ValueError(f"Processor creation returned None for type: {model_type}")

            logger.info(f"Model created successfully: {type(model).__name__}")
            logger.info(f"Processor created successfully: {type(processor).__name__}")

            # Apply common post-processing
            model = self._apply_model_modifications(model)

            return model, processor

        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            logger.error(f"Configuration: {self.config}")
            raise RuntimeError(f"Model creation failed: {e}") from e

    def _get_model_type(self) -> str:
        """Extract model type from configuration"""
        target_class = self.config.get('_target_', '')

        if 'fallback' in target_class.lower():
            return 'fallback'
        elif 'wav2vec2' in target_class.lower():
            return 'wav2vec2'
        elif 'whisper' in target_class.lower():
            return 'whisper'
        elif 'custom' in target_class.lower():
            return 'custom'
        else:
            # Default to wav2vec2
            logger.warning(f"Unknown model type in {target_class}, defaulting to wav2vec2")
            return 'wav2vec2'

    def _create_wav2vec2_model(self) -> Tuple[Wav2Vec2ForTwiASR, Wav2Vec2Processor]:
        """Create Wav2Vec2 model for Twi ASR"""

        # Get pretrained model name
        pretrained_model = self.config.get('pretrained_model_name', 'facebook/wav2vec2-base')
        logger.info(f"Creating Wav2Vec2 model with base: {pretrained_model}")

        try:
            # Create vocabulary
            vocab_dict = self._create_twi_vocabulary()
            logger.debug(f"Created vocabulary with {len(vocab_dict)} tokens")

            # Create tokenizer - need to save vocab to file first
            import json
            import tempfile
            import os

            # Save vocabulary to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                json.dump(vocab_dict, temp_file)
                vocab_file_path = temp_file.name

            try:
                tokenizer = Wav2Vec2CTCTokenizer(
                    vocab_file_path,
                    unk_token=self.config.get('vocab', {}).get('unk_token', '[UNK]'),
                    pad_token=self.config.get('vocab', {}).get('pad_token', '[PAD]'),
                    word_delimiter_token='|'
                )
                logger.debug("Created tokenizer successfully")
            finally:
                # Clean up temporary file
                if os.path.exists(vocab_file_path):
                    os.unlink(vocab_file_path)

            # Create feature extractor
            feature_extractor = Wav2Vec2FeatureExtractor(
                feature_size=self.config.get('feature_extractor', {}).get('feature_size', 1),
                sampling_rate=self.config.get('feature_extractor', {}).get('sampling_rate', 16000),
                padding_value=self.config.get('feature_extractor', {}).get('padding_value', 0.0),
                do_normalize=self.config.get('feature_extractor', {}).get('do_normalize', True),
                return_attention_mask=self.config.get('feature_extractor', {}).get('return_attention_mask', True)
            )
            logger.debug("Created feature extractor successfully")

            # Create processor
            processor = Wav2Vec2Processor(
                feature_extractor=feature_extractor,
                tokenizer=tokenizer
            )
            logger.debug("Created processor successfully")

            # Try to create model with fallback
            model = None

            # First attempt: Load from pretrained with modifications
            try:
                model_config = AutoConfig.from_pretrained(pretrained_model)
                logger.debug("Loaded pretrained configuration")

                # Modify config for Twi-specific adaptations
                model_config.vocab_size = len(vocab_dict)
                model_config.pad_token_id = tokenizer.pad_token_id
                model_config.ctc_loss_reduction = self.config.get('ctc', {}).get('loss_reduction', 'mean')
                model_config.ctc_zero_infinity = self.config.get('ctc', {}).get('zero_infinity', True)

                # Architecture modifications
                arch_config = self.config.get('architecture', {})
                if arch_config:
                    model_config.hidden_dropout = arch_config.get('hidden_dropout', model_config.hidden_dropout)
                    model_config.attention_dropout = arch_config.get('attention_dropout', model_config.attention_dropout)
                    model_config.feat_proj_dropout = arch_config.get('feat_proj_dropout', model_config.feat_proj_dropout)
                    model_config.layerdrop = arch_config.get('layerdrop', model_config.layerdrop)

                logger.debug("Modified configuration for Twi adaptations")

                # Create model - try custom first, then fallback to standard
                try:
                    model = Wav2Vec2ForTwiASR(config=model_config)
                    logger.info("Created Wav2Vec2ForTwiASR model successfully")

                    # Load pretrained weights if available
                    if self.config.get('fine_tuning', {}).get('strategy', 'full') != 'from_scratch':
                        try:
                            pretrained_model_obj = Wav2Vec2ForCTC.from_pretrained(pretrained_model)
                            self._transfer_pretrained_weights(model, pretrained_model_obj)
                            logger.info(f"Loaded pretrained weights from {pretrained_model}")
                        except Exception as e:
                            logger.warning(f"Failed to load pretrained weights: {e}")
                            logger.info("Continuing with randomly initialized weights")

                except Exception as custom_error:
                    logger.warning(f"Custom Wav2Vec2ForTwiASR failed: {custom_error}")
                    logger.info("Falling back to standard Wav2Vec2ForCTC model")

                    # Fallback: Use standard Wav2Vec2ForCTC
                    try:
                        model = Wav2Vec2ForCTC.from_pretrained(
                            pretrained_model,
                            vocab_size=len(vocab_dict),
                            pad_token_id=tokenizer.pad_token_id
                        )
                        logger.info("Created fallback Wav2Vec2ForCTC model successfully")
                    except Exception as std_error:
                        logger.error(f"Standard Wav2Vec2ForCTC also failed: {std_error}")
                        logger.info("Attempting final fallback to minimal model")

                        # Final fallback: Use minimal fallback model
                        try:
                            model, processor = create_fallback_model_and_processor(self.config)
                            logger.warning("Using minimal fallback model - functionality will be limited")
                            return model, processor
                        except Exception as final_error:
                            raise RuntimeError(f"All model creation attempts failed. Custom: {custom_error}, Standard: {std_error}, Fallback: {final_error}")

            except Exception as e:
                logger.error(f"Failed to create Wav2Vec2 model: {e}")
                logger.info("Attempting fallback to standard Wav2Vec2ForCTC")

                # Fallback: Use standard Wav2Vec2ForCTC
                try:
                    model = Wav2Vec2ForCTC.from_pretrained(
                        pretrained_model,
                        vocab_size=len(vocab_dict),
                        pad_token_id=tokenizer.pad_token_id
                    )
                    logger.info("Created fallback Wav2Vec2ForCTC model")
                except Exception as fallback_error:
                    logger.error(f"Standard Wav2Vec2ForCTC also failed: {fallback_error}")
                    logger.info("Attempting final fallback to minimal model")

                    # Final fallback: Use minimal fallback model
                    try:
                        model, processor = create_fallback_model_and_processor(self.config)
                        logger.warning("Using minimal fallback model - functionality will be limited")
                        return model, processor
                    except Exception as final_error:
                        raise RuntimeError(f"All model creation attempts failed. Original: {e}, Standard: {fallback_error}, Fallback: {final_error}")

            if model is None:
                logger.error("Model creation returned None, using fallback")
                model, processor = create_fallback_model_and_processor(self.config)
                logger.warning("Using minimal fallback model due to None return")
                return model, processor

        except Exception as e:
            logger.error(f"Failed to create Wav2Vec2 model: {e}")
            logger.error(f"Configuration used: {self.config}")
            logger.error(f"Pretrained model: {pretrained_model}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Wav2Vec2 model creation failed: {e}") from e

        logger.info(f"Successfully created Wav2Vec2 model and processor")
        return model, processor

    def _create_fallback_model(self) -> Tuple[Any, Any]:
        """Create fallback model for compatibility testing"""
        logger.info("Creating fallback model for testing")

        from .fallback_model import create_fallback_model_and_processor

        try:
            model, processor = create_fallback_model_and_processor(self.config)
            logger.info("Successfully created fallback model and processor")
            return model, processor
        except Exception as e:
            logger.error(f"Failed to create fallback model: {e}")
            raise RuntimeError(f"Fallback model creation failed: {e}") from e

    def _create_whisper_model(self) -> Tuple[WhisperForTwiASR, WhisperProcessor]:
        """Create Whisper model for Twi ASR"""

        pretrained_model = self.config.get('pretrained_model_name', 'openai/whisper-base')

        try:
            # Load processor
            processor = WhisperProcessor.from_pretrained(pretrained_model)

            # Modify tokenizer for Twi
            twi_tokens = self._get_twi_special_tokens()
            processor.tokenizer.add_tokens(twi_tokens)

            # Load model
            model = WhisperForTwiASR.from_pretrained(pretrained_model)

            # Resize token embeddings
            model.resize_token_embeddings(len(processor.tokenizer))

            # Configure for Twi
            model.config.forced_decoder_ids = None
            model.config.suppress_tokens = []

            logger.info(f"Created Whisper model from {pretrained_model}")

        except Exception as e:
            logger.error(f"Failed to create Whisper model: {e}")
            logger.error(f"Configuration used: {self.config}")
            logger.error(f"Pretrained model: {pretrained_model}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")

            # Fallback to minimal model for Whisper as well
            logger.info("Attempting fallback model for Whisper")
            try:
                model, processor = create_fallback_model_and_processor(self.config)
                logger.warning("Using minimal fallback model instead of Whisper")
                return model, processor
            except Exception as fallback_error:
                raise RuntimeError(f"Whisper model creation failed: {e}, Fallback also failed: {fallback_error}") from e

        logger.info(f"Successfully created Whisper model and processor")
        return model, processor

    def _create_custom_model(self) -> Tuple[TwiASRModel, Any]:
        """Create custom model for Twi ASR"""

        # Create vocabulary and processor
        vocab_dict = self._create_twi_vocabulary()
        processor = self._create_custom_processor(vocab_dict)

        # Create model
        model_config = {
            'vocab_size': len(vocab_dict),
            'hidden_size': self.config.get('architecture', {}).get('hidden_size', 768),
            'num_layers': self.config.get('architecture', {}).get('num_hidden_layers', 12),
            'num_heads': self.config.get('architecture', {}).get('num_attention_heads', 12),
            'dropout': self.config.get('architecture', {}).get('hidden_dropout', 0.1),
        }

        model = TwiASRModel(**model_config)

        logger.info("Created custom Twi ASR model")

        return model, processor

    def _create_twi_vocabulary(self) -> Dict[str, int]:
        """Create vocabulary for Twi language"""

        # Base vocabulary with common characters
        base_chars = [
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            ' ', "'", '-'
        ]

        # Twi-specific characters
        twi_chars = [
            'ɔ',  # Open-mid back rounded vowel
            'ɛ',  # Open-mid front unrounded vowel
            'ŋ',  # Velar nasal
        ]

        # Tone marks (if preserving tones)
        tone_marks = ['́', '̀', '̂', '̃', '̄'] if self.config.get('twi_adaptations', {}).get('tone_aware', False) else []

        # Special tokens
        special_tokens = [
            '[PAD]', '[UNK]', '[BOS]', '[EOS]', '[BLANK]'
        ]

        # Additional special tokens from config
        config_special_tokens = self.config.get('vocab', {}).get('special_tokens', [])

        # Combine all tokens
        all_tokens = special_tokens + base_chars + twi_chars + tone_marks + config_special_tokens

        # Create vocabulary dictionary
        vocab_dict = {token: i for i, token in enumerate(all_tokens)}

        logger.info(f"Created Twi vocabulary with {len(vocab_dict)} tokens")

        return vocab_dict

    def _get_twi_special_tokens(self) -> list:
        """Get Twi-specific special tokens for Whisper"""
        return [
            '<|tw|>',  # Twi language token
            '<|asante|>',  # Asante dialect
            '<|akuapem|>',  # Akuapem dialect
            '<|fante|>',  # Fante dialect
            '<|code_switch|>',  # Code-switching marker
        ]

    def _create_custom_processor(self, vocab_dict: Dict[str, int]) -> Any:
        """Create processor for custom model"""

        class CustomProcessor:
            def __init__(self, vocab_dict):
                self.vocab_dict = vocab_dict
                self.id_to_token = {v: k for k, v in vocab_dict.items()}
                self.pad_token_id = vocab_dict.get('[PAD]', 0)
                self.unk_token_id = vocab_dict.get('[UNK]', 1)

            def __call__(self, audio=None, text=None, **kwargs):
                if audio is not None:
                    return self._process_audio(audio, **kwargs)
                elif text is not None:
                    return self._process_text(text, **kwargs)
                else:
                    raise ValueError("Either audio or text must be provided")

            def _process_audio(self, audio, sampling_rate=16000, **kwargs):
                # Ensure audio is tensor
                if not isinstance(audio, torch.Tensor):
                    audio = torch.tensor(audio, dtype=torch.float32)

                # Normalize
                if audio.abs().max() > 1.0:
                    audio = audio / audio.abs().max()

                return {'input_values': audio.unsqueeze(0) if audio.dim() == 1 else audio}

            def _process_text(self, text, **kwargs):
                # Simple character-level tokenization
                tokens = []
                for char in text.lower():
                    tokens.append(self.vocab_dict.get(char, self.unk_token_id))

                return {'input_ids': torch.tensor(tokens).unsqueeze(0)}

            def batch_decode(self, sequences, **kwargs):
                results = []
                for seq in sequences:
                    tokens = [self.id_to_token.get(id.item(), '[UNK]') for id in seq if id != self.pad_token_id]
                    results.append(''.join(tokens).strip())
                return results

            def decode(self, sequence, **kwargs):
                tokens = [self.id_to_token.get(id.item(), '[UNK]') for id in sequence if id != self.pad_token_id]
                return ''.join(tokens).strip()

        return CustomProcessor(vocab_dict)

    def _transfer_pretrained_weights(self, target_model: nn.Module, source_model: nn.Module):
        """Transfer compatible weights from pretrained model"""

        target_dict = target_model.state_dict()
        source_dict = source_model.state_dict()

        transferred_keys = []
        skipped_keys = []

        for key, param in source_dict.items():
            if key in target_dict:
                if target_dict[key].shape == param.shape:
                    target_dict[key].copy_(param)
                    transferred_keys.append(key)
                else:
                    skipped_keys.append(f"{key} (shape mismatch)")
            else:
                skipped_keys.append(f"{key} (not found)")

        logger.info(f"Transferred {len(transferred_keys)} parameters")
        if skipped_keys:
            logger.warning(f"Skipped {len(skipped_keys)} parameters: {skipped_keys[:5]}...")

    def _apply_model_modifications(self, model: nn.Module) -> nn.Module:
        """Apply model modifications based on configuration"""

        # Gradient checkpointing
        if self.config.get('training', {}).get('gradient_checkpointing', False):
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing")

        # Freeze layers if configured
        fine_tuning_config = self.config.get('fine_tuning', {})
        if fine_tuning_config.get('strategy') == 'feature_extractor_frozen':
            self._freeze_feature_extractor(model)
        elif fine_tuning_config.get('strategy') == 'partial':
            self._freeze_partial_layers(model, fine_tuning_config.get('frozen_layers', {}))

        # Apply dialect adaptation
        if self.config.get('twi_adaptations', {}).get('dialect_adaptation', {}).get('enabled', False):
            model = self._add_dialect_adaptation(model)

        return model

    def _freeze_feature_extractor(self, model: nn.Module):
        """Freeze feature extractor layers"""
        for name, param in model.named_parameters():
            if 'feature_extractor' in name or 'feature_projection' in name:
                param.requires_grad = False
        logger.info("Froze feature extractor layers")

    def _freeze_partial_layers(self, model: nn.Module, frozen_config: Dict[str, Any]):
        """Freeze specific layers based on configuration"""

        # Freeze feature extractor if specified
        if frozen_config.get('feature_extractor', False):
            self._freeze_feature_extractor(model)

        # Freeze specific encoder layers
        frozen_layers = frozen_config.get('encoder_layers', [])
        for name, param in model.named_parameters():
            for layer_idx in frozen_layers:
                if f'encoder.layers.{layer_idx}' in name:
                    param.requires_grad = False

        if frozen_layers:
            logger.info(f"Froze encoder layers: {frozen_layers}")

    def _add_dialect_adaptation(self, model: nn.Module) -> nn.Module:
        """Add dialect adaptation module to model"""

        dialect_config = self.config.get('twi_adaptations', {}).get('dialect_adaptation', {})
        embedding_dim = dialect_config.get('dialect_embedding_dim', 128)

        # Add dialect embedding layer
        if hasattr(model, 'config'):
            num_dialects = len(['Asante', 'Akuapem', 'Fante'])  # Twi dialects
            dialect_embedding = nn.Embedding(num_dialects, embedding_dim)

            # Register as buffer so it moves with model
            model.register_buffer('dialect_embedding', dialect_embedding.weight)
            logger.info(f"Added dialect adaptation with {embedding_dim}D embeddings")

        return model

    def get_model_info(self, model: nn.Module) -> Dict[str, Any]:
        """Get information about the created model"""

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        info = {
            'model_class': model.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        }

        # Add architecture-specific info
        if hasattr(model, 'config'):
            config = model.config
            info.update({
                'hidden_size': getattr(config, 'hidden_size', None),
                'num_layers': getattr(config, 'num_hidden_layers', None),
                'num_attention_heads': getattr(config, 'num_attention_heads', None),
                'vocab_size': getattr(config, 'vocab_size', None),
            })

        return info

    @staticmethod
    def list_available_models() -> Dict[str, list]:
        """List available pretrained models for each architecture"""
        return {
            'wav2vec2': [
                'facebook/wav2vec2-base',
                'facebook/wav2vec2-large',
                'facebook/wav2vec2-large-960h',
                'facebook/wav2vec2-large-960h-lv60-self',
                'microsoft/wav2vec2-base-960h',
            ],
            'whisper': [
                'openai/whisper-tiny',
                'openai/whisper-base',
                'openai/whisper-small',
                'openai/whisper-medium',
                'openai/whisper-large',
                'openai/whisper-large-v2',
            ],
            'custom': [
                'twi_asr_base',
                'twi_asr_large',
            ]
        }

    def save_model_config(self, save_path: Union[str, Path]):
        """Save model configuration to file"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            import yaml
            yaml.dump(self.config, f, default_flow_style=False)

        logger.info(f"Model configuration saved to {save_path}")

    @classmethod
    def from_pretrained(cls, model_path: Union[str, Path]) -> 'ModelFactory':
        """Load model factory from pretrained model directory"""
        model_path = Path(model_path)
        config_path = model_path / 'model_config.yaml'

        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found at {config_path}")

        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return cls(DictConfig(config))


# Helper function for standalone usage
def create_model_from_config(config_path: str) -> Tuple[nn.Module, Any]:
    """Create model from configuration file"""
    import yaml
    from omegaconf import DictConfig

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    factory = ModelFactory(DictConfig(config))
    return factory.create_model()


if __name__ == "__main__":
    # Example usage
    from omegaconf import DictConfig

    # Example configuration
    config = DictConfig({
        '_target_': 'src.models.wav2vec2_model.Wav2Vec2ForTwiASR',
        'pretrained_model_name': 'facebook/wav2vec2-base',
        'vocab': {
            'vocab_size': 64,
            'pad_token': '[PAD]',
            'unk_token': '[UNK]',
        },
        'twi_adaptations': {
            'tone_aware': True,
            'dialect_adaptation': {
                'enabled': True,
                'dialect_embedding_dim': 128
            }
        }
    })

    factory = ModelFactory(config)
    model, processor = factory.create_model()

    info = factory.get_model_info(model)
    print("Model Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
