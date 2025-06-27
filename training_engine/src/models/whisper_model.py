"""
Whisper Model Implementation for Twi Speech Recognition

This module implements a Whisper-based model specifically adapted for Twi ASR.
It provides fine-tuning capabilities and Twi-specific optimizations on top of
OpenAI's Whisper architecture.

Key Features:
- Fine-tuning on Twi speech data
- Multi-dialect support (Asante, Akuapem, Fante)
- Efficient inference and training
- Integration with existing training pipeline
- Custom tokenization for Twi language

Author: Twi Speech Recognition Team
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperConfig,
    WhisperTokenizer,
    WhisperFeatureExtractor
)
from transformers.modeling_outputs import Seq2SeqLMOutput
import numpy as np

logger = logging.getLogger(__name__)


class WhisperForTwiASR(nn.Module):
    """
    Whisper model adapted for Twi Automatic Speech Recognition

    This model extends OpenAI's Whisper with Twi-specific adaptations:
    - Custom vocabulary for Twi language
    - Multi-dialect support
    - Optimized training for low-resource scenarios
    - Integration with existing training pipeline
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-base",
        vocab_size: Optional[int] = None,
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
        dialect_adaptation: bool = True,
        custom_vocabulary: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize WhisperForTwiASR model

        Args:
            model_name: Pre-trained Whisper model identifier
            vocab_size: Custom vocabulary size (if None, uses default)
            freeze_encoder: Whether to freeze encoder weights
            freeze_decoder: Whether to freeze decoder weights
            dialect_adaptation: Enable dialect-specific adaptations
            custom_vocabulary: Custom vocabulary for Twi language
            **kwargs: Additional model configuration
        """
        super().__init__()

        self.model_name = model_name
        self.vocab_size = vocab_size
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
        self.dialect_adaptation = dialect_adaptation

        # Load base Whisper model
        self.config = WhisperConfig.from_pretrained(model_name)
        if vocab_size:
            self.config.vocab_size = vocab_size

        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            config=self.config,
            **kwargs
        )

        # Initialize processor components
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer
        self.feature_extractor = self.processor.feature_extractor

        # Apply custom vocabulary if provided
        if custom_vocabulary:
            self._extend_vocabulary(custom_vocabulary)

        # Freeze components if specified
        if freeze_encoder:
            self._freeze_encoder()
        if freeze_decoder:
            self._freeze_decoder()

        # Add dialect-specific adaptations
        if dialect_adaptation:
            self._add_dialect_adaptations()

        # Twi-specific configurations
        self.twi_config = {
            "dialects": ["Asante", "Akuapem", "Fante"],
            "max_length": 448,  # Optimized for Twi utterances
            "num_beams": 5,
            "early_stopping": True,
            "temperature": 0.6,
            "repetition_penalty": 1.2,
        }

        logger.info(f"Initialized WhisperForTwiASR with model: {model_name}")
        if vocab_size:
            logger.info(f"Custom vocabulary size: {vocab_size}")
        if freeze_encoder:
            logger.info("Encoder frozen for training")
        if freeze_decoder:
            logger.info("Decoder frozen for training")

    def _extend_vocabulary(self, custom_vocabulary: List[str]):
        """Extend tokenizer vocabulary with Twi-specific tokens"""
        try:
            # Add Twi-specific tokens
            twi_tokens = [
                # Common Twi words and phrases
                "<twi>", "</twi>",
                "<asante>", "</asante>",
                "<akuapem>", "</akuapem>",
                "<fante>", "</fante>",
                # Add custom vocabulary
                *custom_vocabulary
            ]

            # Filter out tokens that already exist
            new_tokens = [token for token in twi_tokens if token not in self.tokenizer.vocab]

            if new_tokens:
                self.tokenizer.add_tokens(new_tokens)
                self.model.resize_token_embeddings(len(self.tokenizer))
                logger.info(f"Added {len(new_tokens)} new tokens to vocabulary")

        except Exception as e:
            logger.warning(f"Failed to extend vocabulary: {e}")

    def _freeze_encoder(self):
        """Freeze encoder parameters"""
        for param in self.model.model.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder parameters frozen")

    def _freeze_decoder(self):
        """Freeze decoder parameters"""
        for param in self.model.model.decoder.parameters():
            param.requires_grad = False
        logger.info("Decoder parameters frozen")

    def _add_dialect_adaptations(self):
        """Add dialect-specific adaptation layers"""
        try:
            # Add dialect classification head
            hidden_size = self.config.d_model
            self.dialect_classifier = nn.Linear(hidden_size, len(self.twi_config["dialects"]))

            # Add dialect-specific attention mechanisms
            self.dialect_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )

            logger.info("Added dialect adaptation layers")

        except Exception as e:
            logger.warning(f"Failed to add dialect adaptations: {e}")

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        """
        Forward pass through the model

        Args:
            input_features: Mel spectrogram features
            attention_mask: Attention mask for input features
            decoder_input_ids: Decoder input token IDs
            decoder_attention_mask: Decoder attention mask
            labels: Target labels for training
            return_dict: Whether to return ModelOutput object

        Returns:
            Model outputs including loss, logits, and hidden states
        """
        # Forward through base Whisper model
        outputs = self.model(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            return_dict=return_dict,
            **kwargs
        )

        # Add dialect-specific processing if enabled
        if self.dialect_adaptation and hasattr(self, 'dialect_classifier'):
            try:
                # Get encoder hidden states for dialect classification
                encoder_outputs = self.model.model.encoder(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    return_dict=True
                )

                # Classify dialect
                pooled_output = encoder_outputs.last_hidden_state.mean(dim=1)
                dialect_logits = self.dialect_classifier(pooled_output)

                # Add dialect classification to outputs
                if return_dict:
                    outputs.dialect_logits = dialect_logits
                else:
                    outputs = outputs + (dialect_logits,)

            except Exception as e:
                logger.warning(f"Dialect adaptation failed: {e}")

        return outputs

    def generate(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        dialect: Optional[str] = None,
        **generation_kwargs
    ) -> torch.Tensor:
        """
        Generate transcriptions from audio features

        Args:
            input_features: Mel spectrogram features
            attention_mask: Attention mask for input features
            dialect: Specific dialect to optimize for
            **generation_kwargs: Additional generation parameters

        Returns:
            Generated token sequences
        """
        # Use Twi-optimized generation parameters
        generation_config = {**self.twi_config}
        generation_config.update(generation_kwargs)

        # Add dialect-specific tokens if specified
        if dialect and dialect in self.twi_config["dialects"]:
            dialect_token = f"<{dialect.lower()}>"
            if dialect_token in self.tokenizer.vocab:
                forced_decoder_ids = [[self.tokenizer.vocab[dialect_token]]]
                generation_config["forced_decoder_ids"] = forced_decoder_ids

        # Generate transcription
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                **generation_config
            )

        return generated_ids

    def transcribe(
        self,
        audio_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        dialect: Optional[str] = None,
        return_timestamps: bool = False,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        High-level transcription interface

        Args:
            audio_features: Preprocessed audio features
            attention_mask: Attention mask
            dialect: Target dialect for transcription
            return_timestamps: Whether to return word timestamps
            **kwargs: Additional generation parameters

        Returns:
            Transcribed text or detailed transcription with timestamps
        """
        # Generate token sequences
        generated_ids = self.generate(
            input_features=audio_features,
            attention_mask=attention_mask,
            dialect=dialect,
            **kwargs
        )

        # Decode to text
        transcription = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        # Return simple text or detailed output
        if return_timestamps:
            # TODO: Implement timestamp extraction
            return {
                "text": transcription,
                "dialect": dialect,
                "confidence": None,  # Could be computed from logits
                "timestamps": None   # Would require additional processing
            }
        else:
            return transcription

    def compute_loss(
        self,
        input_features: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute training loss

        Args:
            input_features: Audio features
            labels: Target transcription tokens
            attention_mask: Attention mask
            **kwargs: Additional parameters

        Returns:
            Computed loss tensor
        """
        outputs = self(
            input_features=input_features,
            labels=labels,
            attention_mask=attention_mask,
            **kwargs
        )

        return outputs.loss if hasattr(outputs, 'loss') else outputs[0]

    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size(self) -> Dict[str, int]:
        """Get model size information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = self.get_trainable_parameters()

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
        }

    def save_pretrained(self, save_directory: str):
        """Save model and processor"""
        self.model.save_pretrained(save_directory)
        self.processor.save_pretrained(save_directory)

        # Save additional configuration
        config_path = f"{save_directory}/twi_config.json"
        import json
        with open(config_path, 'w') as f:
            json.dump(self.twi_config, f, indent=2)

        logger.info(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        **kwargs
    ) -> "WhisperForTwiASR":
        """Load pre-trained model"""
        try:
            # Load Twi configuration if available
            import json
            config_path = f"{model_path}/twi_config.json"
            try:
                with open(config_path, 'r') as f:
                    twi_config = json.load(f)
                kwargs.update(twi_config)
            except FileNotFoundError:
                logger.warning("No Twi configuration found, using defaults")

            # Initialize model
            model = cls(model_name=model_path, **kwargs)
            logger.info(f"Loaded WhisperForTwiASR from {model_path}")

            return model

        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    def prepare_for_training(self):
        """Prepare model for training"""
        self.train()

        # Ensure gradients are enabled for trainable parameters
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.grad = None

        logger.info("Model prepared for training")

    def prepare_for_inference(self):
        """Prepare model for inference"""
        self.eval()

        # Optimize for inference
        torch.set_grad_enabled(False)

        logger.info("Model prepared for inference")

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            "model_name": self.model_name,
            "vocab_size": self.vocab_size,
            "freeze_encoder": self.freeze_encoder,
            "freeze_decoder": self.freeze_decoder,
            "dialect_adaptation": self.dialect_adaptation,
            "twi_config": self.twi_config,
            "model_size": self.get_model_size()
        }


def create_whisper_for_twi(
    model_name: str = "openai/whisper-base",
    **kwargs
) -> WhisperForTwiASR:
    """
    Factory function to create WhisperForTwiASR model

    Args:
        model_name: Base Whisper model to use
        **kwargs: Additional model configuration

    Returns:
        Initialized WhisperForTwiASR model
    """
    return WhisperForTwiASR(model_name=model_name, **kwargs)


# Model registry for different Whisper variants
WHISPER_MODELS = {
    "whisper-tiny": "openai/whisper-tiny",
    "whisper-base": "openai/whisper-base",
    "whisper-small": "openai/whisper-small",
    "whisper-medium": "openai/whisper-medium",
    "whisper-large": "openai/whisper-large",
    "whisper-large-v2": "openai/whisper-large-v2",
    "whisper-large-v3": "openai/whisper-large-v3"
}


def get_recommended_whisper_model(
    resource_level: str = "medium",
    language_focus: str = "multilingual"
) -> str:
    """
    Get recommended Whisper model based on requirements

    Args:
        resource_level: Available computational resources (low, medium, high)
        language_focus: Language focus (multilingual, english)

    Returns:
        Recommended model identifier
    """
    if resource_level == "low":
        return WHISPER_MODELS["whisper-tiny"]
    elif resource_level == "medium":
        return WHISPER_MODELS["whisper-base"]
    elif resource_level == "high":
        return WHISPER_MODELS["whisper-large-v3"]
    else:
        return WHISPER_MODELS["whisper-base"]  # Default fallback
