"""
Custom Model Architectures for Twi Speech Recognition

This module implements custom neural network architectures specifically designed
for Twi speech recognition. These models are built from scratch to handle the
unique characteristics of the Twi language including tonal features, dialectal
variations, and code-switching patterns.

Key Features:
- Lightweight architectures for resource-constrained environments
- Tonal feature modeling
- Multi-dialect support
- Code-switching detection
- Efficient inference for mobile deployment

Author: Twi Speech Recognition Team
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

logger = logging.getLogger(__name__)


class TwiASRModel(nn.Module):
    """
    Custom Twi ASR model built from scratch for optimal performance
    """

    def __init__(
        self,
        vocab_size: int = 64,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_length: int = 2000,
        use_tonal_features: bool = True,
        use_dialect_adaptation: bool = True,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_length = max_length
        self.use_tonal_features = use_tonal_features
        self.use_dialect_adaptation = use_dialect_adaptation

        # Audio feature extraction
        self.feature_extractor = TwiAudioFeatureExtractor(
            input_dim=1,
            hidden_dim=hidden_size,
            output_dim=hidden_size
        )

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            hidden_size, dropout, max_length
        )

        # Tonal feature modeling
        if use_tonal_features:
            self.tonal_encoder = TonalFeatureEncoder(hidden_size)

        # Dialect adaptation
        if use_dialect_adaptation:
            self.dialect_adapter = DialectAdapter(hidden_size, num_dialects=3)

        # Transformer encoder layers
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

        # CTC head
        self.ctc_head = nn.Linear(hidden_size, vocab_size)

        # Auxiliary heads
        self.dialect_head = nn.Linear(hidden_size, 3)  # Asante, Akuapem, Fante
        self.tone_head = nn.Linear(hidden_size, 5)     # 5 tonal classes

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        dialect: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model

        Args:
            input_values: Raw audio waveform [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            dialect: Dialect indices [batch_size]
            return_dict: Whether to return a dictionary

        Returns:
            Dictionary containing model outputs
        """
        # Extract audio features
        features = self.feature_extractor(input_values)  # [batch_size, seq_len, hidden_size]

        # Add positional encoding
        features = self.positional_encoding(features)

        # Apply tonal features if enabled
        if self.use_tonal_features:
            tonal_features = self.tonal_encoder(features)
            features = features + tonal_features

        # Apply dialect adaptation if enabled
        if self.use_dialect_adaptation and dialect is not None:
            features = self.dialect_adapter(features, dialect)

        # Apply dropout
        features = self.dropout(features)

        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to boolean mask (True for valid positions)
            bool_mask = attention_mask.bool()
            # Invert for transformer (False for valid positions)
            src_key_padding_mask = ~bool_mask
        else:
            src_key_padding_mask = None

        # Pass through transformer
        hidden_states = self.transformer(
            features,
            src_key_padding_mask=src_key_padding_mask
        )

        # Generate outputs
        logits = self.ctc_head(hidden_states)
        dialect_logits = self.dialect_head(hidden_states.mean(dim=1))  # Global average pooling
        tone_logits = self.tone_head(hidden_states)

        if return_dict:
            return {
                'logits': logits,
                'hidden_states': hidden_states,
                'dialect_logits': dialect_logits,
                'tone_logits': tone_logits,
            }
        else:
            return (logits, hidden_states, dialect_logits, tone_logits)


class TwiAudioFeatureExtractor(nn.Module):
    """
    Audio feature extraction module for Twi speech
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 256, output_dim: int = 256):
        super().__init__()

        # Convolutional layers for local feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim // 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Linear projection to output dimension
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from audio

        Args:
            x: Input audio [batch_size, seq_len]

        Returns:
            Features [batch_size, seq_len//8, output_dim]
        """
        # Add channel dimension if not present
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, seq_len]

        # Apply convolutions
        x = self.conv_layers(x)  # [batch_size, hidden_dim, seq_len//8]

        # Transpose for linear layer
        x = x.transpose(1, 2)  # [batch_size, seq_len//8, hidden_dim]

        # Project to output dimension
        x = self.projection(x)  # [batch_size, seq_len//8, output_dim]

        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer layers
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class TonalFeatureEncoder(nn.Module):
    """
    Encoder for tonal features specific to Twi language
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Tonal feature extraction layers
        self.tonal_conv = nn.Conv1d(
            hidden_size, hidden_size,
            kernel_size=5, padding=2, groups=hidden_size // 8
        )
        self.tonal_norm = nn.LayerNorm(hidden_size)
        self.tonal_activation = nn.GELU()

        # Tone classification head for auxiliary learning
        self.tone_classifier = nn.Linear(hidden_size, 5)  # 5 tones

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract tonal features

        Args:
            x: Input features [batch_size, seq_len, hidden_size]

        Returns:
            Tonal features [batch_size, seq_len, hidden_size]
        """
        # Apply convolution for tonal feature extraction
        x_conv = x.transpose(1, 2)  # [batch_size, hidden_size, seq_len]
        x_conv = self.tonal_conv(x_conv)
        x_conv = x_conv.transpose(1, 2)  # [batch_size, seq_len, hidden_size]

        # Normalize and activate
        x_conv = self.tonal_norm(x_conv)
        x_conv = self.tonal_activation(x_conv)

        return x_conv


class DialectAdapter(nn.Module):
    """
    Adapter module for handling multiple Twi dialects
    """

    def __init__(self, hidden_size: int, num_dialects: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_dialects = num_dialects

        # Dialect-specific transformation matrices
        self.dialect_transforms = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_dialects)
        ])

        # Dialect embeddings
        self.dialect_embeddings = nn.Embedding(num_dialects, hidden_size)

    def forward(self, x: torch.Tensor, dialect: torch.Tensor) -> torch.Tensor:
        """
        Apply dialect-specific adaptations

        Args:
            x: Input features [batch_size, seq_len, hidden_size]
            dialect: Dialect indices [batch_size]

        Returns:
            Adapted features [batch_size, seq_len, hidden_size]
        """
        batch_size = x.size(0)
        adapted_features = []

        for i in range(batch_size):
            dialect_idx = dialect[i].item()
            # Apply dialect-specific transformation
            transformed = self.dialect_transforms[dialect_idx](x[i])
            # Add dialect embedding
            dialect_emb = self.dialect_embeddings(dialect[i]).unsqueeze(0)
            adapted = transformed + dialect_emb
            adapted_features.append(adapted)

        return torch.stack(adapted_features)


class LightweightTwiASR(nn.Module):
    """
    Lightweight version of Twi ASR model for mobile deployment
    """

    def __init__(
        self,
        vocab_size: int = 64,
        hidden_size: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Lightweight feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, hidden_size // 2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # LSTM layers for sequence modeling
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        # CTC head
        self.ctc_head = nn.Linear(hidden_size * 2, vocab_size)  # *2 for bidirectional

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for lightweight model

        Args:
            input_values: Raw audio [batch_size, seq_len]

        Returns:
            CTC logits [batch_size, seq_len//4, vocab_size]
        """
        # Add channel dimension
        if input_values.dim() == 2:
            input_values = input_values.unsqueeze(1)

        # Extract features
        features = self.feature_extractor(input_values)  # [batch_size, hidden_size, seq_len//4]
        features = features.transpose(1, 2)  # [batch_size, seq_len//4, hidden_size]

        # Apply LSTM
        lstm_out, _ = self.lstm(features)  # [batch_size, seq_len//4, hidden_size*2]
        lstm_out = self.dropout(lstm_out)

        # Generate logits
        logits = self.ctc_head(lstm_out)

        return logits


class CodeSwitchingDetector(nn.Module):
    """
    Model for detecting code-switching between Twi and English
    """

    def __init__(self, hidden_size: int = 256, num_languages: int = 2):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_languages = num_languages

        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),

            nn.Conv1d(128, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
        )

        # Temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=2,
            dropout=0.1,
            batch_first=True,
            bidirectional=True
        )

        # Language classification head
        self.language_head = nn.Linear(hidden_size, num_languages)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Detect code-switching in audio

        Args:
            input_values: Raw audio [batch_size, seq_len]

        Returns:
            Language predictions [batch_size, seq_len//32, num_languages]
        """
        # Add channel dimension
        if input_values.dim() == 2:
            input_values = input_values.unsqueeze(1)

        # Extract features
        features = self.feature_extractor(input_values)
        features = features.transpose(1, 2)

        # Apply LSTM
        lstm_out, _ = self.lstm(features)

        # Generate language predictions
        language_logits = self.language_head(lstm_out)

        return language_logits


class TwiEndToEndASR(nn.Module):
    """
    End-to-end Twi ASR model with integrated components
    """

    def __init__(
        self,
        vocab_size: int = 64,
        hidden_size: int = 512,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Audio encoder
        self.audio_encoder = TwiAudioFeatureExtractor(
            input_dim=1,
            hidden_dim=hidden_size,
            output_dim=hidden_size
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers
        )

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_decoder_layers
        )

        # Token embeddings for decoder
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size, dropout)

        # Output projection
        self.output_projection = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        input_values: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        End-to-end forward pass

        Args:
            input_values: Raw audio [batch_size, seq_len]
            target_ids: Target token ids [batch_size, target_len]
            attention_mask: Audio attention mask [batch_size, seq_len]

        Returns:
            Output logits [batch_size, target_len, vocab_size]
        """
        # Encode audio
        audio_features = self.audio_encoder(input_values)

        # Create audio attention mask
        if attention_mask is not None:
            # Downsample attention mask to match audio features
            audio_mask = attention_mask[:, ::8]  # Assuming 8x downsampling
            audio_mask = audio_mask[:, :audio_features.size(1)]
        else:
            audio_mask = None

        # Encode with transformer
        if audio_mask is not None:
            src_key_padding_mask = ~audio_mask.bool()
        else:
            src_key_padding_mask = None

        encoded_audio = self.transformer_encoder(
            audio_features,
            src_key_padding_mask=src_key_padding_mask
        )

        # Decode (for training with teacher forcing)
        if target_ids is not None:
            # Embed target tokens
            target_embeds = self.token_embedding(target_ids)
            target_embeds = self.positional_encoding(target_embeds)

            # Create causal mask for decoder
            target_len = target_ids.size(1)
            tgt_mask = self._generate_square_subsequent_mask(target_len).to(target_ids.device)

            # Decode
            decoded = self.transformer_decoder(
                target_embeds,
                encoded_audio,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask
            )

            # Project to vocabulary
            logits = self.output_projection(decoded)
            return logits
        else:
            # Return encoded audio for inference
            return encoded_audio

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


# Utility functions
def count_parameters(model: nn.Module) -> int:
    """Count total parameters in model"""
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


# Model factory functions
def create_custom_twi_asr(
    model_type: str = "standard",
    vocab_size: int = 64,
    **kwargs
) -> nn.Module:
    """
    Factory function to create custom Twi ASR models

    Args:
        model_type: Type of model ('standard', 'lightweight', 'end2end')
        vocab_size: Vocabulary size
        **kwargs: Additional model parameters

    Returns:
        Instantiated model
    """
    if model_type == "standard":
        return TwiASRModel(vocab_size=vocab_size, **kwargs)
    elif model_type == "lightweight":
        return LightweightTwiASR(vocab_size=vocab_size, **kwargs)
    elif model_type == "end2end":
        return TwiEndToEndASR(vocab_size=vocab_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model creation and forward pass
    print("Testing custom Twi ASR models...")

    # Test standard model
    model = TwiASRModel(vocab_size=64, hidden_size=256)
    print(f"Standard model parameters: {count_parameters(model):,}")
    print(f"Standard model size: {get_model_size_mb(model):.2f} MB")

    # Test lightweight model
    lightweight_model = LightweightTwiASR(vocab_size=64, hidden_size=128)
    print(f"Lightweight model parameters: {count_parameters(lightweight_model):,}")
    print(f"Lightweight model size: {get_model_size_mb(lightweight_model):.2f} MB")

    # Test forward pass
    batch_size = 2
    seq_len = 16000  # 1 second at 16kHz

    dummy_audio = torch.randn(batch_size, seq_len)
    dummy_dialect = torch.randint(0, 3, (batch_size,))

    with torch.no_grad():
        outputs = model(dummy_audio, dialect=dummy_dialect)
        print(f"Output logits shape: {outputs['logits'].shape}")
        print(f"Dialect logits shape: {outputs['dialect_logits'].shape}")

        lightweight_outputs = lightweight_model(dummy_audio)
        print(f"Lightweight output shape: {lightweight_outputs.shape}")

    print("All tests passed!")
