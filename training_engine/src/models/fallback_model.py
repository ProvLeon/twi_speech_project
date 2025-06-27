"""
Fallback Model for Twi Speech Recognition Training Engine

This module provides a minimal fallback model implementation that can be used
when the full dependencies are not available or when the main models fail to load.
This is primarily for testing and development purposes.

Key Features:
- Minimal dependencies (only PyTorch)
- Simple CTC-based architecture
- Compatible with training pipeline
- Fallback processor implementation
- Basic Twi vocabulary support

Author: Twi Speech Recognition Team
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import string
import numpy as np

logger = logging.getLogger(__name__)


class FallbackProcessor:
    """Minimal processor for fallback model"""

    def __init__(self, vocab_size: int = 32):
        self.vocab_size = vocab_size

        # Create basic vocabulary
        self.vocab = self._create_basic_vocab()
        self.vocab_to_id = {v: i for i, v in enumerate(self.vocab)}
        self.id_to_vocab = {i: v for i, v in enumerate(self.vocab)}

        # Special tokens
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.blank_token = "<blank>"

        self.pad_token_id = self.vocab_to_id.get(self.pad_token, 0)
        self.unk_token_id = self.vocab_to_id.get(self.unk_token, 1)
        self.blank_token_id = self.vocab_to_id.get(self.blank_token, 2)

        logger.info(f"Fallback processor initialized with vocab size: {len(self.vocab)}")

    @property
    def tokenizer(self):
        """Return self as tokenizer for compatibility"""
        return self

    def _create_basic_vocab(self) -> List[str]:
        """Create basic vocabulary for testing"""
        vocab = ["<pad>", "<unk>", "<blank>"]

        # Add common English letters (for basic testing)
        vocab.extend(list(string.ascii_lowercase))

        # Add space
        vocab.append(" ")

        # Pad to required size
        while len(vocab) < self.vocab_size:
            vocab.append(f"<token_{len(vocab)}>")

        return vocab[:self.vocab_size]

    def __call__(self, audio=None, text=None, return_tensors="pt", padding=True, truncation=True, max_length=None, **kwargs):
        """Process audio and text - unified interface"""

        # If first argument is a string, treat it as text
        if isinstance(audio, str):
            text = audio
            audio = None

        # Handle text-only calls (when used as tokenizer)
        if audio is None and text is not None:
            tokens = self._tokenize_text(text)
            result = {"input_ids": torch.tensor([tokens])}
            return result

        # Handle audio processing
        if audio is not None:
            # Convert numpy array to torch tensor if needed
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).float()

            # Simple processing - just return the input
            batch_size = audio.shape[0] if len(audio.shape) > 1 else 1
            sequence_length = audio.shape[-1] if len(audio.shape) > 1 else audio.shape[0]

            # Create dummy features (normally would be mel spectrograms)
            input_features = audio.unsqueeze(0) if len(audio.shape) == 1 else audio

            # Create a compatible object with attributes like transformers processors
            class ProcessorOutput:
                def __init__(self, input_values, attention_mask):
                    self.input_values = input_values
                    self.attention_mask = attention_mask

            result = ProcessorOutput(
                input_values=input_features,
                attention_mask=torch.ones(batch_size, sequence_length // 160)  # Rough downsampling
            )

            return result

        return {}

    def _tokenize_text(self, text: str) -> List[int]:
        """Simple text tokenization"""
        tokens = []
        for char in text.lower():
            if char in self.vocab_to_id:
                tokens.append(self.vocab_to_id[char])
            else:
                tokens.append(self.unk_token_id)
        return tokens

    def batch_decode(self, sequences: torch.Tensor, skip_special_tokens: bool = True, **kwargs) -> List[str]:
        """Decode token sequences to text"""
        if len(sequences.shape) == 1:
            sequences = sequences.unsqueeze(0)

        results = []
        for sequence in sequences:
            text = ""
            for token_id in sequence:
                token_id = token_id.item() if isinstance(token_id, torch.Tensor) else token_id
                if token_id in self.id_to_vocab:
                    token = self.id_to_vocab[token_id]
                    if skip_special_tokens and token.startswith("<") and token.endswith(">"):
                        continue
                    text += token
                else:
                    if not skip_special_tokens:
                        text += self.unk_token
            results.append(text)

        return results


class FallbackTwiASR(nn.Module):
    """
    Minimal fallback ASR model for testing when dependencies are missing

    This is a simple CNN + RNN + CTC model that can be used for basic testing
    and development when the full Wav2Vec2 or Whisper models are not available.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()

        # Default configuration
        self.config = config or {}

        # Model parameters
        self.vocab_size = self.config.get('vocab_size', 32)
        self.hidden_size = self.config.get('hidden_size', 256)
        self.num_layers = self.config.get('num_layers', 2)
        self.dropout = self.config.get('dropout', 0.1)

        # Feature extraction (simple CNN)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Conv1d(128, self.hidden_size, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
        )

        # Sequence modeling (BiLSTM)
        self.encoder = nn.LSTM(
            self.hidden_size,
            self.hidden_size // 2,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0
        )

        # CTC head
        self.dropout_layer = nn.Dropout(self.dropout)
        self.classifier = nn.Linear(self.hidden_size, self.vocab_size)

        # Initialize weights
        self._init_weights()

        logger.info(f"Fallback ASR model initialized with vocab_size={self.vocab_size}, hidden_size={self.hidden_size}")

    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Forward pass through the model"""

        # Ensure input is in the right format [batch, time] or [batch, channels, time]
        if len(input_features.shape) == 2:
            # [batch, time] -> [batch, 1, time]
            input_features = input_features.unsqueeze(1)
        elif len(input_features.shape) == 3 and input_features.shape[1] > input_features.shape[2]:
            # [batch, time, features] -> [batch, features, time]
            input_features = input_features.transpose(1, 2)

        batch_size = input_features.shape[0]

        # Feature extraction
        features = self.feature_extractor(input_features)  # [batch, hidden_size, time']
        features = features.transpose(1, 2)  # [batch, time', hidden_size]

        # Sequence modeling
        lstm_out, _ = self.encoder(features)  # [batch, time', hidden_size]

        # Apply dropout
        lstm_out = self.dropout_layer(lstm_out)

        # Classification
        logits = self.classifier(lstm_out)  # [batch, time', vocab_size]

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, logits.shape[1], device=logits.device)

        # Prepare output
        loss = None
        if labels is not None:
            # Compute CTC loss
            loss = self._compute_ctc_loss(logits, labels, attention_mask)

        # Return in a format similar to transformers models
        return FallbackModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=lstm_out,
            attentions=None
        )

    def _compute_ctc_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute CTC loss"""
        # Log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Input lengths (actual sequence lengths)
        input_lengths = attention_mask.sum(dim=1).long()

        # Target lengths
        if len(labels.shape) == 1:
            target_lengths = torch.tensor([labels.shape[0]], device=labels.device)
            labels = labels.unsqueeze(0)
        else:
            # Assume labels are padded, find actual lengths
            target_lengths = (labels != 0).sum(dim=1).long()  # Assuming 0 is pad token

        # Transpose for CTC: [time, batch, vocab]
        log_probs = log_probs.transpose(0, 1)

        # Compute CTC loss
        ctc_loss = F.ctc_loss(
            log_probs,
            labels,
            input_lengths,
            target_lengths,
            blank=0,  # Assuming blank token is at index 0
            reduction='mean',
            zero_infinity=True
        )

        return ctc_loss

    def generate(self, input_features: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate predictions (greedy decoding)"""
        with torch.no_grad():
            outputs = self.forward(input_features)
            logits = outputs.logits

            # Greedy decoding
            predictions = torch.argmax(logits, dim=-1)

            return predictions

    def parameters(self):
        """Return model parameters"""
        return super().parameters()

    def train(self, mode: bool = True):
        """Set training mode"""
        return super().train(mode)

    def eval(self):
        """Set evaluation mode"""
        return super().eval()

    def to(self, device):
        """Move model to device"""
        return super().to(device)


class FallbackModelOutput:
    """Output class for fallback model"""

    def __init__(
        self,
        loss: Optional[torch.Tensor] = None,
        logits: torch.Tensor = None,
        hidden_states: Optional[torch.Tensor] = None,
        attentions: Optional[torch.Tensor] = None
    ):
        self.loss = loss
        self.logits = logits
        self.hidden_states = hidden_states
        self.attentions = attentions


def create_fallback_model_and_processor(config: Optional[Dict] = None) -> Tuple[FallbackTwiASR, FallbackProcessor]:
    """
    Create fallback model and processor

    Args:
        config: Optional configuration dictionary

    Returns:
        Tuple of (model, processor)
    """
    vocab_size = config.get('vocab_size', 32) if config else 32

    model = FallbackTwiASR(config)
    processor = FallbackProcessor(vocab_size)

    logger.info("Created fallback model and processor")

    return model, processor


# Test function
def test_fallback_model():
    """Test the fallback model"""
    print("Testing fallback model...")

    # Create model and processor
    model, processor = create_fallback_model_and_processor({'vocab_size': 32})

    # Create dummy audio
    batch_size = 2
    audio_length = 16000  # 1 second at 16kHz
    dummy_audio = torch.randn(batch_size, audio_length)

    # Test processing
    inputs = processor(dummy_audio)
    print(f"Processed input shape: {inputs['input_features'].shape}")

    # Test model forward pass
    outputs = model(inputs['input_features'])
    print(f"Model output logits shape: {outputs.logits.shape}")

    # Test generation
    predictions = model.generate(inputs['input_features'])
    print(f"Predictions shape: {predictions.shape}")

    # Test decoding
    decoded = processor.batch_decode(predictions)
    print(f"Decoded text: {decoded}")

    print("✅ Fallback model test completed successfully!")


if __name__ == "__main__":
    test_fallback_model()
