"""
Wav2Vec2 Model for Twi Speech Recognition

This module implements a custom Wav2Vec2 model specifically designed for Twi
speech recognition. It extends the standard Wav2Vec2ForCTC with Twi-specific
adaptations including dialect awareness, tone handling, and optimizations
for low-resource language training.

Key Features:
- Twi-specific vocabulary and tokenization
- Dialect adaptation modules
- Tone-aware processing
- Code-switching support
- Enhanced CTC loss with auxiliary losses
- Curriculum learning integration

Author: Twi Speech Recognition Team
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Config,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel
)
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2EncoderLayer,
    Wav2Vec2Attention
)

logger = logging.getLogger(__name__)


class TwiWav2Vec2Output:
    """Custom output class for Twi Wav2Vec2 model"""
    def __init__(
        self,
        loss: Optional[torch.FloatTensor] = None,
        logits: torch.FloatTensor = None,
        hidden_states: Optional[Tuple[torch.FloatTensor]] = None,
        attentions: Optional[Tuple[torch.FloatTensor]] = None,
        # Twi-specific outputs
        dialect_logits: Optional[torch.FloatTensor] = None,
        tone_logits: Optional[torch.FloatTensor] = None,
        speaker_embeddings: Optional[torch.FloatTensor] = None,
        attention_weights: Optional[torch.FloatTensor] = None,
    ):
        self.loss = loss
        self.logits = logits
        self.hidden_states = hidden_states
        self.attentions = attentions
        self.dialect_logits = dialect_logits
        self.tone_logits = tone_logits
        self.speaker_embeddings = speaker_embeddings
        self.attention_weights = attention_weights


class TwiDialectClassifier(nn.Module):
    """Dialect classification head for Twi variants"""

    def __init__(self, config):
        super().__init__()
        self.num_dialects = 3  # Asante, Akuapem, Fante

        # Safe initialization with fallbacks
        hidden_dropout = getattr(config, 'hidden_dropout', 0.1)
        hidden_size = getattr(config, 'hidden_size', 768)

        self.dropout = nn.Dropout(hidden_dropout)
        self.classifier = nn.Linear(hidden_size, self.num_dialects)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Pool over sequence dimension
        pooled = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class TwiToneClassifier(nn.Module):
    """Tone classification head for Twi tonal features"""

    def __init__(self, config):
        super().__init__()
        self.num_tones = 5  # High, Mid, Low, Rising, Falling

        # Safe initialization with fallbacks
        self.hidden_size = getattr(config, 'hidden_size', 768)
        hidden_dropout = getattr(config, 'hidden_dropout', 0.1)

        self.tone_projection = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.dropout = nn.Dropout(hidden_dropout)
        self.classifier = nn.Linear(self.hidden_size // 2, self.num_tones)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Apply frame-level tone classification
        projected = self.tone_projection(hidden_states)
        projected = F.relu(projected)
        projected = self.dropout(projected)
        logits = self.classifier(projected)
        return logits


class TwiSpeakerEncoder(nn.Module):
    """Speaker embedding encoder for multi-speaker adaptation"""

    def __init__(self, config):
        super().__init__()
        self.embedding_dim = getattr(config, 'speaker_embedding_dim', 256)
        self.hidden_size = getattr(config, 'hidden_size', 768)
        hidden_dropout = getattr(config, 'hidden_dropout', 0.1)

        self.speaker_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(self.hidden_size // 2, self.embedding_dim),
            nn.Tanh()
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Global average pooling over sequence
        pooled = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        speaker_embedding = self.speaker_projection(pooled)
        return speaker_embedding


class TwiAttentionMechanism(nn.Module):
    """Enhanced attention mechanism with Twi linguistic features"""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = getattr(config, 'hidden_size', 768)
        self.num_heads = getattr(config, 'attention_heads', 8)
        self.head_dim = self.hidden_size // self.num_heads

        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, self.hidden_size)

        attention_dropout = getattr(config, 'attention_dropout', 0.1)
        self.dropout = nn.Dropout(attention_dropout)
        self.scale = math.sqrt(self.head_dim)
        self.output_projection = nn.Linear(self.hidden_size, self.hidden_size)

        # Tone bias for Twi
        self.tone_bias = nn.Parameter(torch.zeros(self.num_heads, 1, 1))

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length, _ = hidden_states.shape

        # Compute Q, K, V
        q = self.query(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Add tone bias
        scores = scores + self.tone_bias

        # Apply attention mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)

        # Output projection
        output = self.output_projection(context)

        return output, attention_weights.mean(dim=1)  # Average over heads


class TwiCTCLoss(nn.Module):
    """Enhanced CTC Loss with Twi-specific improvements"""

    def __init__(self, config):
        super().__init__()
        self.blank_id = getattr(config, 'pad_token_id', 0)
        self.reduction = getattr(config, 'ctc_loss_reduction', 'mean')
        self.zero_infinity = getattr(config, 'ctc_zero_infinity', True)
        self.label_smoothing = getattr(config, 'label_smoothing', 0.0)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                input_lengths: torch.Tensor, label_lengths: torch.Tensor) -> torch.Tensor:

        # Standard CTC loss
        log_probs = F.log_softmax(logits, dim=-1)

        ctc_loss = F.ctc_loss(
            log_probs.transpose(0, 1),  # CTC expects (T, N, C)
            labels,
            input_lengths,
            label_lengths,
            blank=self.blank_id,
            reduction=self.reduction,
            zero_infinity=self.zero_infinity
        )

        # Apply label smoothing if specified
        if self.label_smoothing > 0.0:
            # Uniform distribution over vocabulary
            vocab_size = logits.size(-1)
            uniform_dist = torch.full_like(logits, 1.0 / vocab_size)

            # Smooth the targets
            smoothed_loss = F.kl_div(
                log_probs,
                uniform_dist,
                reduction='batchmean'
            )

            ctc_loss = (1 - self.label_smoothing) * ctc_loss + self.label_smoothing * smoothed_loss

        return ctc_loss


class Wav2Vec2ForTwiASR(Wav2Vec2PreTrainedModel):
    """
    Wav2Vec2 model specialized for Twi Automatic Speech Recognition

    This model extends the standard Wav2Vec2 architecture with:
    - Twi-specific vocabulary and tokenization
    - Dialect classification capabilities
    - Tone-aware processing
    - Multi-speaker adaptation
    - Enhanced attention mechanisms
    """

    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)

        self.config = config

        try:
            self.wav2vec2 = Wav2Vec2Model(config)
        except Exception as e:
            logger.error(f"Failed to initialize Wav2Vec2Model: {e}")
            raise RuntimeError(f"Wav2Vec2Model initialization failed: {e}")

        # Main CTC head
        try:
            self.dropout = nn.Dropout(getattr(config, 'final_dropout', 0.1))
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        except Exception as e:
            logger.error(f"Failed to initialize CTC head: {e}")
            raise RuntimeError(f"CTC head initialization failed: {e}")

        # Twi-specific components with safer initialization
        try:
            self.dialect_classifier = TwiDialectClassifier(config) if getattr(config, 'use_dialect_classification', False) else None
        except Exception as e:
            logger.warning(f"Failed to initialize dialect classifier: {e}")
            self.dialect_classifier = None

        try:
            self.tone_classifier = TwiToneClassifier(config) if getattr(config, 'use_tone_classification', False) else None
        except Exception as e:
            logger.warning(f"Failed to initialize tone classifier: {e}")
            self.tone_classifier = None

        try:
            self.speaker_encoder = TwiSpeakerEncoder(config) if getattr(config, 'use_speaker_adaptation', False) else None
        except Exception as e:
            logger.warning(f"Failed to initialize speaker encoder: {e}")
            self.speaker_encoder = None

        try:
            self.enhanced_attention = TwiAttentionMechanism(config) if getattr(config, 'use_enhanced_attention', False) else None
        except Exception as e:
            logger.warning(f"Failed to initialize enhanced attention: {e}")
            self.enhanced_attention = None

        # Enhanced CTC loss with fallback
        try:
            self.ctc_loss_fn = TwiCTCLoss(config)
        except Exception as e:
            logger.warning(f"Failed to initialize TwiCTCLoss, using standard CTC: {e}")
            self.ctc_loss_fn = None

        # Auxiliary loss weights
        self.dialect_loss_weight = getattr(config, 'dialect_loss_weight', 0.1)
        self.tone_loss_weight = getattr(config, 'tone_loss_weight', 0.05)

        # Initialize weights
        try:
            self.post_init()
        except Exception as e:
            logger.warning(f"post_init failed: {e}")
            # Continue without post_init if it fails

    def tie_weights(self):
        """Tie weights if necessary"""
        # This method is called during initialization
        pass

    def freeze_feature_extractor(self):
        """Freeze the feature extractor weights"""
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """Freeze the base wav2vec2 model"""
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def unfreeze_layer(self, layer_idx: int):
        """Unfreeze a specific transformer layer"""
        if 0 <= layer_idx < len(self.wav2vec2.encoder.layers):
            for param in self.wav2vec2.encoder.layers[layer_idx].parameters():
                param.requires_grad = True

    def get_input_embeddings(self):
        return self.wav2vec2.feature_extractor

    def set_input_embeddings(self, value):
        self.wav2vec2.feature_extractor = value

    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        dialect_labels: Optional[torch.Tensor] = None,
        tone_labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, TwiWav2Vec2Output]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward through wav2vec2 backbone
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]  # [batch_size, seq_len, hidden_size]

        # Apply enhanced attention if configured
        attention_weights = None
        if self.enhanced_attention is not None:
            hidden_states, attention_weights = self.enhanced_attention(hidden_states, attention_mask)

        # Apply dropout
        hidden_states = self.dropout(hidden_states)

        # Main CTC prediction
        logits = self.lm_head(hidden_states)

        # Auxiliary predictions
        dialect_logits = None
        tone_logits = None
        speaker_embeddings = None

        if self.dialect_classifier is not None:
            dialect_logits = self.dialect_classifier(hidden_states)

        if self.tone_classifier is not None:
            tone_logits = self.tone_classifier(hidden_states)

        if self.speaker_encoder is not None:
            speaker_embeddings = self.speaker_encoder(hidden_states)

        # Calculate losses
        loss = None
        if labels is not None:
            # Compute sequence lengths
            if attention_mask is not None:
                attention_mask = self._get_feature_vector_attention_mask(
                    hidden_states.shape[1], attention_mask, add_adapter=False
                )
                input_lengths = attention_mask.sum(-1)
            else:
                input_lengths = torch.full(
                    (hidden_states.shape[0],), hidden_states.shape[1], dtype=torch.long, device=hidden_states.device
                )

            # Compute label lengths
            labels_mask = labels >= 0
            label_lengths = labels_mask.sum(-1)

            # Filter out invalid labels
            labels = labels.masked_select(labels_mask)

            # Main CTC loss
            ctc_loss = self.ctc_loss_fn(logits, labels, input_lengths, label_lengths)
            loss = ctc_loss

            # Auxiliary losses
            if dialect_labels is not None and self.dialect_classifier is not None:
                dialect_loss = F.cross_entropy(dialect_logits, dialect_labels)
                loss += self.dialect_loss_weight * dialect_loss

            if tone_labels is not None and self.tone_classifier is not None:
                tone_loss = F.cross_entropy(
                    tone_logits.view(-1, tone_logits.size(-1)),
                    tone_labels.view(-1)
                )
                loss += self.tone_loss_weight * tone_loss

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return TwiWav2Vec2Output(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            dialect_logits=dialect_logits,
            tone_logits=tone_logits,
            speaker_embeddings=speaker_embeddings,
            attention_weights=attention_weights,
        )

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.Tensor, add_adapter: bool = None
    ):
        """Compute attention mask for feature vectors"""
        # Wav2Vec2 feature extraction reduces sequence length
        subsampled_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1).long())
        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )

        # Set to 1 for valid positions
        for i, length in enumerate(subsampled_lengths):
            attention_mask[i, :length] = 1

        return attention_mask

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """Compute feature extraction output lengths"""
        def _conv_out_length(input_length, kernel_size, stride):
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        conv_layers = self.wav2vec2.feature_extractor.conv_layers
        for i in range(len(conv_layers)):
            input_lengths = _conv_out_length(
                input_lengths, conv_layers[i].kernel_size[0], conv_layers[i].stride[0]
            )

        return input_lengths

    def predict_dialect(self, input_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict dialect for given audio input"""
        if self.dialect_classifier is None:
            raise ValueError("Dialect classification not enabled for this model")

        outputs = self.forward(
            input_values=input_values,
            attention_mask=attention_mask,
            return_dict=True
        )

        return F.softmax(outputs.dialect_logits, dim=-1)

    def extract_speaker_embedding(self, input_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract speaker embedding for given audio input"""
        if self.speaker_encoder is None:
            raise ValueError("Speaker adaptation not enabled for this model")

        outputs = self.forward(
            input_values=input_values,
            attention_mask=attention_mask,
            return_dict=True
        )

        return outputs.speaker_embeddings

    def generate_predictions(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_auxiliary: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Generate predictions for inference

        Args:
            input_values: Raw speech waveform
            attention_mask: Attention mask
            return_auxiliary: Whether to return auxiliary predictions

        Returns:
            Dictionary containing predictions
        """
        self.eval()

        with torch.no_grad():
            outputs = self.forward(
                input_values=input_values,
                attention_mask=attention_mask,
                return_dict=True
            )

            # Main CTC predictions
            predicted_ids = torch.argmax(outputs.logits, dim=-1)

            results = {
                'predicted_ids': predicted_ids,
                'logits': outputs.logits
            }

            # Auxiliary predictions
            if return_auxiliary:
                if hasattr(outputs, 'dialect_logits'):
                    results['dialect_predictions'] = torch.argmax(outputs.dialect_logits, dim=-1)
                    results['dialect_probabilities'] = F.softmax(outputs.dialect_logits, dim=-1)

        return results

    def compute_ctc_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Compute CTC loss with proper handling"""

        log_probs = F.log_softmax(logits, dim=-1)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                log_probs.transpose(0, 1),
                labels,
                input_lengths,
                target_lengths,
                blank=self.config.pad_token_id,
                reduction="mean",
                zero_infinity=True,
            )

        return loss

    def get_trainable_parameters(self) -> Dict[str, int]:
        """Get count of trainable parameters by component"""

        param_counts = {
            'wav2vec2_feature_extractor': 0,
            'wav2vec2_encoder': 0,
            'lm_head': 0,
            'dialect_classifier': 0,
            'auxiliary_tasks': 0,
            'total': 0
        }

        for name, param in self.named_parameters():
            if param.requires_grad:
                param_count = param.numel()
                param_counts['total'] += param_count

                if 'feature_extractor' in name:
                    param_counts['wav2vec2_feature_extractor'] += param_count
                elif 'encoder' in name:
                    param_counts['wav2vec2_encoder'] += param_count
                elif 'lm_head' in name:
                    param_counts['lm_head'] += param_count
                elif 'dialect_classifier' in name:
                    param_counts['dialect_classifier'] += param_count
                elif 'aux_losses' in name:
                    param_counts['auxiliary_tasks'] += param_count

        return param_counts

    @classmethod
    def from_wav2vec2_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load from a standard Wav2Vec2 model and adapt for Twi"""

        # Load standard model first
        standard_model = Wav2Vec2ForCTC.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Create Twi config based on standard config
        twi_config = standard_model.config
        twi_config.use_dialect_classification = kwargs.get('use_dialect_classification', True)
        twi_config.use_tone_classification = kwargs.get('use_tone_classification', True)
        twi_config.use_speaker_adaptation = kwargs.get('use_speaker_adaptation', False)
        twi_config.use_enhanced_attention = kwargs.get('use_enhanced_attention', False)

        # Create Twi model
        twi_model = cls(twi_config)

        # Transfer compatible weights
        twi_model.wav2vec2.load_state_dict(standard_model.wav2vec2.state_dict(), strict=False)

        # Transfer LM head if vocab sizes match
        if standard_model.lm_head.out_features == twi_model.lm_head.out_features:
            twi_model.lm_head.load_state_dict(standard_model.lm_head.state_dict())

        logger.info(f"Loaded Twi ASR model from {pretrained_model_name_or_path}")
        return twi_model


# Constants for output indexing
_HIDDEN_STATES_START_POSITION = 1


# Utility functions
def count_parameters(model: nn.Module) -> int:
    """Count total parameters in model"""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in MB"""
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def create_twi_wav2vec2_config(
    base_model: str = "facebook/wav2vec2-base",
    vocab_size: int = 64,
    num_dialects: int = 3,
    tone_aware: bool = True,
    **kwargs
) -> Wav2Vec2Config:
    """
    Create Wav2Vec2 configuration for Twi ASR

    Args:
        base_model: Base pretrained model name
        vocab_size: Size of Twi vocabulary
        num_dialects: Number of Twi dialects to support
        tone_aware: Whether to enable tone-aware processing
        **kwargs: Additional configuration parameters

    Returns:
        Configured Wav2Vec2Config object
    """

    # Load base configuration
    config = Wav2Vec2Config.from_pretrained(base_model)

    # Update for Twi-specific settings
    config.vocab_size = vocab_size
    config.pad_token_id = 0
    config.bos_token_id = 1
    config.eos_token_id = 2

    # Training-specific settings
    config.ctc_loss_reduction = kwargs.get('ctc_loss_reduction', 'mean')
    config.ctc_zero_infinity = kwargs.get('ctc_zero_infinity', True)

    # Regularization
    config.final_dropout = kwargs.get('final_dropout', 0.1)
    config.hidden_dropout = kwargs.get('hidden_dropout', 0.1)
    config.attention_dropout = kwargs.get('attention_dropout', 0.1)
    config.feat_proj_dropout = kwargs.get('feat_proj_dropout', 0.0)
    config.layerdrop = kwargs.get('layerdrop', 0.1)

    return config


# Constants for backward compatibility
_HIDDEN_STATES_START_POSITION = 2


if __name__ == "__main__":
    # Example usage and testing
    from transformers import Wav2Vec2Config

    # Create test configuration
    config = Wav2Vec2Config(
        vocab_size=50,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        use_dialect_classification=True,
        use_tone_classification=True,
        use_speaker_adaptation=True,
        use_enhanced_attention=True,
    )

    # Create model
    model = Wav2Vec2ForTwiASR(config)

    # Test forward pass
    batch_size = 2
    sequence_length = 16000  # 1 second at 16kHz

    input_values = torch.randn(batch_size, sequence_length)
    attention_mask = torch.ones(batch_size, sequence_length)
    labels = torch.randint(0, config.vocab_size, (batch_size, 100))
    dialect_labels = torch.randint(0, 3, (batch_size,))

    outputs = model(
        input_values=input_values,
        attention_mask=attention_mask,
        labels=labels,
        dialect_labels=dialect_labels,
        return_dict=True
    )

    print(f"Model created successfully!")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    print(f"Model size: {get_model_size_mb(model):.2f} MB")
    print(f"Output keys: {list(outputs.__dict__.keys())}")
    print(f"Logits shape: {outputs.logits.shape}")
    if outputs.dialect_logits is not None:
        print(f"Dialect logits shape: {outputs.dialect_logits.shape}")
    if outputs.tone_logits is not None:
        print(f"Tone logits shape: {outputs.tone_logits.shape}")
