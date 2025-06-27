import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple


class MultiScaleConv1d(nn.Module):
    """Multi-scale 1D convolution for capturing features at different temporal resolutions"""

    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], dilation=1):
        super(MultiScaleConv1d, self).__init__()

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels // len(kernel_sizes),
                     kernel_size=k, padding=k//2, dilation=dilation)
            for k in kernel_sizes
        ])

        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        outputs = [conv(x) for conv in self.convs]
        x = torch.cat(outputs, dim=1)
        x = self.bn(x)
        return self.activation(x)


class AdaptiveAttention(nn.Module):
    """Adaptive attention mechanism with learnable temperature"""

    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(AdaptiveAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1))

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        # Multi-head attention
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with adaptive temperature
        scores = torch.matmul(q, k.transpose(-2, -1)) / (math.sqrt(self.head_dim) * self.temperature)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        output = self.out_proj(attn_output)

        # Residual connection and layer norm
        return self.layer_norm(x + output)


class SqueezeExcitationBlock(nn.Module):
    """Enhanced Squeeze-and-Excitation block with learnable pooling"""

    def __init__(self, channels, reduction=16):
        super(SqueezeExcitationBlock, self).__init__()
        self.channels = channels

        # Learnable global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Excitation network
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, seq_len = x.size()

        # Global pooling (both avg and max)
        avg_pool = self.global_pool(x).view(batch_size, channels)
        max_pool = self.max_pool(x).view(batch_size, channels)

        # Concatenate pooled features
        pooled = torch.cat([avg_pool, max_pool], dim=1)

        # Generate channel attention weights
        attention = self.fc(pooled).view(batch_size, channels, 1)

        # Apply attention
        return x * attention


class ResidualBlock(nn.Module):
    """Advanced residual block with multi-scale convolutions"""

    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5], stride=1, dropout=0.1):
        super(ResidualBlock, self).__init__()

        self.multi_conv = MultiScaleConv1d(in_channels, out_channels, kernel_sizes)
        self.se_block = SqueezeExcitationBlock(out_channels)
        self.dropout = nn.Dropout(dropout)

        # Skip connection
        self.skip_connection = nn.Identity() if in_channels == out_channels else \
                              nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = self.skip_connection(x)

        out = self.multi_conv(x)
        out = self.se_block(out)
        out = self.dropout(out)

        return out + identity


class SuperiorTwiSpeechModel(nn.Module):
    """
    Superior speech recognition model with advanced architectural improvements:
    - Multi-scale convolutions for better feature extraction
    - Adaptive attention with learnable temperature
    - Enhanced squeeze-excitation blocks
    - Progressive feature refinement
    - Advanced regularization techniques
    """

    def __init__(self, input_dim, hidden_dim=256, num_classes=10,
                 num_conv_layers=4, num_attention_layers=3,
                 num_heads=8, dropout=0.1):
        super(SuperiorTwiSpeechModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        print(f"Creating SuperiorTwiSpeechModel with input_dim={input_dim}")

        # Input projection and normalization
        self.input_projection = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim // 2, kernel_size=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Progressive multi-scale convolutional layers
        conv_dims = [hidden_dim // 2] + [hidden_dim * (2**min(i, 2)) // 4 for i in range(num_conv_layers)]
        self.conv_layers = nn.ModuleList([
            ResidualBlock(
                in_channels=conv_dims[i],
                out_channels=conv_dims[i+1],
                kernel_sizes=[3, 5, 7] if i == 0 else [3, 5],
                dropout=dropout
            )
            for i in range(num_conv_layers)
        ])

        # Adaptive temporal pooling
        self.temporal_pool = nn.AdaptiveAvgPool1d(64)  # Fixed temporal dimension

        # Feature dimension matching
        final_conv_dim = conv_dims[-1]
        self.feature_projection = nn.Linear(final_conv_dim, hidden_dim)

        # Transformer-style attention layers
        self.attention_layers = nn.ModuleList([
            AdaptiveAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_attention_layers)
        ])

        # Bidirectional GRU with highway connections
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # Advanced pooling strategy
        self.pooling = AdaptivePooling(hidden_dim * 2)

        # Multi-stage classification head with progressive refinement
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout / 4),

            nn.Linear(hidden_dim // 4, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Advanced weight initialization for optimal convergence"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)

    def forward(self, x):
        # Input validation and adjustment
        if len(x.shape) == 3:
            if x.shape[1] != self.input_dim and x.shape[2] == self.input_dim:
                x = x.transpose(1, 2)

        batch_size = x.size(0)

        # Input projection
        x = self.input_projection(x)

        # Multi-scale convolutional feature extraction
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Temporal pooling to standardize sequence length
        x = self.temporal_pool(x)

        # Prepare for sequential processing (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)
        x = self.feature_projection(x)

        # Apply attention layers
        for attention_layer in self.attention_layers:
            x = attention_layer(x)

        # Bidirectional GRU processing
        gru_output, _ = self.gru(x)

        # Advanced pooling
        pooled_features = self.pooling(gru_output)

        # Classification
        output = self.classifier(pooled_features)

        return output


class AdaptivePooling(nn.Module):
    """Advanced pooling strategy combining multiple pooling methods"""

    def __init__(self, input_dim):
        super(AdaptivePooling, self).__init__()
        self.input_dim = input_dim

        # Learnable pooling weights
        self.pool_weights = nn.Parameter(torch.ones(4))

    def forward(self, x):
        # Different pooling strategies
        avg_pool = torch.mean(x, dim=1)
        max_pool, _ = torch.max(x, dim=1)

        # Attention-based pooling
        attn_weights = F.softmax(torch.mean(x, dim=2), dim=1).unsqueeze(2)
        attn_pool = torch.sum(x * attn_weights, dim=1)

        # Last timestep pooling
        last_pool = x[:, -1, :]

        # Weighted combination
        weights = F.softmax(self.pool_weights, dim=0)
        pooled = (weights[0] * avg_pool +
                 weights[1] * max_pool +
                 weights[2] * attn_pool +
                 weights[3] * last_pool)

        return pooled


class SpecAugment(nn.Module):
    """SpecAugment implementation for regularization during training"""

    def __init__(self, freq_mask_param=15, time_mask_param=35, num_masks=2):
        super(SpecAugment, self).__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_masks = num_masks

    def forward(self, x):
        if not self.training:
            return x

        batch_size, channels, time_steps = x.size()

        # Apply frequency masking
        for _ in range(self.num_masks):
            if channels > self.freq_mask_param:
                f = np.random.randint(0, self.freq_mask_param)
                f0 = np.random.randint(0, channels - f)
                x[:, f0:f0+f, :] = 0

        # Apply time masking
        for _ in range(self.num_masks):
            if time_steps > self.time_mask_param:
                t = np.random.randint(0, self.time_mask_param)
                t0 = np.random.randint(0, time_steps - t)
                x[:, :, t0:t0+t] = 0

        return x
