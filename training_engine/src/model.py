import torch
import torch.nn as nn
import torch.nn.functional as F

class ECommerceCommandModel(nn.Module):
    """
    A simple but effective Convolutional Recurrent Neural Network (CRNN) for
    speech command recognition, tailored for e-commerce actions.

    The model architecture consists of:
    1. CNN layers to extract features from Mel spectrograms.
    2. A GRU layer to model the temporal dependencies in the extracted features.
    3. A linear classifier to predict the command category.
    """

    def __init__(self, n_input_mels: int, n_output_classes: int, rnn_dim: int = 256, n_cnn_layers: int = 3, dropout: float = 0.3):
        """
        Initializes the model layers with enhanced architecture for speech commands.

        Args:
            n_input_mels (int): The number of Mel frequency bins in the input spectrogram.
            n_output_classes (int): The number of command classes to predict.
            rnn_dim (int): The hidden dimension size of the GRU layer. Default: 256
            n_cnn_layers (int): The number of CNN blocks. Default: 3
            dropout (float): Dropout probability. Default: 0.3
        """
        super().__init__()

        self.n_input_mels = n_input_mels
        self.n_output_classes = n_output_classes
        self.rnn_dim = rnn_dim

        # --- CNN Feature Extractor ---
        # Input shape: (batch_size, 1, n_mels, time_steps)
        # We treat the spectrogram as a single-channel image.
        # Enhanced CNN architecture with residual connections and attention
        cnn_layers = []
        in_channels = 1
        channel_multipliers = [64, 96, 128]  # Progressive channel increase

        for i in range(n_cnn_layers):
            out_channels = channel_multipliers[i] if i < len(channel_multipliers) else 128

            # Depthwise separable convolution for efficiency
            cnn_layers.extend([
                # Depthwise convolution
                nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),

                # Pointwise convolution
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),

                # Max pooling with smaller kernel for first layer
                nn.MaxPool2d(kernel_size=(2, 1) if i == 0 else (2, 2)),
                nn.Dropout(dropout * 0.5)  # Less aggressive dropout in CNN
            ])
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # Attention mechanism for better feature selection
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid()
        )

        # --- Calculate RNN input size ---
        # To determine the input size for the GRU, we pass a dummy tensor through the CNN.
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, n_input_mels, 100) # (B, C, H, W)
            cnn_output = self.cnn(dummy_input)

            # Apply attention
            b, c, h, w = cnn_output.shape
            attention_weights = self.attention(cnn_output).view(b, c, 1, 1)
            attended_output = cnn_output * attention_weights

            self.rnn_input_size = c * h # We collapse channels and height (frequency)

        # --- Enhanced Recurrent Layer ---
        # Use LSTM instead of GRU for better long-term dependencies
        self.lstm = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=self.rnn_dim,
            num_layers=2,  # Deeper LSTM
            batch_first=False,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0,
        )

        # Layer normalization for better training stability
        self.layer_norm = nn.LayerNorm(self.rnn_dim * 2)

        # --- Enhanced Classifier Head with residual connection ---
        self.classifier = nn.Sequential(
            nn.Linear(self.rnn_dim * 2, 512),  # Larger hidden layer
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),  # Less dropout in final layers

            nn.Linear(256, self.n_output_classes)
        )

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LSTM, nn.GRU)):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor (Mel spectrogram) of shape
                              (batch_size, 1, n_mels, time_steps).

        Returns:
            torch.Tensor: The output logits of shape (batch_size, n_output_classes).
        """
        # 1. Pass through CNN with attention
        # x shape: (batch_size, 1, n_mels, time_steps)
        x = self.cnn(x)
        # x shape after CNN: (batch_size, channels, freq_reduced, time_reduced)

        # Apply attention mechanism
        b, c, h, w = x.shape
        attention_weights = self.attention(x).view(b, c, 1, 1)
        x = x * attention_weights

        # 2. Reshape and permute for RNN
        # Collapse the channel and frequency dimensions
        b, c, h, w = x.shape
        x = x.view(b, c * h, w)
        # Permute to (time, batch, features) as expected by the GRU
        x = x.permute(2, 0, 1)  # (width, batch, channels*height)
        # x shape after permute: (time_reduced, batch_size, rnn_input_size)

        # 3. Pass through LSTM with enhanced processing
        lstm_output, (h_n, c_n) = self.lstm(x)
        # h_n shape: (num_layers * num_directions, batch_size, rnn_dim)

        # Use attention over all time steps instead of just final hidden state
        # This helps with variable-length sequences
        lstm_output = lstm_output.permute(1, 0, 2)  # (batch, time, features)

        # Compute attention weights over time steps
        attention_scores = torch.mean(lstm_output, dim=2, keepdim=True)  # (batch, time, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)

        # Apply attention
        attended_output = torch.sum(lstm_output * attention_weights, dim=1)  # (batch, features)

        # Apply layer normalization
        attended_output = self.layer_norm(attended_output)

        # 4. Pass through the enhanced classifier
        logits = self.classifier(attended_output)
        # logits shape: (batch_size, n_output_classes)

        return logits


# --- Example Usage ---
if __name__ == '__main__':
    # --- Configuration for the test ---
    N_MELS = 80         # Standard number of Mel bands
    N_CLASSES = 10      # Example: 10 e-commerce commands ("buy", "add to cart", etc.)
    TIME_STEPS = 150    # Example length of the audio spectrogram
    BATCH_SIZE = 4

    # --- Create a dummy input tensor ---
    # This simulates a batch of Mel spectrograms
    dummy_spectrogram = torch.randn(BATCH_SIZE, 1, N_MELS, TIME_STEPS)

    # --- Instantiate the model ---
    print("--- Model Instantiation ---")
    model = ECommerceCommandModel(n_input_mels=N_MELS, n_output_classes=N_CLASSES)
    print(model)
    print(f"\nTotal number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- Perform a forward pass ---
    print("\n--- Forward Pass Test ---")
    print(f"Input shape:  {dummy_spectrogram.shape}")
    output_logits = model(dummy_spectrogram)
    print(f"Output shape: {output_logits.shape}")

    # --- Verify output ---
    assert output_logits.shape == (BATCH_SIZE, N_CLASSES), "Output shape is incorrect!"
    print("\nModel forward pass test successful!")

    # --- Check probabilities (optional) ---
    probabilities = F.softmax(output_logits, dim=1)
    print("\nExample output probabilities (for the first item in the batch):")
    print(probabilities[0].detach().numpy())
