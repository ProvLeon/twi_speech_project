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

    def __init__(self, n_input_mels: int, n_output_classes: int, rnn_dim: int = 128, n_cnn_layers: int = 2, dropout: float = 0.2):
        """
        Initializes the model layers.

        Args:
            n_input_mels (int): The number of Mel frequency bins in the input spectrogram.
            n_output_classes (int): The number of command classes to predict.
            rnn_dim (int): The hidden dimension size of the GRU layer.
            n_cnn_layers (int): The number of CNN blocks.
            dropout (float): Dropout probability.
        """
        super().__init__()

        self.n_input_mels = n_input_mels
        self.n_output_classes = n_output_classes
        self.rnn_dim = rnn_dim

        # --- CNN Feature Extractor ---
        # Input shape: (batch_size, 1, n_mels, time_steps)
        # We treat the spectrogram as a single-channel image.
        cnn_layers = []
        in_channels = 1
        for i in range(n_cnn_layers):
            cnn_layers.extend([
                nn.Conv2d(in_channels, 32 * (i + 1), kernel_size=(3, 3), padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32 * (i + 1)),
                nn.MaxPool2d(kernel_size=(2, 2)),
                nn.Dropout(dropout)
            ])
            in_channels = 32 * (i + 1)
        self.cnn = nn.Sequential(*cnn_layers)

        # --- Calculate RNN input size ---
        # To determine the input size for the GRU, we pass a dummy tensor through the CNN.
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, n_input_mels, 100) # (B, C, H, W)
            cnn_output = self.cnn(dummy_input)
            b, c, h, w = cnn_output.shape
            self.rnn_input_size = c * h # We collapse channels and height (frequency)

        # --- Recurrent Layer (GRU) ---
        self.gru = nn.GRU(
            input_size=self.rnn_input_size,
            hidden_size=self.rnn_dim,
            num_layers=1,
            batch_first=False, # We will permute the input to (time, batch, features)
            bidirectional=True # Bidirectional helps capture context from both directions
        )

        # --- Classifier Head ---
        self.classifier = nn.Sequential(
            nn.Linear(self.rnn_dim * 2, 128), # *2 because GRU is bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, self.n_output_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor (Mel spectrogram) of shape
                              (batch_size, 1, n_mels, time_steps).

        Returns:
            torch.Tensor: The output logits of shape (batch_size, n_output_classes).
        """
        # 1. Pass through CNN
        # x shape: (batch_size, 1, n_mels, time_steps)
        x = self.cnn(x)
        # x shape after CNN: (batch_size, channels, freq_reduced, time_reduced)

        # 2. Reshape and permute for RNN
        # Collapse the channel and frequency dimensions
        b, c, h, w = x.shape
        x = x.view(b, c * h, w)
        # Permute to (time, batch, features) as expected by the GRU
        x = x.permute(2, 0, 1)  # (width, batch, channels*height)
        # x shape after permute: (time_reduced, batch_size, rnn_input_size)

        # 3. Pass through GRU
        # We only need the final hidden state of the GRU
        # The output contains the hidden state for each time step.
        # h_n contains the final hidden state.
        _output, h_n = self.gru(x)
        # h_n shape: (num_layers * num_directions, batch_size, rnn_dim)
        # which is (2, batch_size, rnn_dim)

        # Concatenate the final forward and backward hidden states
        h_n = h_n.view(self.gru.num_layers, 2, -1, self.rnn_dim) # (num_layers, directions, batch, dim)
        h_n = torch.cat((h_n[-1, 0, :, :], h_n[-1, 1, :, :]), dim=1) # (batch, rnn_dim * 2)
        # h_n shape: (batch_size, rnn_dim * 2)

        # 4. Pass through the classifier
        logits = self.classifier(h_n)
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
