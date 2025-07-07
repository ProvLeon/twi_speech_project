import os
import logging
import json
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from .model import ECommerceCommandModel

from pydub import AudioSegment

# --- Basic Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Training and Audio Hyperparameters ---
# Paths
MODEL_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'e_commerce_model')
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_OUTPUT_DIR, 'best_model.pth')
LABEL_MAP_PATH = os.path.join(MODEL_OUTPUT_DIR, 'label_map.json')

# Training settings
LEARNING_RATE = 0.001
BATCH_SIZE = 16
EPOCHS = 30
TEST_SPLIT_SIZE = 0.2

# Audio processing settings
TARGET_SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 5
N_MELS = 128  # Number of Mel frequency bins
N_FFT = 1024  # Size of the FFT window
HOP_LENGTH = 512 # Hop length for the FFT

# --- Custom PyTorch Dataset ---
class AudioCommandDataset(Dataset):
    """
    A custom PyTorch Dataset to load, preprocess, and serve audio data for training/testing.
    Robust to missing/corrupt files and always returns mono waveform.
    """
    def __init__(self, df: pd.DataFrame, transform, label_map: dict, sample_rate: int, max_seconds: int):
        self.df = df
        self.transform = transform
        self.label_map = label_map
        self.sample_rate = sample_rate
        self.max_samples = max_seconds * sample_rate

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        audio_path = self.df.iloc[idx]['local_path']
        label_str = self.df.iloc[idx]['prompt_text']
        label_int = self.label_map[label_str]

        # Check file existence and minimum size (1KB)
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1024:
            logging.error(f"Audio file missing or too small: {audio_path}")
            return torch.zeros(1, N_MELS, self.max_samples // HOP_LENGTH + 1), 0

        try:
            if audio_path.endswith('.m4a'):
                audio = AudioSegment.from_file(audio_path)
                audio = audio.set_frame_rate(self.sample_rate).set_channels(1)
                samples = np.array(audio.get_array_of_samples()).astype(np.float32) / (2**15)
                waveform = torch.from_numpy(samples).unsqueeze(0) if len(samples.shape) == 1 else torch.from_numpy(samples)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                sr = self.sample_rate
            else:
                waveform, sr = torchaudio.load(audio_path)
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                elif waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
        except Exception as e:
            logging.error(f"Error loading audio file {audio_path}: {e}")
            return torch.zeros(1, N_MELS, self.max_samples // HOP_LENGTH + 1), 0

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        if waveform.shape[1] > self.max_samples:
            waveform = waveform[:, :self.max_samples]
        else:
            padding_needed = self.max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding_needed))
        spectrogram = self.transform(waveform)
        # Ensure output shape [1, n_mels, time]
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)
        elif spectrogram.shape[0] != 1:
            spectrogram = torch.mean(spectrogram, dim=0, keepdim=True)
        return spectrogram, label_int


# --- Training and Evaluation Functions ---
def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    """Runs a single training epoch."""
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for i, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track stats
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy


def validate(model, data_loader, loss_fn, device):
    """Evaluates the model on the validation set."""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy


# --- Main Orchestration ---
def run_training(metadata_csv=None):
    """
    Orchestrates the entire training pipeline.
    If metadata_csv is provided, loads data from CSV instead of fetching from DB.
    """
    logging.info("--- Starting E-commerce Command Model Training ---")

    # 1. Load data
    if metadata_csv is not None and os.path.exists(metadata_csv):
        df = pd.read_csv(metadata_csv)
        logging.info(f"Loaded metadata from {metadata_csv} ({len(df)} records).")
    else:
        df = load_and_prepare_data()
        if df is not None and not df.empty:
            logging.info("Fetched data from database.")
        else:
            logging.error("Data loading failed. Aborting training.")
            return

    # 2. Encode Labels
    unique_labels = df['prompt_text'].unique()
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_to_label = {i: label for label, i in label_to_int.items()}
    num_classes = len(unique_labels)
    logging.info(f"Found {num_classes} unique command classes from 'prompt_text'.")

    # Save the label map for later use during inference
    with open(LABEL_MAP_PATH, 'w') as f:
        json.dump(label_to_int, f, indent=4)
    logging.info(f"Label map saved to: {LABEL_MAP_PATH}")

    # 3. Split Data
    train_df, val_df = train_test_split(df, test_size=TEST_SPLIT_SIZE, random_state=42, stratify=df['prompt_text'])
    logging.info(f"Training set size: {len(train_df)}, Validation set size: {len(val_df)}")

    # 4. Create Datasets and DataLoaders
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE,
        n_fft=N_FFT,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH
    )

    train_dataset = AudioCommandDataset(train_df, mel_spectrogram, label_to_int, TARGET_SAMPLE_RATE, MAX_AUDIO_SECONDS)
    val_dataset = AudioCommandDataset(val_df, mel_spectrogram, label_to_int, TARGET_SAMPLE_RATE, MAX_AUDIO_SECONDS)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. Initialize Model, Loss, and Optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    model = ECommerceCommandModel(n_input_mels=N_MELS, n_output_classes=num_classes).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. Training Loop
    best_val_accuracy = 0.0
    logging.info("--- Starting Training Loop ---")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)

        logging.info(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Save the best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logging.info(f"New best model saved with validation accuracy: {best_val_accuracy:.4f}")

    logging.info("--- Training Complete ---")
    logging.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
    logging.info(f"Trained model and label map saved in: {MODEL_OUTPUT_DIR}")

def run_testing(metadata_csv=None):
    """
    Loads the best model and evaluates it on the validation set.
    """
    logging.info("--- Starting Model Evaluation (Test Mode) ---")
    if metadata_csv is None or not os.path.exists(metadata_csv):
        logging.error("Metadata CSV not found. Cannot run test mode.")
        return

    df = pd.read_csv(metadata_csv)
    if df.empty:
        logging.error("No data found in metadata CSV.")
        return

    # Load label map
    if not os.path.exists(LABEL_MAP_PATH):
        logging.error("Label map not found. Train the model first.")
        return
    with open(LABEL_MAP_PATH, 'r') as f:
        label_to_int = json.load(f)
    int_to_label = {i: label for label, i in label_to_int.items()}
    num_classes = len(label_to_int)

    # Split data (use same split as training)
    _, val_df = train_test_split(df, test_size=TEST_SPLIT_SIZE, random_state=42, stratify=df['prompt_text'])
    logging.info(f"Validation set size: {len(val_df)}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE,
        n_fft=N_FFT,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH
    )
    val_dataset = AudioCommandDataset(val_df, mel_spectrogram, label_to_int, TARGET_SAMPLE_RATE, MAX_AUDIO_SECONDS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ECommerceCommandModel(n_input_mels=N_MELS, n_output_classes=num_classes).to(device)
    if not os.path.exists(MODEL_SAVE_PATH):
        logging.error("Trained model not found. Train the model first.")
        return
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = correct_predictions / total_predictions
    logging.info(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Optionally print a confusion matrix or classification report
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=[int_to_label[i] for i in sorted(int_to_label)]))
        print("\nConfusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))
    except ImportError:
        logging.info("Install scikit-learn to see classification report and confusion matrix.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="E-commerce Command Recognition Training/Testing")
    parser.add_argument('--mode', choices=['train', 'test'], default='train', help="Mode: train or test")
    parser.add_argument('--metadata_csv', type=str, default=None, help="Path to metadata CSV (optional)")
    args = parser.parse_args()

    if args.mode == 'train':
        run_training(metadata_csv=args.metadata_csv)
    elif args.mode == 'test':
        run_testing(metadata_csv=args.metadata_csv)
