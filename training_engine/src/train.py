import os
import logging
import json
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

from .data_loader import load_and_prepare_data


from .model import ECommerceCommandModel

from pydub import AudioSegment
from tqdm import tqdm

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
CHECKPOINT_PATH = os.path.join(MODEL_OUTPUT_DIR, 'checkpoint.pth')

# Training settings
LEARNING_RATE = 0.001
BATCH_SIZE = 16
EPOCHS = 30
TEST_SPLIT_SIZE = 0.2
EARLY_STOPPING_PATIENCE = 5  # Number of epochs to wait for improvement before stopping

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
    def __init__(self, df: pd.DataFrame, transform, label_map: dict, sample_rate: int, max_seconds: int, augmentations=None):
        self.df = df
        self.transform = transform
        self.label_map = label_map
        self.sample_rate = sample_rate
        self.max_samples = max_seconds * sample_rate
        self.augmentations = augmentations

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

        # Apply augmentations before resampling and padding
        if self.augmentations:
            waveform_np = waveform.numpy()
            augmented_samples = self.augmentations(samples=waveform_np, sample_rate=self.sample_rate)
            waveform = torch.from_numpy(augmented_samples)

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

    for i, (inputs, labels) in enumerate(tqdm(data_loader, desc="Training")):
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


def save_training_plots(train_history, val_history, metric, output_dir):
    """Saves plots of training and validation metrics."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_history, label=f'Training {metric}')
    plt.plot(val_history, label=f'Validation {metric}')
    plt.title(f'Training and Validation {metric}')
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(output_dir, f'{metric.lower()}_plot.png')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Saved {metric} plot to {save_path}")


def validate(model, data_loader, loss_fn, device):
    """Evaluates the model on the validation set."""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Validation"):
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
    Orchestrates the entire training pipeline with checkpointing and resuming.
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
    num_classes = len(unique_labels)
    logging.info(f"Found {num_classes} unique command classes from 'prompt_text'.")
    with open(LABEL_MAP_PATH, 'w') as f:
        json.dump(label_to_int, f, indent=4)
    logging.info(f"Label map saved to: {LABEL_MAP_PATH}")

    # 3. Split Data
    train_df, val_df = train_test_split(df, test_size=TEST_SPLIT_SIZE, random_state=42, stratify=df['prompt_text'])
    logging.info(f"Training set size: {len(train_df)}, Validation set size: {len(val_df)}")

    # 4. Create Datasets and DataLoaders
    # Define augmentations
    augmentations = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
    ])

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, hop_length=HOP_LENGTH
    )
    # Apply augmentations only to the training set
    train_dataset = AudioCommandDataset(train_df, mel_spectrogram, label_to_int, TARGET_SAMPLE_RATE, MAX_AUDIO_SECONDS, augmentations=augmentations)
    val_dataset = AudioCommandDataset(val_df, mel_spectrogram, label_to_int, TARGET_SAMPLE_RATE, MAX_AUDIO_SECONDS)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. Initialize Model, Loss, and Optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    model = ECommerceCommandModel(n_input_mels=N_MELS, n_output_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()
    # Add a learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    # --- Checkpoint Loading ---
    start_epoch = 0
    best_val_accuracy = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []
    gdrive_checkpoint_path = '/content/drive/MyDrive/twi_speech_model_checkpoints/checkpoint.pth'

    # If in Colab, try to copy checkpoint from Drive first
    try:
        import sys
        if 'google.colab' in sys.modules and os.path.exists(gdrive_checkpoint_path):
            import shutil
            logging.info("Found checkpoint in Google Drive. Copying to local environment...")
            shutil.copy(gdrive_checkpoint_path, CHECKPOINT_PATH)
    except Exception as e:
        logging.warning(f"Could not copy checkpoint from Google Drive: {e}")

    if os.path.exists(CHECKPOINT_PATH):
        logging.info(f"Loading checkpoint from {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logging.info(f"Resuming training from epoch {start_epoch}")
    else:
        logging.info("No checkpoint found. Starting training from scratch.")

    # 6. Training Loop
    logging.info("--- Starting Training Loop ---")
    for epoch in range(start_epoch, EPOCHS):
        logging.info(f"--- Epoch {epoch+1}/{EPOCHS} ---")
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)

        # Log metrics
        logging.info(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Store history for plotting
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Step the scheduler
        scheduler.step(val_loss)

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logging.info(f"New best model saved with validation loss: {best_val_loss:.4f} and accuracy: {best_val_accuracy:.4f}")
            patience_counter = 0  # Reset patience
        else:
            patience_counter += 1

        # Save a checkpoint at the end of every epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_accuracy': best_val_accuracy,
            'best_val_loss': best_val_loss,
        }
        torch.save(checkpoint, CHECKPOINT_PATH)
        logging.info(f"Saved checkpoint for epoch {epoch}")

        # If in Colab, copy checkpoint to Drive
        try:
            import sys
            if 'google.colab' in sys.modules:
                import shutil
                gdrive_dest_dir = os.path.dirname(gdrive_checkpoint_path)
                os.makedirs(gdrive_dest_dir, exist_ok=True)
                shutil.copy(CHECKPOINT_PATH, gdrive_dest_dir)
                logging.info(f"Copied checkpoint for epoch {epoch} to Google Drive.")
        except Exception as e:
            logging.warning(f"Could not copy checkpoint to Google Drive for epoch {epoch}: {e}")

        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            logging.info(f"Validation loss has not improved for {EARLY_STOPPING_PATIENCE} epochs. Stopping early.")
            break

    logging.info("--- Training Complete ---")
    # Save the training plots
    save_training_plots(train_loss_history, val_loss_history, "Loss", MODEL_OUTPUT_DIR)
    save_training_plots(train_acc_history, val_acc_history, "Accuracy", MODEL_OUTPUT_DIR)
    logging.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
    logging.info(f"Trained model and label map saved in: {MODEL_OUTPUT_DIR}")

    # --- Final Step: Copy artifacts to Google Drive if in Colab ---
    try:
        import sys
        import shutil
        if 'google.colab' in sys.modules:
            logging.info("Detected Google Colab environment. Copying model and label map to Google Drive...")

            from google.colab import drive
            drive.mount('/content/drive', force_remount=True)

            gdrive_dest_dir = '/content/drive/MyDrive/twi_speech_model_checkpoints/'
            os.makedirs(gdrive_dest_dir, exist_ok=True)

            # Copy the best model
            if os.path.exists(MODEL_SAVE_PATH):
                shutil.copy(MODEL_SAVE_PATH, gdrive_dest_dir)
                logging.info(f"Successfully copied model to {os.path.join(gdrive_dest_dir, 'best_model.pth')}")

            # Copy the label map
            if os.path.exists(LABEL_MAP_PATH):
                shutil.copy(LABEL_MAP_PATH, gdrive_dest_dir)
                logging.info(f"Successfully copied label map to {os.path.join(gdrive_dest_dir, 'label_map.json')}")

    except ImportError:
        # This will trigger if not in Colab, which is fine.
        pass
    except Exception as e:
        logging.error(f"Failed to copy files to Google Drive: {e}")

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
