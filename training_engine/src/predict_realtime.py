import os
import json
import torch
import torchaudio
import numpy as np
import sounddevice as sd
import time
from scipy.io.wavfile import write

from model import ECommerceCommandModel

# --- Configuration ---
# These should match the settings in train.py
MODEL_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'e_commerce_model')
MODEL_SAVE_PATH = os.path.join(MODEL_OUTPUT_DIR, 'best_model.pth')
LABEL_MAP_PATH = os.path.join(MODEL_OUTPUT_DIR, 'label_map.json')

# Audio processing settings from training
TARGET_SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 5
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
RECORDING_CHANNELS = 1

# --- Helper Functions ---

def load_model_and_labels():
    """Loads the trained model and label mappings."""
    print("--- Loading Model and Labels ---")

    # Check if model and label map exist
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Model file not found at {MODEL_SAVE_PATH}")
        print("Please train the model first by running the 'train.py' script.")
        return None, None, None

    if not os.path.exists(LABEL_MAP_PATH):
        print(f"Error: Label map file not found at {LABEL_MAP_PATH}")
        return None, None, None

    # Load label map
    with open(LABEL_MAP_PATH, 'r') as f:
        label_to_int = json.load(f)
    int_to_label = {i: label for label, i in label_to_int.items()}
    num_classes = len(label_to_int)
    print(f"Loaded {num_classes} command labels.")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = ECommerceCommandModel(n_input_mels=N_MELS, n_output_classes=num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()  # Set model to evaluation mode
    print("Model loaded successfully.")

    return model, int_to_label, device

def record_audio(duration, sample_rate):
    """Records audio from the default microphone."""
    print(f"\nGet ready to speak. Recording for {duration} seconds...")
    sd.play(np.sin(2 * np.pi * 440 * np.arange(sample_rate * 0.2) / sample_rate), samplerate=sample_rate, blocking=True) # Start beep

    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=RECORDING_CHANNELS, dtype='float32')

    for i in range(duration, 0, -1):
        print(f"Recording... {i}", end='\r')
        time.sleep(1)

    sd.wait()  # Wait until recording is finished
    sd.play(np.sin(2 * np.pi * 440 * np.arange(sample_rate * 0.2) / sample_rate), samplerate=sample_rate, blocking=True) # End beep
    print("\nRecording finished.")

    return recording.T  # Transpose to get shape (channels, samples)

def preprocess_audio(waveform, sample_rate):
    """
    Preprocesses a raw audio waveform to match the model's input format.
    """
    # Convert to a PyTorch tensor
    waveform = torch.from_numpy(waveform)

    # 1. Resample if necessary (though we record at the target rate)
    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)

    # 2. Pad or truncate to the max length
    max_samples = MAX_AUDIO_SECONDS * TARGET_SAMPLE_RATE
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]
    else:
        padding_needed = max_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding_needed))

    # 3. Generate Mel Spectrogram
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, hop_length=HOP_LENGTH
    )
    spectrogram = mel_spectrogram_transform(waveform)

    # 4. Add a batch dimension for the model
    spectrogram = spectrogram.unsqueeze(0)

    return spectrogram

def predict(model, spectrogram, int_to_label, device):
    """Runs inference on the preprocessed spectrogram."""
    spectrogram = spectrogram.to(device)
    with torch.no_grad():
        outputs = model(spectrogram)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted_index = torch.max(probabilities, 1)

    predicted_label = int_to_label[predicted_index.item()]
    confidence = probabilities[0, predicted_index.item()].item()

    return predicted_label, confidence

# --- Main Application ---

def main():
    """Main function to run the real-time prediction loop."""
    model, int_to_label, device = load_model_and_labels()

    if not model:
        return  # Exit if model loading failed

    while True:
        try:
            input("Press Enter to start recording, or Ctrl+C to exit.")

            # 1. Record audio
            audio_waveform = record_audio(MAX_AUDIO_SECONDS, TARGET_SAMPLE_RATE)

            # 2. Preprocess audio
            print("Processing audio...")
            spectrogram = preprocess_audio(audio_waveform, TARGET_SAMPLE_RATE)

            # 3. Predict
            print("Sending to model for prediction...")
            predicted_command, confidence = predict(model, spectrogram, int_to_label, device)

            # 4. Display result
            print("\n" + "="*30)
            print(f"   Predicted Command: >> {predicted_command} <<")
            print(f"   Confidence: {confidence:.2%}")
            print("="*30 + "\n")

        except KeyboardInterrupt:
            print("\nExiting application. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break

if __name__ == '__main__':
    main()
