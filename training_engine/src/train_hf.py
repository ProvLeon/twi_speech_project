import os
import logging
import json
import torch
import pandas as pd
import numpy as np
from datasets import load_dataset, Audio, Dataset, DatasetDict, ClassLabel, Features, Value
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
# Replaced broad import (some symbols not exported at top-level in certain transformers versions)
from transformers import (
    AutoConfig,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.data.data_collator import DataCollatorWithPadding
import librosa
import wandb

# --- Basic Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
    logging.info(f"Loaded environment variables from {env_path}")
else:
    logging.warning("No .env file found, using system environment variables")

# --- Hugging Face Model and Training Configuration ---
MODEL_CHECKPOINT = "facebook/wav2vec2-base-960h"
MODEL_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'e_commerce_model_hf_optimized')
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_cache_optimized"

# --- Audio Augmentation Functions ---

def apply_noise(audio, noise_factor=0.005):
    """Adds random noise to the audio signal."""
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return augmented_audio

def apply_pitch_shift(audio, sampling_rate, n_steps):
    """Shifts the pitch of the audio."""
    return librosa.effects.pitch_shift(y=audio, sr=sampling_rate, n_steps=n_steps)

def apply_time_stretch(audio, rate):
    """Stretches the time of the audio without changing pitch."""
    return librosa.effects.time_stretch(y=audio, rate=rate)

def apply_volume_change(audio, volume_factor):
    """Changes the volume of the audio."""
    return audio * volume_factor

def analyze_class_distribution(df, column='prompt_text'):
    """Analyzes and prints the distribution of classes."""
    distribution = df[column].value_counts()
    logging.info("Class Distribution:\n" + str(distribution))
    return distribution

def filter_classes_by_frequency(class_distribution, min_samples=5):
    """
    Identifies classes with fewer samples than min_samples.
    These are candidates for targeted augmentation.
    """
    underrepresented = class_distribution[class_distribution < min_samples].index.tolist()
    logging.info(f"Classes needing augmentation ({len(underrepresented)}): {underrepresented[:5]}...")
    return underrepresented

def augment_audio_data(batch, class_distribution, target_classes, sampling_rate=16000):
    """
    Applies targeted augmentation to underrepresented classes with multiple variations.
    Creates 2-3 augmented versions for very underrepresented classes.
    """
    audio_dicts = batch["local_path"]
    labels = batch["label_str"]

    augmented_audio = []
    augmented_labels = []

    for i in range(len(audio_dicts)):
        # Always keep the original
        augmented_audio.append(audio_dicts[i])
        augmented_labels.append(labels[i])

        # Apply multiple augmentations for underrepresented classes
        if labels[i] in target_classes:
            original_audio = audio_dicts[i]['array'].copy()
            sr = audio_dicts[i]['sampling_rate']

            # Determine number of augmentations based on class frequency
            class_count = class_distribution[labels[i]]
            num_augmentations = min(3, max(1, 5 - class_count))  # 1-3 augmentations

            for aug_idx in range(num_augmentations):
                # Apply different combinations of augmentations
                augmented = original_audio.copy()

                # Primary augmentation
                aug_type = np.random.randint(0, 4)
                if aug_type == 0:
                    augmented = apply_noise(augmented, noise_factor=np.random.uniform(0.005, 0.015))
                elif aug_type == 1:
                    augmented = apply_pitch_shift(augmented, sr, n_steps=np.random.uniform(-3, 3))
                elif aug_type == 2:
                    augmented = apply_time_stretch(augmented, rate=np.random.uniform(0.85, 1.15))
                elif aug_type == 3:
                    augmented = apply_volume_change(augmented, volume_factor=np.random.uniform(0.7, 1.3))

                # Sometimes apply a second mild augmentation
                if np.random.random() < 0.3:  # 30% chance of double augmentation
                    second_aug = np.random.randint(0, 2)
                    if second_aug == 0:
                        augmented = apply_noise(augmented, noise_factor=np.random.uniform(0.001, 0.005))
                    else:
                        augmented = apply_volume_change(augmented, volume_factor=np.random.uniform(0.9, 1.1))

                # Ensure audio is still valid
                augmented = np.clip(augmented, -1.0, 1.0)

                # Create new audio dict
                aug_dict = audio_dicts[i].copy()
                aug_dict['array'] = augmented
                augmented_audio.append(aug_dict)
                augmented_labels.append(labels[i])

    # Return batch with augmented data
    return {
        "local_path": augmented_audio,
        "label_str": augmented_labels,
        "label": [class_distribution.index[class_distribution.index == label].tolist()[0] if hasattr(class_distribution, 'index') else label for label in augmented_labels]
    }

def load_and_prepare_dataset(metadata_csv_path: str, augment=False):
    """
    Loads, prepares, and optionally augments the dataset.
    """
    logging.info(f"Loading metadata from {metadata_csv_path}")
    df = pd.read_csv(metadata_csv_path)

    if 'local_path' not in df.columns or 'prompt_text' not in df.columns:
        raise ValueError("Metadata CSV must contain 'local_path' and 'prompt_text' columns.")

    # Fix relative paths in the dataframe
    def fix_path(path):
        if not os.path.isabs(path):
            if path.startswith('training_engine/'):
                # Get the project root (parent of parent of src directory)
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                # Remove 'training_engine/' prefix and join with project root
                relative_path = path[len('training_engine/'):]
                return os.path.join(project_root, 'training_engine', relative_path)
            else:
                # If path doesn't start with training_engine/, assume it's relative to project root
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                return os.path.join(project_root, path)
        return path

    df['local_path'] = df['local_path'].apply(fix_path)

    # Verify files exist and log details
    missing_files = []
    valid_files = []
    for idx, row in df.iterrows():
        file_path = row['local_path']
        if os.path.exists(file_path):
            # Check if it's a valid audio file by extension
            if file_path.lower().endswith(('.wav', '.m4a', '.mp3', '.flac')):
                valid_files.append(file_path)
            else:
                logging.warning(f"File has unsupported audio extension: {file_path}")
                missing_files.append(file_path)
        else:
            logging.warning(f"File does not exist: {file_path}")
            missing_files.append(file_path)

    if missing_files:
        logging.warning(f"Found {len(missing_files)} missing or invalid files. Removing from dataset.")
        logging.info(f"Sample missing files: {missing_files[:5]}")
        df = df[df['local_path'].apply(lambda x: os.path.exists(x) and x.lower().endswith(('.wav', '.m4a', '.mp3', '.flac')))]
        logging.info(f"Dataset reduced to {len(df)} samples after removing missing files.")

    if len(df) == 0:
        raise ValueError("No valid audio files found in the dataset after path validation!")

    # Create label mappings
    unique_labels = sorted(df['prompt_text'].unique())
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}

    logging.info(f"Found {len(unique_labels)} unique labels: {list(unique_labels)[:10]}...")

    label_map_path = os.path.join(MODEL_OUTPUT_DIR, 'label_map.json')
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    with open(label_map_path, 'w') as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=4)
    logging.info(f"Label map saved to {label_map_path}")

    df["label"] = df["prompt_text"].map(label2id)

    # Verify all labels were mapped correctly
    if df["label"].isna().any():
        logging.error("Some labels could not be mapped!")
        logging.error(f"Unmapped labels: {df[df['label'].isna()]['prompt_text'].unique()}")
        raise ValueError("Label mapping failed!")

    # Analyze class distribution for weighting and augmentation targeting
    class_distribution = analyze_class_distribution(df)
    target_aug_classes = filter_classes_by_frequency(class_distribution, min_samples=8) # Augment classes with less than 8 samples
    logging.info(f"Found {len(target_aug_classes)} classes with < 5 samples for targeted augmentation.")

    # Create Hugging Face Dataset
    logging.info(f"Creating HuggingFace dataset from {len(df)} samples")
    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("label", ClassLabel(num_classes=len(unique_labels), names=unique_labels))

    # IMPORTANT: Cast the audio column *before* splitting and augmentation.
    # This ensures that all splits have a consistent data structure.
    logging.info("Casting audio column to Audio feature...")
    try:
        dataset = dataset.cast_column("local_path", Audio(sampling_rate=16000))
        logging.info("Audio column cast successfully")

        # Test load a sample to ensure audio loading works
        sample = dataset[0]
        if "local_path" in sample and "array" in sample["local_path"]:
            logging.info(f"Sample audio shape: {sample['local_path']['array'].shape}")
        else:
            logging.error("Audio casting may have failed - sample doesn't have expected structure")

    except Exception as e:
        logging.error(f"Failed to cast audio column: {e}")
        logging.error("This usually means audio files are corrupted or in unsupported format")
        raise

    # Split first, then augment only the training set
    train_test_split = dataset.train_test_split(test_size=0.2, stratify_by_column="label", seed=42)

    if augment:
        logging.info("Applying on-the-fly augmentation to the training set...")
        # We need the string label for our augmentation logic, so add it temporarily
        train_set = train_test_split['train'].map(
            lambda example: {'label_str': id2label[example['label']]},
            num_proc=os.cpu_count() // 2 or 1
        )

        augmented_train_dataset = train_set.map(
            augment_audio_data,
            fn_kwargs={"class_distribution": class_distribution, "target_classes": target_aug_classes},
            batched=True,
            batch_size=10, # Process in small batches
            num_proc=os.cpu_count() // 2 or 1
        )
        # Remove the temporary string label column
        augmented_train_dataset = augmented_train_dataset.remove_columns(['label_str'])
        logging.info("Augmentation mapping complete.")

        dataset_dict = DatasetDict({
            'train': augmented_train_dataset,
            'eval': train_test_split['test']
        })
    else:
        dataset_dict = DatasetDict({
            'train': train_test_split['train'],
            'eval': train_test_split['test']
        })

    # Rename original text label column
    dataset_dict = dataset_dict.rename_column("prompt_text", "label_str")

    return dataset_dict, label2id, id2label

def preprocess_function(examples, feature_extractor, max_duration_s=5.0):
    """Preprocesses audio data for the Wav2Vec2 model with robust normalization and debugging."""
    # Debug: Check the actual structure of examples
    logging.info(f"Processing batch with keys: {list(examples.keys())}")

    # Extract audio arrays from HuggingFace Audio format
    # When using Audio feature, the audio data is in examples["local_path"]["array"]
    try:
        if isinstance(examples["local_path"], list):
            # Batched format - each item is a dict with "array" and "sampling_rate" keys
            audio_arrays = []
            valid_indices = []
            for i, audio_item in enumerate(examples["local_path"]):
                if audio_item is not None and isinstance(audio_item, dict) and "array" in audio_item:
                    audio_arrays.append(audio_item["array"])
                    valid_indices.append(i)
                else:
                    logging.warning(f"Invalid audio item at index {i}: {type(audio_item)}")

            if not audio_arrays:
                logging.error("No valid audio arrays found in batch")
                return {"input_values": [], "labels": []}

            # Filter labels to match valid audio arrays
            valid_labels = [examples["label"][i] for i in valid_indices]
        else:
            # Single example format
            if examples["local_path"] is None or not isinstance(examples["local_path"], dict) or "array" not in examples["local_path"]:
                logging.error(f"Invalid single audio item: {type(examples['local_path'])}")
                return {"input_values": [], "labels": []}
            audio_arrays = [examples["local_path"]["array"]]
            valid_labels = [examples["label"]]
    except (KeyError, TypeError) as e:
        logging.error(f"Error extracting audio arrays: {e}")
        logging.error(f"local_path structure: {type(examples['local_path'])}")
        if isinstance(examples["local_path"], list) and len(examples["local_path"]) > 0:
            logging.error(f"First item structure: {type(examples['local_path'][0])}")
            if isinstance(examples["local_path"][0], dict):
                logging.error(f"First item keys: {list(examples['local_path'][0].keys())}")
        return {"input_values": [], "labels": []}

    # Debug: Check input data
    logging.info(f"Processing batch of {len(audio_arrays)} audio files")

    # Robust audio normalization to prevent numerical instability
    processed_audio = []
    final_labels = []

    for i, (audio, label) in enumerate(zip(audio_arrays, valid_labels)):
        try:
            # Check for completely empty or invalid audio
            if audio is None or len(audio) == 0:
                logging.warning(f"Empty or None audio file at index {i}, skipping")
                continue

            # Convert to numpy array and ensure float32 dtype with proper handling
            try:
                if not isinstance(audio, np.ndarray):
                    audio = np.array(audio, dtype=np.float32)
                else:
                    # Ensure proper dtype conversion without issues
                    audio = np.asarray(audio, dtype=np.float32)
            except (ValueError, TypeError) as e:
                logging.error(f"Failed to convert audio to numpy array at index {i}: {e}")
                continue

            # Check for non-finite values and replace them with safer numpy operations
            finite_mask = np.isfinite(audio)
            if not np.all(finite_mask):
                logging.warning(f"Found non-finite values in audio at index {i}, replacing...")
                # Replace non-finite values more carefully
                audio = np.where(finite_mask, audio, 0.0)

            # Check if audio is all zeros
            if np.all(audio == 0):
                logging.warning(f"Audio file at index {i} is all zeros, adding small noise")
                audio = np.random.normal(0, 1e-6, audio.shape).astype(np.float32)

            # Remove DC offset (mean centering)
            audio_mean = np.mean(audio)
            if np.isfinite(audio_mean):
                audio = audio - audio_mean

            # Apply robust normalization using peak normalization for Wav2Vec2
            max_val = np.max(np.abs(audio))
            if max_val > 1e-10 and np.isfinite(max_val):  # Only normalize if there's significant signal
                audio = audio / max_val * 0.05  # Scale to 5% of max range for Wav2Vec2 stability
            else:
                # If signal is too small, replace with small noise
                audio = np.random.normal(0, 1e-6, audio.shape).astype(np.float32)

            # Final clipping to very conservative range for Wav2Vec2
            audio = np.clip(audio, -0.1, 0.1)

            # Final safety check for finite values
            if not np.all(np.isfinite(audio)):
                logging.error(f"Audio still contains non-finite values after processing at index {i}, zeroing")
                audio = np.zeros_like(audio, dtype=np.float32)

            processed_audio.append(audio)
            final_labels.append(label)

        except Exception as e:
            logging.error(f"Error processing audio at index {i}: {e}")
            # Skip this sample but don't fail the entire batch
            continue

    if len(processed_audio) == 0:
        logging.error("No valid audio samples in batch after processing!")
        return {"input_values": [], "labels": []}

    # Debug: Log statistics
    audio_stats = [f"min={np.min(a):.6f}, max={np.max(a):.6f}, mean={np.mean(a):.6f}" for a in processed_audio[:3]]
    logging.info(f"Audio stats (first 3): {audio_stats}")

    try:
        # Process audio through feature extractor with proper numpy handling
        inputs = feature_extractor(
            processed_audio,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=int(feature_extractor.sampling_rate * max_duration_s),
            truncation=True,
            padding=False,  # Collator will handle padding
            return_tensors="np"
        )

        # Normalize and validate feature extractor output (can be list, ndarray, or object-dtype ndarray)
        input_values = inputs.input_values

        # Convert to list of 1D float32 arrays (collator will handle padding later)
        if isinstance(input_values, np.ndarray):
            if input_values.dtype == object:
                input_list = [np.asarray(x, dtype=np.float32) for x in input_values]
            else:
                # Numeric ndarray (batch, time) -> split rows
                input_list = [np.asarray(x, dtype=np.float32) for x in input_values]
        elif isinstance(input_values, list):
            input_list = [np.asarray(x, dtype=np.float32) for x in input_values]
        else:
            logging.error(f"Unexpected input_values type from feature_extractor: {type(input_values)}")
            return {"input_values": [], "labels": []}

        cleaned_values = []
        for idx, arr in enumerate(input_list):
            # Ensure ndarray float32
            if not isinstance(arr, np.ndarray):
                try:
                    arr = np.asarray(arr, dtype=np.float32)
                except Exception as conv_err:
                    logging.warning(f"Could not convert feature array {idx}: {conv_err}")
                    continue
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32, copy=False)

            if arr.size == 0:
                logging.warning(f"Empty feature array at index {idx}, skipping")
                continue

            finite_mask = np.isfinite(arr)
            if not np.all(finite_mask):
                logging.warning(f"Non-finite values in extracted features sample {idx}, fixing...")
                arr = np.where(finite_mask, arr, 0.0).astype(np.float32)

            scale = np.max(np.abs(arr)) if arr.size else 0.0
            if np.isfinite(scale) and scale > 1.0:
                # Rescale overly large amplitudes conservatively
                arr = arr / scale * 0.1

            cleaned_values.append(arr)

        if len(cleaned_values) == 0:
            logging.error("All feature extractor outputs were invalid after cleaning")
            return {"input_values": [], "labels": []}

        # Debug stats (preview concatenated subset)
        try:
            concat_preview = np.concatenate([cv[:1000] for cv in cleaned_values if cv.size > 0])
            logging.info(
                f"Feature extractor output - batch={len(cleaned_values)}, "
                f"example0_shape={cleaned_values[0].shape}, "
                f"range=[{concat_preview.min():.6f}, {concat_preview.max():.6f}]"
            )
        except Exception as stats_err:
            logging.debug(f"Could not compute preview stats: {stats_err}")

        inputs.input_values = cleaned_values

        return {"input_values": inputs.input_values, "labels": final_labels}

    except Exception as e:
        logging.error(f"Feature extractor failed: {e}")
        import traceback
        logging.error(f"Full traceback: {traceback.format_exc()}")
        return {"input_values": [], "labels": []}

def compute_metrics(eval_pred):
    """Computes accuracy and F1 score for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}





# Custom Data Collator for padding audio sequences
class DataCollatorForWav2Vec2Classification:
    """
    Data collator that dynamically pads the inputs received, as well as the labels.
    Enhanced with debugging and validation.
    """
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: bool
    max_length: int

    def __init__(self, feature_extractor: Wav2Vec2FeatureExtractor, padding: bool = True, max_length: int = None):
        self.feature_extractor = feature_extractor
        self.padding = padding
        self.max_length = max_length

    def __call__(self, features):
        # Debug: Log batch info
        logging.debug(f"Collating batch of {len(features)} samples")

        # Filter out any invalid features
        valid_features = []
        valid_labels = []

        for i, feature in enumerate(features):
            try:
                if "input_values" not in feature or "labels" not in feature:
                    logging.warning(f"Feature {i} missing required keys")
                    continue

                # Extract raw values
                input_vals = feature["input_values"]
                label = feature["labels"]

                # Handle cases where preprocess_function produced a list or nested list
                # (e.g., accidental wrapping or object-dtype arrays)
                if isinstance(input_vals, list):
                    # If it's a list containing a single array/list, unwrap
                    if len(input_vals) == 1 and isinstance(input_vals[0], (list, np.ndarray)):
                        input_vals = input_vals[0]
                    elif len(input_vals) > 1 and all(isinstance(x, (int, float, np.floating, np.integer)) for x in input_vals):
                        # Plain list of numbers
                        input_vals = np.asarray(input_vals, dtype=np.float32)
                    else:
                        # Try flattening nested structure
                        try:
                            flat = []
                            for x in input_vals:
                                if isinstance(x, (list, np.ndarray)):
                                    flat.extend(np.asarray(x).ravel().tolist())
                                else:
                                    flat.append(x)
                            input_vals = np.asarray(flat, dtype=np.float32)
                        except Exception as nest_err:
                            logging.warning(f"Feature {i} could not be flattened: {nest_err}")
                            continue

                # If it's a numpy array with object dtype, attempt to concatenate elements
                if isinstance(input_vals, np.ndarray) and input_vals.dtype == object:
                    try:
                        input_vals = np.concatenate(
                            [np.asarray(x, dtype=np.float32).ravel() for x in input_vals if x is not None]
                        ).astype(np.float32)
                    except Exception as obj_err:
                        logging.warning(f"Feature {i} object-dtype concat failed: {obj_err}")
                        continue

                # Final conversion to float32 numpy array
                if not isinstance(input_vals, np.ndarray):
                    try:
                        input_vals = np.asarray(input_vals, dtype=np.float32)
                    except Exception as arr_err:
                        logging.warning(f"Feature {i} could not be converted to ndarray: {arr_err}")
                        continue

                if input_vals.ndim > 1:
                    # Flatten multi-dim (should be 1D waveform)
                    input_vals = input_vals.reshape(-1).astype(np.float32)

                # Validate length
                if input_vals.size == 0:
                    logging.warning(f"Feature {i} has empty input_values after processing")
                    continue

                # Replace non-finite values
                finite_mask = np.isfinite(input_vals)
                if not np.all(finite_mask):
                    logging.warning(f"Feature {i} has non-finite input_values (fixing)")
                    input_vals = np.where(finite_mask, input_vals, 0.0).astype(np.float32)

                # Optional amplitude normalization (keep consistent with preprocessing)
                peak = np.max(np.abs(input_vals)) if input_vals.size else 0.0
                if np.isfinite(peak) and peak > 1.0:
                    input_vals = (input_vals / peak * 0.1).astype(np.float32)

                # Validate label
                if not isinstance(label, (int, np.integer)):
                    logging.warning(f"Feature {i} has invalid label type: {type(label)}")
                    continue

                if label < 0:
                    logging.warning(f"Feature {i} has negative label: {label}")
                    continue

                valid_features.append({"input_values": input_vals})
                valid_labels.append(int(label))  # Ensure int type

            except Exception as e:
                logging.error(f"Error validating feature {i}: {e}")
                continue

        if len(valid_features) == 0:
            logging.error("No valid features in batch!")
            # Return empty batch
            return {
                "input_values": torch.empty((0, 1000)),  # Dummy shape
                "labels": torch.empty((0,), dtype=torch.long)
            }

        try:
            batch = self.feature_extractor.pad(
                valid_features,
                padding=self.padding,
                max_length=self.max_length,
                return_tensors="pt",
            )

            # Validate padded input_values
            if not torch.isfinite(batch["input_values"]).all():
                logging.error("Padded input_values contain NaN/Inf!")
                batch["input_values"] = torch.nan_to_num(batch["input_values"], nan=0.0, posinf=1.0, neginf=-1.0)

            batch["labels"] = torch.tensor(valid_labels, dtype=torch.long)

            # Debug: Log batch statistics
            logging.debug(f"Batch - input_values shape: {batch['input_values'].shape}, "
                         f"labels shape: {batch['labels'].shape}, "
                         f"labels range: [{batch['labels'].min()}, {batch['labels'].max()}]")

            return batch

        except Exception as e:
            logging.error(f"Error in data collator: {e}")
            # Return minimal valid batch
            return {
                "input_values": torch.zeros((len(valid_features), 1000)),
                "labels": torch.tensor(valid_labels, dtype=torch.long)
            }







class LearningRateCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        # Log learning rate at the end of each epoch
        if state.is_world_process_zero:
            optimizer = kwargs['optimizer']
            for i, param_group in enumerate(optimizer.param_groups):
                wandb.log({f"learning_rate_group_{i}": param_group['lr'], "epoch": state.epoch})

# --- Main Orchestration ---
def run_hf_training(metadata_csv: str, augment_data: bool):
    """
    Orchestrates the entire fine-tuning pipeline with enhanced stability and performance.
    """
    logging.info(f"--- Starting Enhanced HF Model Fine-Tuning for {MODEL_CHECKPOINT} ---")

    # Initialize wandb with better configuration
    run_name = f"twi-speech-enhanced-{pd.Timestamp.now():%Y%m%d-%H%M%S}"
    wandb.init(
        project="twi-speech-e-commerce",
        name=run_name,
        config={
            "model_checkpoint": MODEL_CHECKPOINT,
            "augment_data": augment_data,
            "dataset_path": metadata_csv
        }
    )

    # 1. Load and prepare dataset with enhanced logging
    logging.info("Loading and preparing dataset...")
    dataset, label2id, id2label = load_and_prepare_dataset(metadata_csv, augment=augment_data)

    train_size = len(dataset['train'])
    eval_size = len(dataset['eval'])
    logging.info(f"Dataset prepared - Training: {train_size}, Validation: {eval_size}")
    logging.info(f"Number of classes: {len(label2id)}")
    logging.info(f"Classes: {list(label2id.keys())[:10]}...")  # Show first 10 classes

    # Log dataset statistics to wandb
    wandb.log({
        "dataset/train_size": train_size,
        "dataset/eval_size": eval_size,
        "dataset/num_classes": len(label2id)
    })

    # 2. Load Feature Extractor with validation
    logging.info("Loading feature extractor...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_CHECKPOINT)
    logging.info(f"Feature extractor sampling rate: {feature_extractor.sampling_rate}")

    # 3. Preprocess the dataset with error handling
    logging.info("Preprocessing dataset...")
    try:
        encoded_dataset = dataset.map(
            lambda x: preprocess_function(x, feature_extractor),
            batched=True,
            batch_size=8,  # Increase batch size slightly
            num_proc=1,  # Single process for stability
            remove_columns=[col for col in dataset["train"].column_names if col not in ["input_values", "labels"]],
            desc="Preprocessing audio",
            load_from_cache_file=False  # Don't use cache to ensure fresh processing
        )
        logging.info(f"Dataset preprocessing complete. Train: {len(encoded_dataset['train'])}, Eval: {len(encoded_dataset['eval'])}")

        # Check if datasets are empty after preprocessing
        if len(encoded_dataset['train']) == 0:
            raise ValueError("Training dataset is empty after preprocessing! Check audio file paths and formats.")
        if len(encoded_dataset['eval']) == 0:
            raise ValueError("Evaluation dataset is empty after preprocessing! Check audio file paths and formats.")

    except Exception as e:
        logging.error(f"Dataset preprocessing failed: {e}")
        import traceback
        logging.error(f"Full traceback: {traceback.format_exc()}")
        raise

    # 4. Load and configure the model
    logging.info("Loading and configuring model...")
    config = AutoConfig.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="wav2vec2_clf",
        classifier_dropout=0.1,
        final_dropout=0.1,
    )

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        config=config,
        ignore_mismatched_sizes=True
    )
    model = model.to(torch.float32)

    # Fix NaN/Inf in masked_spec_embed parameter (known Wav2Vec2 issue)
    if hasattr(model.wav2vec2, 'masked_spec_embed'):
        with torch.no_grad():
            if not torch.isfinite(model.wav2vec2.masked_spec_embed).all():
                logging.warning("Fixing NaN/Inf values in masked_spec_embed parameter")
                model.wav2vec2.masked_spec_embed.fill_(0.0)

    logging.info("Model loaded and configured successfully.")

    # Progressive unfreezing strategy
    model.freeze_feature_encoder()
    logging.info("Feature encoder frozen for initial training.")

    # 5. Perform stability checks
    logging.info("Performing model stability checks...")
    model.eval()
    try:
        # Test with actual feature extractor output dimensions
        dummy_length = feature_extractor.sampling_rate * 2  # 2 seconds of audio
        dummy_input_values = torch.randn(2, dummy_length, dtype=torch.float32)
        dummy_labels = torch.randint(0, len(label2id), (2,), dtype=torch.long)

        with torch.no_grad():
            outputs = model(input_values=dummy_input_values, labels=dummy_labels)
            logging.info(f"Stability check passed. Loss: {outputs.loss:.4f}")

    except Exception as e:
        logging.error(f"Model stability check failed: {e}")
        raise

    model.train()

    # 6. Setup data collator
    data_collator = DataCollatorForWav2Vec2Classification(
        feature_extractor=feature_extractor,
        padding=True,
    )

    # 6. Define Optimized Training Arguments
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        per_device_train_batch_size=1,  # Smallest possible batch for debugging
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,  # Maintain effective batch size = 16
        evaluation_strategy="steps",
        eval_steps=25,  # More frequent evaluation for debugging
        save_strategy="steps",
        save_steps=25,
        logging_steps=5,  # Very frequent logging for debugging
        num_train_epochs=3,  # Fewer epochs for debugging
        fp16=False,  # Keep fp32 for maximum stability
        bf16=False,
        label_smoothing_factor=0.0,  # Disable label smoothing for debugging
        gradient_checkpointing=False,  # Disable for debugging
        max_grad_norm=1.0,  # Less aggressive gradient clipping
        learning_rate=5e-6,  # Even lower learning rate for stability
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type='linear',  # Simple scheduler
        adam_epsilon=1e-8,
        adam_beta1=0.9,
        adam_beta2=0.999,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",  # Use accuracy instead of f1 for debugging
        greater_is_better=True,
        report_to="wandb",
        dataloader_drop_last=True,  # Drop last batch to ensure consistent batch sizes
        dataloader_num_workers=0,  # Single-threaded for debugging
        remove_unused_columns=False,
        push_to_hub=False,
        logging_first_step=True,
        logging_nan_inf_filter=False,  # Don't filter NaN/Inf logs - we want to see them
    )

    # Debug: Check dataset before training and apply final input scaling
    logging.info("Checking and post-processing dataset samples before training...")
    try:
        # Check and fix input scaling across entire dataset
        def scale_input_values(example):
            input_vals = example['input_values']
            # Ensure conservative scaling for Wav2Vec2
            max_val = np.max(np.abs(input_vals))
            if max_val > 2.0:  # If values are too large
                input_vals = input_vals / max_val * 1.0
                logging.debug(f"Scaled input values by factor {max_val:.3f}")
            return {'input_values': input_vals, 'labels': example['labels']}

        # Apply scaling to datasets
        encoded_dataset = encoded_dataset.map(scale_input_values, num_proc=1)

        train_sample = encoded_dataset["train"][0]
        iv = train_sample['input_values']
        if isinstance(iv, np.ndarray):
            logging.info(f"Sample input_values shape: {iv.shape}")
            logging.info(f"Sample input_values range: [{np.min(iv):.6f}, {np.max(iv):.6f}]")
        elif isinstance(iv, list):
            logging.info(f"Sample input_values is list(len={len(iv)}) first_element_type={type(iv[0]) if iv else 'n/a'}")
            if iv and isinstance(iv[0], (list, np.ndarray)):
                flat_preview = np.asarray(iv[0], dtype=np.float32)
                logging.info(f"First element preview len={len(flat_preview)} range=[{flat_preview.min():.6f}, {flat_preview.max():.6f}]")
            elif iv and isinstance(iv[0], (int, float, np.floating)):
                arr_preview = np.asarray(iv[: min(1000, len(iv))], dtype=np.float32)
                logging.info(f"Flat list preview len={arr_preview.shape[0]} range=[{arr_preview.min():.6f}, {arr_preview.max():.6f}]")
        else:
            logging.info(f"Sample input_values type: {type(iv)} (no shape attribute)")
        logging.info(f"Sample label: {train_sample['labels']}")
        logging.info(f"Sample label type: {type(train_sample['labels'])}")
    except Exception as e:
        logging.error(f"Error checking dataset sample: {e}")

    # 7. Initialize the Trainer with enhanced configuration and debugging
    logging.info("Initializing trainer...")

    class DebugTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            """Override compute_loss to add debugging (supports HF's num_items_in_batch kwarg). Ensures a valid loss tensor is always returned."""
            loss = None  # Predefine to avoid UnboundLocalError
            try:
                # --- Input validation / sanitation ---
                if "input_values" in inputs:
                    input_vals = inputs["input_values"]
                    if isinstance(input_vals, list):
                        # Convert list of 1D arrays to padded tensor on-the-fly if Trainer bypassed our collator
                        try:
                            np_arrays = [np.asarray(x, dtype=np.float32) for x in input_vals]
                            max_len = max(a.shape[0] for a in np_arrays)
                            padded = []
                            for a in np_arrays:
                                if a.shape[0] < max_len:
                                    pad = np.zeros(max_len - a.shape[0], dtype=np.float32)
                                    a = np.concatenate([a, pad])
                                padded.append(a)
                            input_vals = torch.tensor(np.stack(padded), dtype=torch.float32, device=next(model.parameters()).device)
                            inputs["input_values"] = input_vals
                        except Exception as conv_err:
                            logging.error(f"Failed to auto-pad list input_values: {conv_err}")
                    elif isinstance(input_vals, np.ndarray):
                        inputs["input_values"] = torch.tensor(input_vals, dtype=torch.float32, device=next(model.parameters()).device)
                    # Final tensor check
                    if isinstance(inputs["input_values"], torch.Tensor):
                        if not torch.isfinite(inputs["input_values"]).all():
                            logging.error("Input values contain NaN/Inf! Repairing...")
                            inputs["input_values"] = torch.nan_to_num(inputs["input_values"], nan=0.0, posinf=1.0, neginf=-1.0)

                if "labels" in inputs:
                    labels = inputs["labels"]
                    if isinstance(labels, list):
                        labels = torch.tensor(labels, dtype=torch.long, device=next(model.parameters()).device)
                        inputs["labels"] = labels
                    elif isinstance(labels, np.ndarray):
                        inputs["labels"] = torch.tensor(labels, dtype=torch.long, device=next(model.parameters()).device)

                    if isinstance(inputs["labels"], torch.Tensor):
                        if not torch.isfinite(inputs["labels"].float()).all():
                            logging.error("Labels contain NaN/Inf!")
                        if inputs["labels"].numel() == 0:
                            logging.error("Labels tensor is empty - substituting dummy label 0")
                            inputs["labels"] = torch.zeros(1, dtype=torch.long, device=next(model.parameters()).device)
                        if inputs["labels"].max() >= model.config.num_labels or inputs["labels"].min() < 0:
                            logging.error(f"Invalid label range: [{inputs['labels'].min()}, {inputs['labels'].max()}] expected [0,{model.config.num_labels-1}]")
                            inputs["labels"] = torch.clamp(inputs["labels"], 0, model.config.num_labels - 1)

                # --- Forward pass ---
                outputs = model(**inputs)

                logits = outputs.logits if hasattr(outputs, "logits") else None

                # --- Derive / validate loss ---
                if hasattr(outputs, "loss") and outputs.loss is not None:
                    loss = outputs.loss
                else:
                    # Manual loss computation if model did not supply it
                    if logits is not None and "labels" in inputs:
                        if not isinstance(inputs["labels"], torch.Tensor):
                            inputs["labels"] = torch.tensor(inputs["labels"], dtype=torch.long, device=logits.device)
                        ce = torch.nn.CrossEntropyLoss()
                        try:
                            loss = ce(logits, inputs["labels"])
                        except Exception as ce_err:
                            logging.error(f"Manual CE loss failed: {ce_err}")
                            loss = torch.tensor(1e-6, device=logits.device if logits is not None else next(model.parameters()).device, requires_grad=True)
                    else:
                        # Fallback dummy loss
                        loss = torch.tensor(1e-6, device=logits.device if logits is not None else next(model.parameters()).device, requires_grad=True)

                    # Attach to outputs for Trainer hooks
                    if hasattr(outputs, "loss"):
                        outputs.loss = loss

                # --- Debug logging ---
                if logits is not None:
                    if not torch.isfinite(logits).all():
                        logging.error("Model outputs contain NaN/Inf after forward!")
                        logging.error(f"Logits range: [{logits.min():.6f}, {logits.max():.6f}]")
                        logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
                    else:
                        logging.debug(f"Logits OK - range: [{logits.min():.6f}, {logits.max():.6f}]")

                if not torch.isfinite(loss):
                    logging.error(f"Loss is NaN/Inf ({loss}); substituting safe loss.")
                    loss = torch.tensor(1e-6, device=loss.device, requires_grad=True)
                    if hasattr(outputs, "loss"):
                        outputs.loss = loss
                else:
                    logging.debug(f"Loss OK: {loss:.6f}")

                return (loss, outputs) if return_outputs else loss

            except Exception as e:
                logging.error(f"Error in compute_loss: {e}", exc_info=True)
                # Safe fallback loss
                dummy_loss = torch.tensor(1e-6, device=next(model.parameters()).device, requires_grad=True)
                if return_outputs:
                    return dummy_loss, {"error": str(e)}
                return dummy_loss

    trainer = DebugTrainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["eval"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[LearningRateCallback()]
    )

    # 8. Start Training with error handling and progressive unfreezing
    logging.info("--- Starting Enhanced Training Loop ---")
    try:
        # Initial training with frozen feature encoder
        logging.info("Phase 1: Training with frozen feature encoder...")
        trainer.train()

        # Save intermediate model
        intermediate_path = os.path.join(MODEL_OUTPUT_DIR, "phase1_model")
        trainer.save_model(intermediate_path)
        logging.info(f"Phase 1 complete. Model saved to {intermediate_path}")

        # Phase 2: Unfreeze and fine-tune with lower learning rate
        if len(encoded_dataset["train"]) > 50:  # Only if we have enough data
            logging.info("Phase 2: Unfreezing feature encoder for fine-tuning...")
            model.unfreeze_feature_encoder()

            # Reduce learning rate for fine-tuning
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] *= 0.1

            # Train for fewer additional epochs
            trainer.args.num_train_epochs = trainer.state.epoch + 5
            trainer.train(resume_from_checkpoint=False)
            logging.info("Phase 2 complete.")
        else:
            logging.info("Skipping Phase 2 - insufficient data for feature encoder unfreezing")

    except Exception as e:
        logging.error(f"Training failed: {e}")
        # Save current state before exiting
        emergency_path = os.path.join(MODEL_OUTPUT_DIR, "emergency_checkpoint")
        try:
            trainer.save_model(emergency_path)
            logging.info(f"Emergency checkpoint saved to {emergency_path}")
        except:
            pass
        raise

    logging.info("--- Training Complete ---")

    # 9. Comprehensive Final Evaluation
    logging.info("--- Starting Final Evaluation ---")
    try:
        eval_results = trainer.evaluate()
        logging.info(f"Final Evaluation Results: {eval_results}")

        # Log detailed metrics
        wandb.log({
            "final_eval/accuracy": eval_results.get("eval_accuracy", 0),
            "final_eval/f1": eval_results.get("eval_f1", 0),
            "final_eval/loss": eval_results.get("eval_loss", 0),
            "training/total_steps": trainer.state.global_step,
            "training/epochs_completed": trainer.state.epoch
        })

    except Exception as e:
        logging.error(f"Final evaluation failed: {e}")
        eval_results = {"error": str(e)}

    # 10. Save Model and Artifacts
    logging.info("--- Saving Model and Artifacts ---")
    try:
        trainer.save_model(MODEL_OUTPUT_DIR)
        feature_extractor.save_pretrained(MODEL_OUTPUT_DIR)

        # Save training configuration
        config_path = os.path.join(MODEL_OUTPUT_DIR, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump({
                "model_checkpoint": MODEL_CHECKPOINT,
                "training_args": training_args.to_dict(),
                "dataset_info": {
                    "train_size": train_size,
                    "eval_size": eval_size,
                    "num_classes": len(label2id)
                },
                "final_results": eval_results
            }, f, indent=2)

        logging.info(f"Model, feature extractor, and config saved to: {MODEL_OUTPUT_DIR}")

    except Exception as e:
        logging.error(f"Model saving failed: {e}")
        raise

    finally:
        wandb.finish()

    return eval_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced Wav2Vec2 fine-tuning for Twi speech commands")
    parser.add_argument(
        '--metadata_csv',
        type=str,
        required=True,
        help="Path to the metadata CSV file containing audio paths and labels."
    )
    parser.add_argument(
        '--augment',
        action='store_true',
        help="Enable targeted audio augmentation for underrepresented classes."
    )
    parser.add_argument(
        '--model_checkpoint',
        type=str,
        default=MODEL_CHECKPOINT,
        help=f"Hugging Face model checkpoint to use. Default: {MODEL_CHECKPOINT}"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=MODEL_OUTPUT_DIR,
        help=f"Directory to save the trained model. Default: {MODEL_OUTPUT_DIR}"
    )
    args = parser.parse_args()

    # Update global variables based on arguments
    if args.model_checkpoint != MODEL_CHECKPOINT:
        globals()['MODEL_CHECKPOINT'] = args.model_checkpoint
    if args.output_dir != MODEL_OUTPUT_DIR:
        globals()['MODEL_OUTPUT_DIR'] = args.output_dir
        os.makedirs(args.output_dir, exist_ok=True)

    # Setup wandb authentication
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key and api_key.strip():  # Check if key exists and is not empty
        try:
            wandb.login(key=api_key)
            logging.info("Successfully logged into Weights & Biases")
        except Exception as e:
            logging.warning(f"Wandb login failed: {e}. Disabling wandb logging.")
            os.environ["WANDB_DISABLED"] = "true"
    else:
        logging.warning("WANDB_API_KEY not found or empty. Wandb logging will be disabled.")
        os.environ["WANDB_DISABLED"] = "true"

    # Validate metadata file exists
    if not os.path.exists(args.metadata_csv):
        logging.error(f"Metadata CSV file not found: {args.metadata_csv}")
        exit(1)

    # Run training
    try:
        results = run_hf_training(metadata_csv=args.metadata_csv, augment_data=args.augment)
        logging.info("Training completed successfully!")
        logging.info(f"Final results: {results}")
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        exit(1)
