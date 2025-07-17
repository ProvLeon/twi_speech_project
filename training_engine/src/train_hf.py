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
from transformers import (
    AutoConfig,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
import librosa
import wandb
import warnings
warnings.filterwarnings("ignore")

# --- Basic Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Hugging Face Model and Training Configuration ---
MODEL_CHECKPOINT = "facebook/wav2vec2-base-960h"
MODEL_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'e_commerce_model_hf_optimized')
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_cache_optimized"

# Set environment variables to suppress CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# --- Audio Augmentation Functions ---

def apply_noise(audio, noise_factor=0.003):
    """Adds random noise to the audio signal."""
    if len(audio) == 0:
        return audio
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return augmented_audio

def apply_pitch_shift(audio, sampling_rate, n_steps):
    """Shifts the pitch of the audio."""
    if len(audio) == 0:
        return audio
    try:
        return librosa.effects.pitch_shift(y=audio, sr=sampling_rate, n_steps=n_steps)
    except Exception as e:
        logging.warning(f"Pitch shift failed: {e}")
        return audio

def apply_time_stretch(audio, rate):
    """Stretches the time of the audio without changing pitch."""
    if len(audio) == 0:
        return audio
    try:
        return librosa.effects.time_stretch(y=audio, rate=rate)
    except Exception as e:
        logging.warning(f"Time stretch failed: {e}")
        return audio

def apply_volume_change(audio, volume_factor):
    """Changes the volume of the audio."""
    return audio * volume_factor

def analyze_class_distribution(df, column='prompt_text'):
    """Analyzes and prints the distribution of classes."""
    distribution = df[column].value_counts()
    logging.info("Class Distribution:\n" + str(distribution))
    return distribution

def filter_classes_by_frequency(class_distribution, min_samples=3):
    """
    Identifies classes with fewer samples than min_samples.
    These are candidates for augmentation.
    """
    return class_distribution[class_distribution < min_samples].index.tolist()

def augment_audio_data(batch, class_distribution, target_classes, sampling_rate=16000):
    """
    Applies augmentation to a batch of audio data, focusing on underrepresented classes.
    This function is designed to be used with `dataset.map()` and assumes audio is pre-loaded.
    """
    audio_dicts = batch["local_path"]
    labels = batch["label_str"]  # Using string labels to check against target classes

    for i in range(len(audio_dicts)):
        # Augment only if the class is in our target list of underrepresented classes
        if labels[i] in target_classes:
            audio = audio_dicts[i]['array'].copy() # Work on a copy
            sr = audio_dicts[i]['sampling_rate']

            # Apply lighter augmentation to avoid corruption
            choice = np.random.randint(0, 3)  # Reduced from 4 to 3 choices
            if choice == 0:
                audio = apply_noise(audio, noise_factor=np.random.uniform(0.001, 0.003))
            elif choice == 1:
                audio = apply_pitch_shift(audio, sr, n_steps=np.random.uniform(-1, 1))
            elif choice == 2:
                audio = apply_volume_change(audio, volume_factor=np.random.uniform(0.9, 1.1))

            # Update the array in the dictionary
            audio_dicts[i]['array'] = audio

    return batch

def load_and_prepare_dataset(metadata_csv_path: str, augment=False):
    """
    Loads, prepares, and optionally augments the dataset.
    """
    logging.info(f"Loading metadata from {metadata_csv_path}")
    df = pd.read_csv(metadata_csv_path)

    if 'local_path' not in df.columns or 'prompt_text' not in df.columns:
        raise ValueError("Metadata CSV must contain 'local_path' and 'prompt_text' columns.")

    # Filter out any rows with missing data
    df = df.dropna(subset=['local_path', 'prompt_text'])

    # Verify audio files exist
    existing_files = []
    for idx, row in df.iterrows():
        if os.path.exists(row['local_path']):
            existing_files.append(row)
        else:
            logging.warning(f"Audio file not found: {row['local_path']}")

    if len(existing_files) == 0:
        raise ValueError("No valid audio files found!")

    df = pd.DataFrame(existing_files)
    logging.info(f"Found {len(df)} valid audio files")

    # Create label mappings
    unique_labels = sorted(df['prompt_text'].unique())
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}

    logging.info(f"Found {len(unique_labels)} unique classes: {unique_labels}")

    label_map_path = os.path.join(MODEL_OUTPUT_DIR, 'label_map.json')
    with open(label_map_path, 'w') as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=4)
    logging.info(f"Label map saved to {label_map_path}")

    df["label"] = df["prompt_text"].map(label2id)

    # Analyze class distribution for weighting and augmentation targeting
    class_distribution = analyze_class_distribution(df)
    target_aug_classes = filter_classes_by_frequency(class_distribution, min_samples=5)
    logging.info(f"Found {len(target_aug_classes)} classes with < 5 samples for targeted augmentation.")

    # Create Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("label", ClassLabel(num_classes=len(unique_labels), names=unique_labels))

    # Cast the audio column *before* splitting and augmentation
    dataset = dataset.cast_column("local_path", Audio(sampling_rate=16000))

    # Split first, then augment only the training set
    train_test_split = dataset.train_test_split(test_size=0.2, stratify_by_column="label", seed=42)

    if augment and len(target_aug_classes) > 0:
        logging.info("Applying on-the-fly augmentation to the training set...")
        train_set = train_test_split['train'].map(
            lambda example: {'label_str': id2label[example['label']]},
            num_proc=1  # Reduced from multiprocessing to avoid issues
        )

        augmented_train_dataset = train_set.map(
            augment_audio_data,
            fn_kwargs={"class_distribution": class_distribution, "target_classes": target_aug_classes},
            batched=True,
            batch_size=4,  # Reduced batch size
            num_proc=1
        )
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
    """Preprocesses audio data for the Wav2Vec2 model with robust normalization."""
    audio_arrays = [x["array"] for x in examples["local_path"]]

    processed_audio = []
    for audio in audio_arrays:
        # Handle empty or invalid audio
        if len(audio) == 0:
            audio = np.zeros(int(feature_extractor.sampling_rate * 0.1))  # 0.1 second silence

        # Check for non-finite values and replace them
        if not np.isfinite(audio).all():
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # Remove DC offset
        audio = audio - np.mean(audio)

        # Robust normalization with clipping
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / (peak + 1e-8)

        # Clip to prevent extreme values
        audio = np.clip(audio, -1.0, 1.0)

        processed_audio.append(audio)

    inputs = feature_extractor(
        processed_audio,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration_s),
        truncation=True,
        padding=False,
        return_tensors="np"
    )

    return {"input_values": inputs.input_values, "labels": examples["label"]}

def compute_metrics(eval_pred):
    """Computes accuracy and F1 score for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted", zero_division=0)
    return {"accuracy": acc, "f1": f1}

# Custom Data Collator for padding audio sequences
class DataCollatorForWav2Vec2Classification:
    """
    Data collator that dynamically pads the inputs received, as well as the labels.
    """
    def __init__(self, feature_extractor: Wav2Vec2FeatureExtractor, padding: bool = True, max_length: int = None):
        self.feature_extractor = feature_extractor
        self.padding = padding
        self.max_length = max_length

    def __call__(self, features):
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [feature["labels"] for feature in features]

        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(label_features, dtype=torch.long)

        return batch

class LearningRateCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            optimizer = kwargs.get('optimizer')
            if optimizer:
                for i, param_group in enumerate(optimizer.param_groups):
                    wandb.log({f"learning_rate_group_{i}": param_group['lr'], "epoch": state.epoch})

# --- Main Orchestration ---
def run_hf_training(metadata_csv: str, augment_data: bool):
    """
    Orchestrates the entire fine-tuning pipeline.
    """
    logging.info(f"--- Starting Optimized HF Model Fine-Tuning for {MODEL_CHECKPOINT} ---")

    # Initialize wandb with better error handling
    try:
        wandb.init(
            project="twi-speech-e-commerce",
            name=f"optimized-run-{pd.Timestamp.now():%Y%m%d-%H%M}",
            reinit=True
        )
    except Exception as e:
        logging.warning(f"Wandb initialization failed: {e}")
        os.environ["WANDB_DISABLED"] = "true"

    # 1. Load and prepare dataset
    dataset, label2id, id2label = load_and_prepare_dataset(metadata_csv, augment=augment_data)
    logging.info(f"Training set size: {len(dataset['train'])}, Validation set size: {len(dataset['eval'])}")

    # 2. Load Feature Extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_CHECKPOINT)

    # 3. Preprocess the dataset
    encoded_dataset = dataset.map(
        lambda x: preprocess_function(x, feature_extractor),
        batched=True,
        batch_size=4,  # Reduced batch size
        num_proc=1,  # Use single process to avoid multiprocessing issues
        remove_columns=dataset["train"].column_names
    )
    logging.info("Dataset preprocessed for the model.")

    # 4. Load the Model with proper configuration
    config = AutoConfig.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="wav2vec2_clf",
        hidden_dropout=0.1,
        attention_dropout=0.1,
        feat_proj_dropout=0.0,
        layerdrop=0.1,
    )

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        config=config,
        ignore_mismatched_sizes=True
    )

    # Ensure model is in float32 and move to device
    model = model.float()
    if torch.cuda.is_available():
        model = model.cuda()

    logging.info("Model loaded and configured.")

    # Freeze the feature encoder for initial training
    model.freeze_feature_encoder()
    logging.info("Froze feature encoder.")

    # 5. Setup custom data collator
    data_collator = DataCollatorForWav2Vec2Classification(
        feature_extractor=feature_extractor,
        padding=True,
    )

    # 6. Define Training Arguments with conservative settings
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        per_device_train_batch_size=4,  # Reduced batch size
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # Increased to maintain effective batch size
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        num_train_epochs=10,  # Reduced epochs
        fp16=False,  # Keep FP16 disabled
        dataloader_pin_memory=False,
        dataloader_num_workers=0,  # Disable multiprocessing
        label_smoothing_factor=0.1,
        gradient_checkpointing=False,  # Disable to avoid potential issues
        max_grad_norm=0.5,  # Reduced gradient clipping
        learning_rate=1e-5,  # Much lower learning rate
        weight_decay=0.01,
        warmup_ratio=0.2,  # Increased warmup
        lr_scheduler_type='cosine',
        adam_epsilon=1e-8,
        adam_beta1=0.9,
        adam_beta2=0.999,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="wandb" if not os.environ.get("WANDB_DISABLED") else None,
        seed=42,
        push_to_hub=False,
        remove_unused_columns=False,
    )

    # 7. Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["eval"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            LearningRateCallback(),
            EarlyStoppingCallback(early_stopping_patience=3)
        ]
    )

    # 8. Start Training with error handling
    logging.info("--- Starting Training Loop ---")
    try:
        trainer.train()
        logging.info("--- Training Complete ---")
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        raise

    # 9. Final Evaluation and Model Saving
    logging.info("--- Starting Final Evaluation ---")
    try:
        eval_results = trainer.evaluate()
        logging.info(f"Final Evaluation Results: {eval_results}")
        if not os.environ.get("WANDB_DISABLED"):
            wandb.log({"final_eval_results": eval_results})
    except Exception as e:
        logging.error(f"Final evaluation failed: {e}")

    # Save model
    try:
        trainer.save_model(MODEL_OUTPUT_DIR)
        feature_extractor.save_pretrained(MODEL_OUTPUT_DIR)
        logging.info(f"Fine-tuned model and artifacts saved to: {MODEL_OUTPUT_DIR}")
    except Exception as e:
        logging.error(f"Model saving failed: {e}")

    if not os.environ.get("WANDB_DISABLED"):
        wandb.finish()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune a Wav2Vec2 model with optimizations for small datasets.")
    parser.add_argument(
        '--metadata_csv',
        type=str,
        required=True,
        help="Path to the metadata CSV file."
    )
    parser.add_argument(
        '--augment',
        action='store_true',
        help="Enable on-the-fly audio augmentation for the training set."
    )
    args = parser.parse_args()

    # Handle wandb authentication
    if not os.environ.get("WANDB_DISABLED"):
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            try:
                wandb.login(key=api_key)
            except Exception as e:
                logging.warning(f"Wandb login failed: {e}. Disabling wandb.")
                os.environ["WANDB_DISABLED"] = "true"
        else:
            logging.warning("WANDB_API_KEY not found. Wandb logging will be disabled.")
            os.environ["WANDB_DISABLED"] = "true"

    run_hf_training(metadata_csv=args.metadata_csv, augment_data=args.augment)
