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
)
import librosa
import wandb

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

    # Create label mappings
    unique_labels = sorted(df['prompt_text'].unique())
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}

    label_map_path = os.path.join(MODEL_OUTPUT_DIR, 'label_map.json')
    with open(label_map_path, 'w') as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=4)
    logging.info(f"Label map saved to {label_map_path}")

    df["label"] = df["prompt_text"].map(label2id)

    # Analyze class distribution for weighting and augmentation targeting
    class_distribution = analyze_class_distribution(df)
    target_aug_classes = filter_classes_by_frequency(class_distribution, min_samples=8) # Augment classes with less than 8 samples
    logging.info(f"Found {len(target_aug_classes)} classes with < 5 samples for targeted augmentation.")

    # Create Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("label", ClassLabel(num_classes=len(unique_labels), names=unique_labels))

    # IMPORTANT: Cast the audio column *before* splitting and augmentation.
    # This ensures that all splits have a consistent data structure.
    dataset = dataset.cast_column("local_path", Audio(sampling_rate=16000))

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
    """Preprocesses audio data for the Wav2Vec2 model with robust normalization."""
    audio_arrays = [x["array"] for x in examples["local_path"]]

    # Robust audio normalization to prevent numerical instability
    processed_audio = []
    for audio in audio_arrays:
        # Check for non-finite values and replace them
        if not np.isfinite(audio).all():
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
            logging.warning("Found non-finite values in audio, replaced with safe values")

        # Remove DC offset (mean centering)
        audio = audio - np.mean(audio)

        # Apply robust normalization
        # Use RMS normalization for better stability
        rms = np.sqrt(np.mean(audio**2))
        if rms > 1e-8:  # Only normalize if RMS is significant
            audio = audio / (rms + 1e-8) * 0.1  # Scale to reasonable range

        # Final clipping to ensure values are in valid range
        audio = np.clip(audio, -1.0, 1.0)

        processed_audio.append(audio)

    inputs = feature_extractor(
        processed_audio,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration_s),
        truncation=True,
        padding=False, # Collator will handle padding
        return_tensors="np"
    )
    return {"input_values": inputs.input_values, "labels": examples["label"]}

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
    """
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: bool
    max_length: int

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
        batch["labels"] = torch.tensor(label_features)

        return batch







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
            batch_size=4,  # Smaller batch size for preprocessing
            num_proc=1,  # Single process for stability
            remove_columns=[col for col in dataset["train"].column_names if col not in ["input_values", "labels"]],
            desc="Preprocessing audio"
        )
        logging.info(f"Dataset preprocessing complete. Train: {len(encoded_dataset['train'])}")
    except Exception as e:
        logging.error(f"Dataset preprocessing failed: {e}")
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
        per_device_train_batch_size=4,  # Smaller batch size for stability
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # Effective batch size = 16
        eval_strategy="steps",
        eval_steps=50,  # Evaluate more frequently
        save_strategy="steps",
        save_steps=50,
        logging_steps=10,  # More frequent logging
        num_train_epochs=25,  # More epochs for small dataset
        fp16=False,  # Keep fp32 for stability
        bf16=False,
        label_smoothing_factor=0.15,  # Increased label smoothing for small dataset
        gradient_checkpointing=True,
        max_grad_norm=0.5,  # Stricter gradient clipping
        learning_rate=3e-5,  # Slightly lower learning rate
        weight_decay=0.01,
        warmup_ratio=0.15,  # More warmup for stability
        lr_scheduler_type='cosine_with_restarts',
        adam_epsilon=1e-8,
        adam_beta1=0.9,
        adam_beta2=0.999,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="wandb",
        dataloader_drop_last=False,  # Don't drop last batch with small dataset
        dataloader_num_workers=2,
        remove_unused_columns=False,  # Keep all columns for debugging
        push_to_hub=False,
    )

    # 7. Initialize the Trainer with enhanced configuration
    logging.info("Initializing trainer...")
    trainer = Trainer(
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
    if api_key:
        try:
            wandb.login(key=api_key)
            logging.info("Successfully logged into Weights & Biases")
        except Exception as e:
            logging.warning(f"Wandb login failed: {e}. Disabling wandb logging.")
            os.environ["WANDB_DISABLED"] = "true"
    else:
        logging.warning("WANDB_API_KEY not found. Wandb logging will be disabled.")
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
