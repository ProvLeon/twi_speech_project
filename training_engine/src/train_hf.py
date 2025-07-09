import os
import logging
import json
import torch
import pandas as pd
import numpy as np
from datasets import load_dataset, Audio
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoConfig,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
)
import wandb
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import torchaudio
import random

# Initialize wandb
if not wandb.api.api_key:
    os.environ["WANDB_API_KEY"] = "7037e1e9536dba5af8324bc01133b75b17c9193f"
    wandb.login(key="7037e1e9536dba5af8324bc01133b75b17c9193f")

# --- Basic Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# --- Hugging Face Model and Training Configuration ---
MODEL_CHECKPOINT = "facebook/wav2vec2-base-960h"
MODEL_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'e_commerce_model_hf')

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_cache"

# --- Audio Augmentation Functions ---
def apply_noise(audio, noise_level=0.005):
    """Add Gaussian noise to audio"""
    noise = np.random.normal(0, noise_level, audio.shape)
    return audio + noise

def apply_pitch_shift(audio, sample_rate, max_shift=2):
    """Shift pitch of audio"""
    shift = random.uniform(-max_shift, max_shift)
    return torchaudio.functional.pitch_shift(
        torch.tensor(audio).unsqueeze(0),
        sample_rate,
        shift
    ).squeeze(0).numpy()

def apply_time_stretch(audio, min_factor=0.8, max_factor=1.2):
    """Time stretch audio"""
    factor = random.uniform(min_factor, max_factor)
    return torchaudio.functional.resample(
        torch.tensor(audio).unsqueeze(0),
        16000,
        int(16000 * factor)
    ).squeeze(0).numpy()

def apply_volume_change(audio, min_gain=0.7, max_gain=1.3):
    """Change volume of audio"""
    gain = random.uniform(min_gain, max_gain)
    return audio * gain

# --- Helper Functions ---

def analyze_class_distribution(df):
    """Analyze and visualize class distribution"""
    class_counts = df['prompt_text'].value_counts()

    logging.info(f"Total classes: {len(class_counts)}")
    logging.info(f"Total samples: {len(df)}")
    logging.info(f"Average samples per class: {len(df) / len(class_counts):.2f}")
    logging.info(f"Min samples per class: {class_counts.min()}")
    logging.info(f"Max samples per class: {class_counts.max()}")

    # Classes with very few samples
    rare_classes = class_counts[class_counts < 3]
    logging.info(f"Classes with < 3 samples: {len(rare_classes)}")

    # Classes with only 1 sample
    singleton_classes = class_counts[class_counts == 1]
    logging.info(f"Classes with only 1 sample: {len(singleton_classes)}")

    return class_counts

def filter_classes_by_frequency(df, min_samples=3):
    """Filter out classes with too few samples"""
    class_counts = df['prompt_text'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index

    filtered_df = df[df['prompt_text'].isin(valid_classes)].copy()

    logging.info(f"Filtered from {len(df)} to {len(filtered_df)} samples")
    logging.info(f"Filtered from {df['prompt_text'].nunique()} to {filtered_df['prompt_text'].nunique()} classes")

    return filtered_df

def augment_audio_data(df, target_samples_per_class=10):
    """
    Augment audio data using various techniques for underrepresented classes
    """
    augmented_rows = []

    for class_label in df['prompt_text'].unique():
        class_samples = df[df['prompt_text'] == class_label]
        current_count = len(class_samples)
        augmented_rows.extend(class_samples.to_dict('records'))

        # If we need more samples, create augmented versions
        if current_count < target_samples_per_class:
            needed = target_samples_per_class - current_count
            augmentation_functions = [
                apply_noise,
                lambda x: apply_pitch_shift(x, 16000),
                apply_time_stretch,
                apply_volume_change
            ]

            # Create multiple augmented versions
            for i in range(needed):
                sample = class_samples.iloc[i % current_count].to_dict()

                # Apply 1-3 random augmentations
                num_augmentations = random.randint(1, 3)
                audio_path = sample['local_path']

                try:
                    # Load audio
                    audio, sr = torchaudio.load(audio_path)
                    audio = audio.numpy().squeeze()

                    # Apply augmentations
                    for _ in range(num_augmentations):
                        aug_func = random.choice(augmentation_functions)
                        audio = aug_func(audio)

                    # Save augmented audio
                    aug_path = audio_path.replace(".wav", f"_aug{i}.wav")
                    torchaudio.save(aug_path, torch.tensor(audio).unsqueeze(0), 16000)
                    sample['local_path'] = aug_path
                    augmented_rows.append(sample)
                except Exception as e:
                    logging.warning(f"Audio augmentation failed: {e}")
                    # Fallback to duplication
                    augmented_rows.append(sample)

    augmented_df = pd.DataFrame(augmented_rows)
    logging.info(f"Augmented dataset from {len(df)} to {len(augmented_df)} samples")
    return augmented_df

def load_and_prepare_dataset(metadata_csv_path: str, min_samples_per_class=3, target_samples_per_class=8):
    """
    Loads the dataset from a CSV file, filters classes, and prepares it for training.
    """
    logging.info(f"Loading metadata from {metadata_csv_path}")
    df = pd.read_csv(metadata_csv_path)

    # Ensure required columns exist
    if 'local_path' not in df.columns or 'prompt_text' not in df.columns:
        raise ValueError("Metadata CSV must contain 'local_path' and 'prompt_text' columns.")

    # Analyze original distribution
    logging.info("=== Original Dataset Analysis ===")
    analyze_class_distribution(df)

    # Filter classes with too few samples
    logging.info("=== Filtering Classes ===")
    df = filter_classes_by_frequency(df, min_samples=min_samples_per_class)

    # Augment data to balance classes
    logging.info("=== Augmenting Data ===")
    df = augment_audio_data(df, target_samples_per_class=target_samples_per_class)

    # Final analysis
    logging.info("=== Final Dataset Analysis ===")
    analyze_class_distribution(df)

    # Create label mappings
    unique_labels = sorted(df['prompt_text'].unique())
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}

    # Save the label map
    label_map_path = os.path.join(MODEL_OUTPUT_DIR, 'label_map.json')
    with open(label_map_path, 'w') as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=4)
    logging.info(f"Label map saved to {label_map_path}")

    # Map string labels to integer IDs in the DataFrame
    df["label"] = df["prompt_text"].map(label2id)

    # Split the data using stratified sampling
    train_df, eval_df = train_test_split(
        df,
        test_size=0.15,  # Smaller test size for small dataset
        stratify=df['label'],
        random_state=42
    )

    # Create datasets from the split DataFrames
    from datasets import Dataset, DatasetDict
    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(train_df.reset_index(drop=True)),
        'eval': Dataset.from_pandas(eval_df.reset_index(drop=True))
    })

    # Cast the 'local_path' column to Audio
    dataset_dict = dataset_dict.cast_column("local_path", Audio(sampling_rate=16000))
    dataset_dict = dataset_dict.rename_column("prompt_text", "label_str")

    # Compute class weights for handling imbalance
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_df['label']),
        y=train_df['label']
    )
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

    return dataset_dict, label2id, id2label, class_weights_dict

def preprocess_function(examples, feature_extractor, max_duration_s=5):
    """
    Preprocesses audio data for the Wav2Vec2 model with better normalization.
    """
    audio_arrays = [x["array"] for x in examples["local_path"]]

    # Improved audio normalization
    normalized_audio = []
    for audio in audio_arrays:
        if len(audio) > 0:
            # Remove DC offset
            audio = audio - np.mean(audio)

            # Normalize to [-1, 1] range with small epsilon to prevent division by zero
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio_norm = audio / max_val
            else:
                audio_norm = audio

            # Apply gentle clipping
            audio_norm = np.clip(audio_norm, -0.95, 0.95)
            normalized_audio.append(audio_norm)
        else:
            # Handle empty audio with small noise instead of silence
            normalized_audio.append(np.random.normal(0, 0.001, 1600))

    # Process audio with feature extractor
    inputs = feature_extractor(
        normalized_audio,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration_s),
        truncation=True,
        padding=True,
        return_tensors="np"
    )

    return {
        "input_values": inputs.input_values,
        "labels": examples["label"]
    }

def compute_metrics(eval_pred):
    """
    Computes comprehensive metrics for evaluation.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    f1_macro = f1_score(labels, predictions, average="macro")

    return {
        "accuracy": acc,
        "f1_weighted": f1,
        "f1_macro": f1_macro
    }

# Custom Data Collator
class DataCollatorForWav2Vec2Classification:
    """
    Data collator that handles padding.
    """
    def __init__(self, feature_extractor, padding=True, max_length=None):
        self.feature_extractor = feature_extractor
        self.padding = padding
        self.max_length = max_length

    def __call__(self, features):
        # Separate input_values and labels
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        labels = [feature["labels"] for feature in features]

        # Pad input_values
        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Add labels to batch
        batch["labels"] = torch.tensor(labels, dtype=torch.long)

        return batch

# Custom Trainer with class weighting and learning rate scheduling
class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.lr_scheduler = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            # Create class weights tensor
            weight_tensor = torch.tensor(
                [self.class_weights[i] for i in range(len(self.class_weights))],
                dtype=torch.float,
                device=logits.device
            )

            # Compute weighted loss
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight_tensor)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        # Create different learning rates for different parts of the model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                          if "wav2vec2" in n and not any(nd in n for nd in no_decay)],
                "lr": self.args.learning_rate / 10,
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                          if "wav2vec2" in n and any(nd in n for nd in no_decay)],
                "lr": self.args.learning_rate / 10,
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                          if "classifier" in n or "projector" in n],
                "lr": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon,
        )

        return optimizer

# Learning rate scheduler callback
class LearningRateCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch > 0:
            current_lr = state.log_history[-1].get('learning_rate', 0)
            logging.info(f"End of epoch {state.epoch}: Learning rate = {current_lr:.2e}")

# --- Main Orchestration ---
def run_hf_training(metadata_csv: str):
    """
    Orchestrates the entire fine-tuning pipeline with improvements.
    """
    logging.info(f"--- Starting Improved Hugging Face Model Fine-Tuning ---")

    # 1. Load and prepare dataset with filtering and augmentation
    dataset, label2id, id2label, class_weights = load_and_prepare_dataset(
        metadata_csv,
        min_samples_per_class=3,
        target_samples_per_class=15  # Increased for small dataset
    )

    logging.info(f"Final dataset - Training: {len(dataset['train'])}, Validation: {len(dataset['eval'])}")
    logging.info(f"Number of classes: {len(label2id)}")

    # 2. Load Feature Extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_CHECKPOINT)
    logging.info("Loaded Wav2Vec2FeatureExtractor.")

    # 3. Preprocess the dataset
    encoded_dataset = dataset.map(
        lambda x: preprocess_function(x, feature_extractor),
        remove_columns=[col for col in dataset["train"].column_names if col not in ["input_values", "labels"]],
        batched=True,
        batch_size=16
    )
    logging.info("Dataset preprocessed for the model.")

    # 4. Load the Model with optimized configuration
    config = AutoConfig.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="wav2vec2_clf",
        layerdrop=0.05,  # Reduced dropout
        mask_time_prob=0.03,  # Reduced masking
        mask_feature_prob=0.03,
        hidden_dropout=0.05,  # Reduced dropout
        attention_dropout=0.05,  # Reduced dropout
        classifier_dropout=0.1,
    )

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        config=config,
        ignore_mismatched_sizes=True
    )

    # Move model to device
    model.to(device)
    logging.info(f"Model moved to device: {device}")

    # Freeze only the first 10 layers of the feature encoder
    for i in range(10):
        model.wav2vec2.encoder.layers[i].requires_grad_(False)
    logging.info("First 10 layers of feature encoder frozen")

    # Initialize classification layers properly
    def init_classification_layers(model):
        if hasattr(model, 'classifier'):
            torch.nn.init.kaiming_normal_(model.classifier.weight, nonlinearity='relu')
            torch.nn.init.constant_(model.classifier.bias, 0.0)

        if hasattr(model, 'projector') and model.projector is not None:
            torch.nn.init.kaiming_normal_(model.projector.weight, nonlinearity='relu')
            torch.nn.init.constant_(model.projector.bias, 0.0)

    init_classification_layers(model)

    # 5. Setup data collator
    data_collator = DataCollatorForWav2Vec2Classification(
        feature_extractor=feature_extractor,
        padding=True,
        max_length=80000,  # 5 seconds at 16kHz
    )

    # 6. Define optimized training arguments
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        per_device_train_batch_size=4,  # Smaller batch size for small dataset
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,  # More accumulation steps
        eval_strategy="epoch",  # Evaluate every epoch for small dataset
        save_strategy="epoch",
        logging_steps=10,
        num_train_epochs=30,  # More epochs for small dataset
        fp16=torch.cuda.is_available(),
        learning_rate=3e-4,  # Higher learning rate
        weight_decay=0.001,  # Lower weight decay
        warmup_ratio=0.1,  # Warmup ratio instead of steps
        max_grad_norm=1.0,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        report_to="wandb",
        adam_epsilon=1e-8,
        optim="adamw_torch",
        lr_scheduler_type="cosine",  # Simpler scheduler
        dataloader_num_workers=2,
        seed=42,
        run_name="small-dataset-wav2vec2",
        skip_memory_metrics=True,
        dataloader_prefetch_factor=2,
        gradient_checkpointing=True,  # Enable to save memory
        evaluation_strategy="epoch",
        logging_strategy="steps",
    )

    # 7. Initialize the custom trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["eval"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[
            LearningRateCallback(),
            EarlyStoppingCallback(early_stopping_patience=10, early_stopping_threshold=0.001)  # More patience
        ],
    )

    logging.info("Optimized trainer initialized.")

    # 8. Test small batch
    logging.info("--- Testing small batch ---")
    try:
        small_batch = [encoded_dataset["train"][i] for i in range(2)]
        collated_batch = data_collator(small_batch)

        # Move to device manually for test
        collated_batch = {k: v.to(device) for k, v in collated_batch.items()}

        # Test forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(**collated_batch)
            logging.info(f"Test batch - Loss: {outputs.loss:.4f}, Logits shape: {outputs.logits.shape}")
    except Exception as e:
        logging.error(f"Error in small batch test: {e}")
        raise

    # 9. Start Training
    logging.info("--- Starting Optimized Training Loop ---")
    try:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.save_metrics("train", metrics)
        logging.info("--- Training Complete ---")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise

    # 10. Final evaluation with detailed metrics
    logging.info("--- Final Evaluation ---")
    eval_results = trainer.evaluate()
    logging.info(f"Final Results: {eval_results}")

    # Generate classification report
    predictions = trainer.predict(encoded_dataset["eval"])
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids

    # Create classification report
    target_names = [id2label[i] for i in range(len(id2label))]
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)

    # Save detailed results
    results_path = os.path.join(MODEL_OUTPUT_DIR, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'final_metrics': eval_results,
            'classification_report': report,
            'class_weights': class_weights
        }, f, indent=4)

    # 11. Save model and artifacts
    model.save_pretrained(MODEL_OUTPUT_DIR)
    feature_extractor.save_pretrained(MODEL_OUTPUT_DIR)
    logging.info(f"Model and artifacts saved to: {MODEL_OUTPUT_DIR}")
    logging.info("Optimized model training completed successfully!")

if __name__ == '__main__':
    import argparse
    import torch.multiprocessing as mp

    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="Fine-tune Wav2Vec2 with improvements.")
    parser.add_argument(
        '--metadata_csv',
        type=str,
        required=True,
        help="Path to the metadata CSV file."
    )
    args = parser.parse_args()

    run_hf_training(metadata_csv=args.metadata_csv)
