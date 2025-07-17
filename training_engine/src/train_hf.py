import os
import logging
import json
import torch
import pandas as pd
import numpy as np
from datasets import load_dataset, Audio, Dataset, DatasetDict, ClassLabel, Features, Value
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
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

            choice = np.random.randint(0, 4)
            if choice == 0:
                audio = apply_noise(audio, noise_factor=np.random.uniform(0.003, 0.007))
            elif choice == 1:
                audio = apply_pitch_shift(audio, sr, n_steps=np.random.uniform(-2, 2))
            elif choice == 2:
                # Avoid extreme stretching which can corrupt audio
                audio = apply_time_stretch(audio, rate=np.random.uniform(0.9, 1.1))
            elif choice == 3:
                audio = apply_volume_change(audio, volume_factor=np.random.uniform(0.8, 1.2))

            # Update the array in the dictionary
            audio_dicts[i]['array'] = audio

    # The whole batch dict is returned by map, with the 'local_path' column modified.
    return batch

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
    target_aug_classes = filter_classes_by_frequency(class_distribution, min_samples=5) # Augment classes with less than 5 samples
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

    # Compute class weights for handling imbalance in the loss function
    y_train = dataset_dict["train"]["label"]
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    return dataset_dict, label2id, id2label, class_weights

def preprocess_function(examples, feature_extractor, max_duration_s=5.0):
    """Preprocesses audio data for the Wav2Vec2 model with normalization."""
    audio_arrays = [x["array"] for x in examples["local_path"]]

    # Normalize audio to [-1, 1] range to prevent numerical instability
    processed_audio = []
    for audio in audio_arrays:
        # Check for non-finite values and replace them
        if not np.isfinite(audio).all():
            audio = np.nan_to_num(audio) # Replaces NaN with 0 and Inf with large numbers

        # Remove DC offset
        audio = audio - np.mean(audio)

        # Normalize with a small epsilon to prevent division by zero
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / (peak + 1e-9)

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


def init_weights(m):
    """Applies Kaiming He initialization to linear layers."""
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


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


class CustomTrainer(Trainer):
    """Custom Trainer to implement a weighted loss function."""
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Overrides the default loss computation."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Add a check for NaN logits to catch instability early
        if torch.isnan(logits).any():
            logging.error("NaN detected in model logits. Training cannot continue.")
            raise RuntimeError("NaN logits detected. Check data and model stability.")

        # Use weighted CrossEntropyLoss
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """
        Creates an optimizer with layer-wise learning rates (discriminative fine-tuning).
        The classification head gets a higher learning rate than the pre-trained base model.
        """
        model = self.model
        lr_head = self.args.learning_rate * 10  # Higher LR for the new layers
        lr_base = self.args.learning_rate       # Lower LR for the pre-trained layers

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if "wav2vec2" in n],
                "lr": lr_base,
            },
            {
                "params": [p for n, p in model.named_parameters() if "wav2vec2" not in n],
                "lr": lr_head,
            },
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate, # This is a default, will be overridden by group LRs
            eps=self.args.adam_epsilon
        )
        logging.info(f"Created AdamW optimizer with differential learning rates: Base LR={lr_base}, Head LR={lr_head}")
        return self.optimizer

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
    Orchestrates the entire fine-tuning pipeline.
    """
    logging.info(f"--- Starting Optimized HF Model Fine-Tuning for {MODEL_CHECKPOINT} ---")
    wandb.init(project="twi-speech-e-commerce", name=f"optimized-run-{pd.Timestamp.now():%Y%m%d-%H%M}")

    # 1. Load and prepare dataset, getting class weights
    dataset, label2id, id2label, class_weights = load_and_prepare_dataset(metadata_csv, augment=augment_data)
    logging.info(f"Training set size: {len(dataset['train'])}, Validation set size: {len(dataset['eval'])}")
    logging.info(f"Class weights computed for {len(class_weights)} classes.")

    # 2. Load Feature Extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_CHECKPOINT)

    # 3. Preprocess the dataset
    # After augmentation, the 'local_path' column now contains the dictionary with the audio array
    encoded_dataset = dataset.map(
        lambda x: preprocess_function(x, feature_extractor),
        batched=True,
        batch_size=8,
        num_proc=os.cpu_count() // 2 or 1,
        remove_columns=dataset["train"].column_names # Keep only input_values and labels
    )
    logging.info("Dataset preprocessed for the model.")

    # 4. Load the Model with a new head
    config = AutoConfig.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="wav2vec2_clf",
    )
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, config=config, ignore_mismatched_sizes=True)

    # Apply Kaiming He initialization to the randomly initialized classification layers
    model.projector.apply(init_weights)
    model.classifier.apply(init_weights)
    logging.info("Applied Kaiming He initialization to the classification head.")

    # Freeze the feature encoder (CNNs) and all but the last 2 transformer layers
    model.freeze_feature_encoder()
    for layer in model.wav2vec2.encoder.layers[:-2]:
        for param in layer.parameters():
            param.requires_grad = False
    logging.info("Froze feature encoder and all but the last 2 transformer layers.")

    # 5. Setup custom data collator
    data_collator = DataCollatorForWav2Vec2Classification(
        feature_extractor=feature_extractor,
        padding=True,
    )

    # 6. Define Training Arguments
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2, # Effective batch size = 16
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        num_train_epochs=15, # More epochs because of augmentation
        fp16=False, # Disabled fp16 to enforce numerical stability
        gradient_checkpointing=True, # Saves memory, can also help stability
        max_grad_norm=1.0,
        learning_rate=3e-5, # Reduced learning rate for stability
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        adam_epsilon=1e-6, # Stabilize optimizer
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="wandb",
    )

    # 6. Initialize the Custom Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["eval"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[LearningRateCallback()]
    )

    # 8. Start Training
    logging.info("--- Starting Training Loop ---")
    trainer.train()
    logging.info("--- Training Complete ---")

    # 8. Final Evaluation and Model Saving
    logging.info("--- Starting Final Evaluation ---")
    eval_results = trainer.evaluate()
    logging.info(f"Final Evaluation Results: {eval_results}")
    wandb.log({"final_eval_results": eval_results})

    trainer.save_model(MODEL_OUTPUT_DIR)
    feature_extractor.save_pretrained(MODEL_OUTPUT_DIR)
    logging.info(f"Fine-tuned model and artifacts saved to: {MODEL_OUTPUT_DIR}")
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

    # It's good practice to log in to wandb at the start
    if not wandb.api.api_key:
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)
        else:
            logging.warning("WANDB_API_KEY not found. Wandb logging will be disabled.")
            os.environ["WANDB_DISABLED"] = "true"

    run_hf_training(metadata_csv=args.metadata_csv, augment_data=args.augment)
