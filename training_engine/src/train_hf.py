import os
import logging
import json
import torch
import pandas as pd
import numpy as np
from datasets import load_dataset, Audio
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoConfig,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
import wandb

# Initialize wandb if not already logged in
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

# --- Hugging Face Model and Training Configuration ---
MODEL_CHECKPOINT = "facebook/wav2vec2-base-960h"
MODEL_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'e_commerce_model_hf')

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_cache"

# --- Helper Functions ---

def load_and_prepare_dataset(metadata_csv_path: str):
    """
    Loads the dataset from a CSV file using pandas, splits it, and prepares it for Hugging Face.
    """
    logging.info(f"Loading metadata from {metadata_csv_path}")
    df = pd.read_csv(metadata_csv_path)

    # Ensure required columns exist
    if 'local_path' not in df.columns or 'prompt_text' not in df.columns:
        raise ValueError("Metadata CSV must contain 'local_path' and 'prompt_text' columns.")

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

    # Split the data using sklearn first, then create HF datasets
    train_df, eval_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['label'],
        random_state=42
    )

    # Create datasets from the split DataFrames
    from datasets import Dataset, DatasetDict
    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(train_df.reset_index(drop=True)),
        'eval': Dataset.from_pandas(eval_df.reset_index(drop=True))
    })

    # Cast the 'local_path' column to Audio, which automatically loads and resamples
    dataset_dict = dataset_dict.cast_column("local_path", Audio(sampling_rate=16000))
    dataset_dict = dataset_dict.rename_column("prompt_text", "label_str")

    return dataset_dict, label2id, id2label

def preprocess_function(examples, feature_extractor, max_duration_s=5):
    """
    Preprocesses audio data for the Wav2Vec2 model.
    """
    audio_arrays = [x["array"] for x in examples["local_path"]]

    # Normalize audio arrays to prevent numerical instability
    normalized_audio = []
    for audio in audio_arrays:
        if len(audio) > 0:
            # Normalize audio to [-1, 1] range
            audio_norm = audio / (np.max(np.abs(audio)) + 1e-8)
            # Apply additional scaling to prevent extreme values
            audio_norm = np.clip(audio_norm, -0.9, 0.9)
            normalized_audio.append(audio_norm)
        else:
            # Handle empty audio
            normalized_audio.append(np.zeros(1600))  # 0.1 seconds of silence

    # Process audio with proper parameters
    inputs = feature_extractor(
        normalized_audio,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration_s),
        truncation=True,
        padding=False,
        return_tensors="np"
    )

    result = {
        "input_values": inputs.input_values,
        "labels": examples["label"]
    }

    return result

def compute_metrics(eval_pred):
    """
    Computes accuracy and F1 score for evaluation.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

# Custom Data Collator for Wav2Vec2
class DataCollatorForWav2Vec2Classification:
    """
    Data collator that dynamically pads the inputs received.
    """
    def __init__(self, feature_extractor, padding=True, max_length=None, pad_to_multiple_of=None):
        self.feature_extractor = feature_extractor
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        # Separate input_values and labels
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        labels = [feature["labels"] for feature in features]

        # Pad input_values
        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )

        # Add labels to batch
        batch["labels"] = torch.tensor(labels, dtype=torch.long)

        return batch

# Custom callback to handle NaN gradients properly
class NaNGradientCallback(TrainerCallback):
    """
    Callback to monitor and handle NaN gradients during training.
    """
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of each training step."""
        if model is not None:
            # Check for NaN gradients and handle them
            nan_found = False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        logging.warning(f"NaN/Inf gradient detected in {name}, zeroing gradients")
                        param.grad = torch.zeros_like(param.grad)
                        nan_found = True

            if nan_found:
                logging.warning("NaN gradients detected and handled")
                # Optionally stop training if NaN persists
                control.should_training_stop = True

# --- Main Orchestration ---
def run_hf_training(metadata_csv: str):
    """
    Orchestrates the entire fine-tuning pipeline for a Hugging Face model.
    """
    logging.info(f"--- Starting Hugging Face Model Fine-Tuning for {MODEL_CHECKPOINT} ---")

    # 1. Load and prepare dataset
    dataset, label2id, id2label = load_and_prepare_dataset(metadata_csv)
    logging.info(f"Training set size: {len(dataset['train'])}, Validation set size: {len(dataset['eval'])}")

    # 2. Load Feature Extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_CHECKPOINT)
    logging.info("Loaded Wav2Vec2FeatureExtractor.")

    # 3. Preprocess the dataset
    encoded_dataset = dataset.map(
        lambda x: preprocess_function(x, feature_extractor),
        remove_columns=[col for col in dataset["train"].column_names if col not in ["input_values", "labels"]],
        batched=True,
        batch_size=8
    )
    logging.info("Dataset preprocessed for the model.")

    # --- Debugging: Check processed data ---
    print("Sample processed training example keys:", encoded_dataset["train"][0].keys())
    print("Sample processed label:", encoded_dataset["train"][0].get("labels"))
    unique_labels = set([ex["labels"] for ex in encoded_dataset["train"]])
    print("Unique labels in training set:", len(unique_labels))

    # Check for data issues
    input_vals = encoded_dataset["train"][0]["input_values"]
    print("Input values shape:", np.array(input_vals).shape)
    print("Input values mean:", np.mean(input_vals))
    print("Input values std:", np.std(input_vals))
    print("Any NaNs in input_values?", np.isnan(input_vals).any())
    print("Any Infs in input_values?", np.isinf(input_vals).any())

    # 4. Load the Model with better initialization
    config = AutoConfig.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="wav2vec2_clf",
        layerdrop=0.0,  # Disable layer dropout during fine-tuning
        mask_time_prob=0.0,  # Disable masking during fine-tuning
        mask_feature_prob=0.0,
    )

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        config=config,
        ignore_mismatched_sizes=True
    )

    # Freeze the feature extractor to prevent gradient explosion
    model.wav2vec2.feature_extractor._freeze_parameters()

    # Improved initialization of classifier layers
    def init_weights(module):
        if isinstance(module, torch.nn.Linear):
            # Use Xavier uniform initialization for linear layers
            torch.nn.init.xavier_uniform_(module.weight, gain=0.02)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0.0)
            torch.nn.init.constant_(module.weight, 1.0)

    # Apply initialization to classifier and projector
    if hasattr(model, 'classifier'):
        init_weights(model.classifier)
    if hasattr(model, 'projector') and model.projector is not None:
        init_weights(model.projector)

    logging.info("Loaded Wav2Vec2ForSequenceClassification model with properly initialized classification head.")

    # 5. Setup custom data collator
    data_collator = DataCollatorForWav2Vec2Classification(
        feature_extractor=feature_extractor,
        padding=True,
        max_length=80000,  # 5 seconds at 16kHz
    )

    # 6. Define Training Arguments with stable hyperparameters
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        per_device_train_batch_size=1,  # Very small batch size
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,  # Larger accumulation for effective batch size
        eval_strategy="steps",
        eval_steps=200,
        save_steps=400,
        logging_steps=20,
        num_train_epochs=3,
        fp16=False,  # Disable mixed precision to avoid NaN
        learning_rate=1e-5,  # Very conservative learning rate
        weight_decay=0.01,
        warmup_steps=200,
        max_grad_norm=0.1,  # Very strict gradient clipping
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="wandb",
        adam_epsilon=1e-8,
        optim="adamw_torch",
        lr_scheduler_type="cosine",  # Cosine scheduler for stability
        dataloader_num_workers=0,
        seed=42,
        data_seed=42,
        run_name="wav2vec2-audio-classification",
        # Additional stability settings
        skip_memory_metrics=True,
        dataloader_persistent_workers=False,
    )
    logging.info("Training arguments configured.")

    # 7. Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["eval"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[NaNGradientCallback()],  # Add our custom callback
    )

    logging.info("Trainer initialized.")

    # 8. Test a small batch before full training
    logging.info("--- Testing small batch ---")
    try:
        small_batch = [encoded_dataset["train"][i] for i in range(2)]
        collated_batch = data_collator(small_batch)
        print("Collated batch keys:", list(collated_batch.keys()))
        print("Input values shape:", collated_batch["input_values"].shape)
        print("Labels shape:", collated_batch["labels"].shape)
        print("Labels:", collated_batch["labels"])

        # Test forward pass
        model.train()
        with torch.no_grad():  # Test without gradients first
            outputs = model(**collated_batch)
            print("Model output logits shape:", outputs.logits.shape)
            print("Model loss:", outputs.loss.item() if outputs.loss is not None else "No loss")
            print("Logits range:", outputs.logits.min().item(), "to", outputs.logits.max().item())
            print("Any NaNs in logits?", torch.isnan(outputs.logits).any().item())

        # Test backward pass
        model.zero_grad()
        outputs = model(**collated_batch)
        if outputs.loss is not None and not torch.isnan(outputs.loss):
            outputs.loss.backward()

            # Check gradients
            total_norm = 0
            param_count = 0
            nan_count = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
                    if torch.isnan(param.grad).any():
                        nan_count += 1

            total_norm = total_norm ** (1. / 2)
            print(f"Total gradient norm: {total_norm}")
            print(f"Parameters with gradients: {param_count}")
            print(f"Parameters with NaN gradients: {nan_count}")

            if nan_count > 0:
                logging.error("NaN gradients detected in test batch!")
                return

            model.zero_grad()
        else:
            logging.error("Loss is NaN in test batch!")
            return

    except Exception as e:
        logging.error(f"Error in small batch test: {e}")
        raise

    # 9. Start Training
    logging.info("--- Starting Training Loop ---")
    try:
        trainer.train()
        logging.info("--- Training Complete ---")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise

    # 10. Final Evaluation and Model Saving
    logging.info("--- Starting Final Evaluation ---")
    eval_results = trainer.evaluate()
    logging.info(f"Final Evaluation Results: {eval_results}")

    # Save the final model and artifacts
    model.save_pretrained(MODEL_OUTPUT_DIR)
    feature_extractor.save_pretrained(MODEL_OUTPUT_DIR)
    logging.info(f"Fine-tuned model and artifacts saved to: {MODEL_OUTPUT_DIR}")
    logging.info("Model training completed successfully!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune a Hugging Face Wav2Vec2 model.")
    parser.add_argument(
        '--metadata_csv',
        type=str,
        required=True,
        help="Path to the metadata CSV file containing 'local_path' and 'prompt_text' columns."
    )
    args = parser.parse_args()

    run_hf_training(metadata_csv=args.metadata_csv)
