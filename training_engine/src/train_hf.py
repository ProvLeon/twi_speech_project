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
)

# --- Basic Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Hugging Face Model and Training Configuration ---
# The pre-trained model we will fine-tune
MODEL_CHECKPOINT = "facebook/wav2vec2-base-960h"

# Directory where the fine-tuned model and artifacts will be saved
MODEL_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'e_commerce_model_hf')
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# --- Helper Functions ---

def load_and_prepare_dataset(metadata_csv_path: str):
    """
    Loads the dataset from a CSV file, splits it, and prepares it for Hugging Face.
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

    # Split the data
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['prompt_text']
    )

    # Save train and val splits BEFORE loading them
    train_csv_path = os.path.join(os.path.dirname(metadata_csv_path), 'train.csv')
    val_csv_path = os.path.join(os.path.dirname(metadata_csv_path), 'val.csv')
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)

    # Now load them
    dataset = load_dataset('csv', data_files={'train': 'train.csv', 'eval': 'val.csv'},
                           data_dir=os.path.dirname(metadata_csv_path))


    # Cast the 'local_path' column to Audio, which automatically loads and resamples
    dataset = dataset.cast_column("local_path", Audio(sampling_rate=16000))
    dataset = dataset.rename_column("prompt_text", "label_str")

    # Map string labels to integer IDs
    def map_label_to_id(example):
        example["label"] = label2id[example["label_str"]]
        return example

    dataset = dataset.map(map_label_to_id)

    return dataset, label2id, id2label

def preprocess_function(examples, feature_extractor, max_duration_s=5):
    """
    Preprocesses audio data for the Wav2Vec2 model.
    """
    audio_arrays = [x["array"] for x in examples["local_path"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration_s),
        truncation=True,
        padding=True
    )
    return inputs

def compute_metrics(eval_pred):
    """
    Computes accuracy and F1 score for evaluation.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}


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
        remove_columns=["local_path", "label_str"],
        batched=True,
        batch_size=8
    )
    logging.info("Dataset preprocessed for the model.")

    # 4. Load the Model
    config = AutoConfig.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="wav2vec2_clf",
    )
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, config=config)
    logging.info("Loaded Wav2Vec2ForSequenceClassification model with a new classification head.")

    # 5. Define Training Arguments
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=10,
        fp16=True if torch.cuda.is_available() else False,
        learning_rate=3e-5,
        warmup_ratio=0.1,
        logging_steps=10,
    )
    logging.info("Training arguments configured.")

    # 6. Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["eval"],
        tokenizer=feature_extractor, # The feature extractor is passed as the tokenizer
        compute_metrics=compute_metrics,
    )
    logging.info("Trainer initialized.")

    # 7. Start Training
    logging.info("--- Starting Training Loop ---")
    trainer.train()
    logging.info("--- Training Complete ---")

    # 8. Manual Evaluation and Model Saving
    logging.info("--- Starting Manual Evaluation ---")
    eval_results = trainer.evaluate()
    logging.info(f"Manual Evaluation Results: {eval_results}")

    # Save the final model and artifacts
    model.save_pretrained(MODEL_OUTPUT_DIR)
    logging.info(f"Fine-tuned model and artifacts saved to: {MODEL_OUTPUT_DIR}")
    logging.info("To use the model, load it from this directory using Wav2Vec2ForSequenceClassification.from_pretrained()")


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
