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
    DataCollatorWithPadding,
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
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_cache"
# train_csv_path = os.path.abspath(os.path.join(os.path.dirname(metadata_csv_path), 'train.csv'))
# val_csv_path = os.path.abspath(os.path.join(os.path.dirname(metadata_csv_path), 'val.csv'))

# --- Helper Functions ---

def load_and_prepare_dataset(metadata_csv_path: str):
    """
    Loads the dataset from a CSV file using pandas, splits it, and prepares it for Hugging Face.
    This avoids local caching issues in Colab and similar environments.
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

    # Split the data using Hugging Face Dataset's train_test_split for consistency
    from datasets import Dataset, DatasetDict
    dataset = Dataset.from_pandas(df)
    train_test_split = dataset.train_test_split(test_size=0.2, stratify_by_column="label")
    dataset_dict = DatasetDict({
        'train': train_test_split['train'],
        'eval': train_test_split['test']
    })

    # Cast the 'local_path' column to Audio, which automatically loads and resamples
    dataset_dict = dataset_dict.cast_column("local_path", Audio(sampling_rate=16000))
    dataset_dict = dataset_dict.rename_column("prompt_text", "label_str")

    return dataset_dict, label2id, id2label

def preprocess_function(examples, feature_extractor, max_duration_s=5):
    """
    Preprocesses audio data for the Wav2Vec2 model.
    Ensures input_values are 1D arrays and lets the data collator handle padding.
    """
    audio_arrays = [x["array"] for x in examples["local_path"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration_s),
        truncation=True,
        padding=False,  # Let the data collator handle padding!
        # return_attention_mask=True
    )
    # Add labels
    inputs["label"] = examples["label"]
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
    # Remove all non-numeric fields after preprocessing
    keep_cols = ("input_values", "attention_mask", "label")
    encoded_dataset = dataset.map(
        lambda x: preprocess_function(x, feature_extractor),
        remove_columns=[col for col in dataset["train"].column_names if col not in keep_cols],
        batched=True,
        batch_size=8
    )
    logging.info("Dataset preprocessed for the model.")

    # --- Debugging: Check processed data for learning issues ---
    print("Sample processed training example:", encoded_dataset["train"][0])
    print("Sample processed label:", encoded_dataset["train"][0].get("label"))
    unique_labels = set([ex["label"] for ex in encoded_dataset["train"]])
    print("Unique labels in training set:", unique_labels)
    import numpy as np
    input_vals = encoded_dataset["train"][0]["input_values"]
    print("Any NaNs in input_values?", np.isnan(input_vals).any())
    print("All zeros in input_values?", np.all(np.array(input_vals) == 0))
    # Check label distribution
    from collections import Counter
    print("Label distribution in training set:", Counter([ex["label"] for ex in encoded_dataset["train"]]))
    # Check input shapes for first batch
    print("Shape of input_values for first 8 examples:")
    for i in range(8):
        print(np.array(encoded_dataset["train"][i]["input_values"]).shape)

    # --- Setup Data Collator for Audio ---
    data_collator = DataCollatorWithPadding(tokenizer=feature_extractor, padding=True)

    # --- Debug: Print batch shape before training ---
    from torch.utils.data import DataLoader
    dl = DataLoader(encoded_dataset["train"], batch_size=8, collate_fn=data_collator)
    batch = next(iter(dl))
    print("Batch input_values shape:", batch["input_values"].shape)

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
        evaluation_strategy="epoch",
        num_train_epochs=10,
        fp16=True if torch.cuda.is_available() else False,
        learning_rate=1e-4, #3e-5,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        logging_steps=10,
    )
    logging.info("Training arguments configured.")

    data_collator = DataCollatorWithPadding(tokenizer=feature_extractor, padding=True)
    # 6. Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["eval"],
        tokenizer=feature_extractor, # The feature extractor is passed as the tokenizer
        data_collator=data_collator,
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
