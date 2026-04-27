from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report, precision_recall_fscore_support
import numpy as np
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the saved model and tokenizer
model = BertForTokenClassification.from_pretrained("./bert_ner_model").to(device)
tokenizer = BertTokenizerFast.from_pretrained("./bert_ner_model")

# Align predictions with labels, filtering out special tokens
def align_predictions(predictions, label_ids):
    preds = np.argmax(predictions, axis=2)  # Take the highest scoring class per token

    aligned_preds, aligned_labels = [], []
    for pred, label in zip(preds, label_ids):
        # Filter out special tokens ([PAD], [CLS], [SEP])
        valid_preds = [p for p, l in zip(pred, label) if l != -100]
        valid_labels = [l for l in label if l != -100]

        aligned_preds.extend(valid_preds)
        aligned_labels.extend(valid_labels)

    return aligned_preds, aligned_labels

# Custom function to compute evaluation metrics
def compute_metrics(eval_pred):
    predictions, label_ids = eval_pred
    preds, labels = align_predictions(predictions, label_ids)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    report = classification_report(labels, preds, target_names=['O', 'CHEMICAL', 'GENE-Y', 'GENE-N'])

    print("\nClassification Report:\n", report)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# Load and flatten the data
def load_data(file_path):
    all_tokens, all_labels = [], []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # If line is not empty
                token, label = line.strip().split()
                all_tokens.append(token)
                all_labels.append(model.config.label2id[label])

    return all_tokens, all_labels

# Load dataset
tokens, labels = load_data('/home/akshatkrishna/chemprot-relexner-pipeline-main/ner_files/bio_ner_testing_data.txt')
# Reload your validation dataset (if necessary)
# This assumes the val_tokens and val_labels have been prepared earlier in the same way as during training
# Create train-test split with flattened lists
train_tokens, val_tokens, train_labels, val_labels = train_test_split(
    tokens, labels, test_size=0.5, random_state=42
)
# import ipdb; ipdb.set_trace()
val_dataset = Dataset.from_dict({"tokens": val_tokens, "labels": val_labels})

# Tokenization and alignment function with label padding
def tokenize_and_align_labels(tokens, labels):
    # Tokenize the input tokens
    tokenized_inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        padding='max_length',  # Pad to max length (512 by default)
        truncation=True,
        max_length=512,
        return_tensors="pt"  # Return PyTorch tensors
    )

    # Align labels to match the input length
    aligned_labels = [-100] * tokenized_inputs.input_ids.shape[1]  # Initialize with -100 (ignored during training)
    
    # Fill in the valid labels
    valid_token_count = min(len(labels), tokenized_inputs.input_ids.shape[1])
    aligned_labels[:valid_token_count] = labels[:valid_token_count]

    # Add aligned labels to tokenized input
    tokenized_inputs["labels"] = torch.tensor(aligned_labels).unsqueeze(0)  # Add batch dimension

    return tokenized_inputs

# Tokenize and align datasets
train_encodings = tokenize_and_align_labels(train_tokens, train_labels)
val_encodings = tokenize_and_align_labels(val_tokens, val_labels)

# Convert to Hugging Face datasets
train_dataset = Dataset.from_dict(train_encodings)
val_dataset = Dataset.from_dict(val_encodings)

# Set training arguments for evaluation
training_args = TrainingArguments(
    output_dir="./ner_model",
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,
)

# Initialize the Trainer with compute_metrics for evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Evaluate the model on the validation dataset
print("Evaluating the loaded model...")
metrics = trainer.evaluate()

print("\nEvaluation Metrics:", metrics)