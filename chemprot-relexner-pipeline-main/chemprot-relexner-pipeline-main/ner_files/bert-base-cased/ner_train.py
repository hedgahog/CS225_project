import os
import random
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    get_scheduler, 
    DataCollatorForTokenClassification
)
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim import AdamW

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define labels and mappings
labels = ['O', 'B-GENE-Y', 'I-GENE-Y', 'B-CHEMICAL', 'I-CHEMICAL', 'B-GENE-N', 'I-GENE-N']
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}

# Load tokenizer and model
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
).to(device)

# Specify the log file path
# log_file_path = "result_log/training_logs_new.txt"

# Function to write log messages to a file
# def log_message(message, file_path=log_file_path):
#     with open(file_path, "a") as log_file:  # Open file in append mode
#         log_file.write(message + "\n")

# Load BIO-tagged data
def load_data(file_path, sample_fraction=0.01):
    tokens, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        current_tokens, current_labels = [], []
        for line in f:
            if line.strip() == "":
                if current_tokens:
                    tokens.append(current_tokens)
                    labels.append(current_labels)
                    current_tokens, current_labels = [], []
            else:
                token, label = line.strip().split()
                current_tokens.append(token)
                current_labels.append(label2id[label])
    return tokens, labels

train_tokens, train_labels = load_data('/home/akshatkrishna/chemprot-relexner-pipeline-main/ner_files/bio_ner_training_data.txt')

# Split into training and validation sets
train_tokens, val_tokens, train_labels, val_labels = train_test_split(
    train_tokens, train_labels, test_size=0.1, random_state=42
)

# Tokenization and alignment function
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        padding='max_length',  # Ensure equal length sequences
        max_length=128, 
        is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    aligned_labels = []

    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)  # Ignore padding tokens in loss calculation
            else:
                label_ids.append(labels[word_id])
        aligned_labels.append(label_ids)

    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs

# Create datasets
train_dataset = Dataset.from_dict({"tokens": train_tokens, "ner_tags": train_labels})
val_dataset = Dataset.from_dict({"tokens": val_tokens, "ner_tags": val_labels})

# Apply tokenization
train_dataset = train_dataset.map(
    tokenize_and_align_labels, batched=True, remove_columns=train_dataset.column_names
)
val_dataset = val_dataset.map(
    tokenize_and_align_labels, batched=True, remove_columns=val_dataset.column_names
)

# Data collator for dynamic padding (returns tensors)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset, batch_size=8, shuffle=True, pin_memory=True, 
    collate_fn=data_collator, num_workers=4
)
val_loader = DataLoader(
    val_dataset, batch_size=8, shuffle=False, pin_memory=True, 
    collate_fn=data_collator, num_workers=4
)

num_epochs = 50
# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
num_training_steps = len(train_loader) * num_epochs  # 10 epochs
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Initialize variables to track the minimum loss
best_val_loss = float("inf")
best_model_path = "./best_ner_model"

# Custom training loop with progress bars
if __name__ == "__main__":
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        print(f"[INFO] Epoch {epoch+1}/{num_epochs} started...")

        for batch in train_loader:
            # Move tensors to GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Log training loss
        avg_train_loss = total_loss / len(train_loader)
        print(f"[INFO] Epoch {epoch + 1} completed. Avg Training Loss: {avg_train_loss:.4f}")
        torch.cuda.empty_cache()

        # Evaluation loop with progress bar
        model.eval()
        total_eval_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                total_eval_loss += outputs.loss.item()

        # Calculate average validation loss
        avg_val_loss = total_eval_loss / len(val_loader)
        print(f"[INFO] Validation Loss after Epoch {epoch + 1}: {avg_val_loss:.4f}")

        # log_message(validation_message)

        # Save the model if the validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"[INFO] New best model saved with validation loss: {best_val_loss:.4f}")


print("Training completed!")