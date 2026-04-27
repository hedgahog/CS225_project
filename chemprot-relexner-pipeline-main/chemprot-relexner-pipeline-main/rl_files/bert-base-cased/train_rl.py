import os
import torch
from tqdm import tqdm
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_scheduler,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the log_message function
log_file_path = "result_logs/training_logs.txt"
def log_message(message, file_path=log_file_path):
    with open(file_path, "a") as log_file:
        log_file.write(message + "\n")
    print(message)

# Load data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data["text"].tolist(), data["label"].tolist()

# Load train and test datasets
train_texts, train_labels = load_data("/data/akshatkrishna/chemprot-relexner-pipeline-main/rl_files/preprocessed_data/train_relation_extraction_dataset.csv")
test_texts, test_labels = load_data("/data/akshatkrishna/chemprot-relexner-pipeline-main/rl_files/preprocessed_data/test_relation_extraction_dataset.csv")

# Map labels to sequential values
label_mapping = {3: 0, 4: 1, 5: 2, 6: 3, 9: 4}
train_labels = [label_mapping[label] for label in train_labels]
test_labels = [label_mapping[label] for label in test_labels]

# Tokenizer and model
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(label_mapping)
).to(device)

# Tokenization function
def tokenize_function(texts, labels):
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
        return_attention_mask=True,
        return_token_type_ids=False,
    ), torch.tensor(labels)

# Tokenize train and test data
train_encodings, train_labels = tokenize_function(train_texts, train_labels)
test_encodings, test_labels = tokenize_function(test_texts, test_labels)

# DataLoaders
train_data = torch.utils.data.TensorDataset(
    train_encodings["input_ids"], train_encodings["attention_mask"], train_labels
)
test_data = torch.utils.data.TensorDataset(
    test_encodings["input_ids"], test_encodings["attention_mask"], test_labels
)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

num_epochs = 50
# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
num_training_steps = len(train_loader) * num_epochs  # 10 epochs
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Mixed precision scaler
scaler = GradScaler()

# Training loop with mixed precision and gradient accumulation
gradient_accumulation_steps = 4  # Adjust based on GPU memory
best_val_loss = float("inf")
best_model_path = "./best_relation_extraction_model"

for epoch in range(num_epochs):  # Reduce epochs if necessary
    model.train()
    total_loss = 0

    # Training loop
    train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1} - Training")
    for step, batch in enumerate(train_progress_bar):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        with autocast():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            lr_scheduler.step()

        total_loss += loss.item() * gradient_accumulation_steps
        train_progress_bar.set_postfix(loss=loss.item())

    # Validation
    model.eval()
    total_eval_loss = 0
    eval_progress_bar = tqdm(test_loader, desc=f"Epoch {epoch + 1} - Evaluation")
    with torch.no_grad():
        for batch in eval_progress_bar:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            with autocast():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            total_eval_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    avg_val_loss = total_eval_loss / len(test_loader)

    log_message(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model.save_pretrained(best_model_path)
        tokenizer.save_pretrained(best_model_path)
        log_message(f"New best model saved with validation loss: {best_val_loss:.4f}")

# Final log
log_message("Training completed!")
print("Training completed!")