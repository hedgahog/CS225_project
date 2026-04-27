import os
import torch
from tqdm import tqdm
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

# Load PubMedBERT tokenizer and model
model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
).to(device)

# Log file setup
log_file_path = "result_log/training_logs_pubmedbert.txt"
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

def log_message(message, file_path=log_file_path):
    with open(file_path, "a") as log_file:
        log_file.write(message + "\n")
    print(message)

# Load BIO-tagged data
def load_data(file_path):
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

train_tokens, train_labels = load_data('../bio_ner_training_data.txt')

# Split into training and validation sets
train_tokens, val_tokens, train_labels, val_labels = train_test_split(
    train_tokens, train_labels, test_size=0.1, random_state=42
)

# Tokenization and alignment function
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        padding='max_length',
        max_length=128, 
        is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    aligned_labels = []

    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)  # Special tokens get -100
            elif word_id != previous_word_idx:
                label_ids.append(labels[word_id])  # First sub-token gets label
            else:
                label_ids.append(-100)  # Sub-token gets -100
            previous_word_idx = word_id
        aligned_labels.append(label_ids)

    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs

# Create datasets
train_dataset = Dataset.from_dict({"tokens": train_tokens, "ner_tags": train_labels})
val_dataset = Dataset.from_dict({"tokens": val_tokens, "ner_tags": val_labels})

train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=val_dataset.column_names)

# Data collator for dynamic padding
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=data_collator)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 10
num_training_steps = len(train_loader) * num_epochs
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Training loop
best_val_loss = float("inf")
best_model_path = "./best_pubmedbert_ner_model"

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1} - Training")

    for batch in train_progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        train_progress_bar.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_loader)
    log_message(f"Epoch {epoch + 1} completed, Average Training Loss: {avg_train_loss:.4f}")

    # Evaluation loop
    model.eval()
    total_eval_loss = 0
    eval_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1} - Evaluation")
    with torch.no_grad():
        for batch in eval_progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            total_eval_loss += outputs.loss.item()

    avg_val_loss = total_eval_loss / len(val_loader)
    log_message(f"Validation Loss after Epoch {epoch + 1}: {avg_val_loss:.4f}")

    # Save the model if validation loss improves
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model.save_pretrained(best_model_path)
        tokenizer.save_pretrained(best_model_path)
        log_message(f"Best model saved with Validation Loss: {best_val_loss:.4f}")

log_message("Training completed!")
print("Training completed!")