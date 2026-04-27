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

# Load BioGPT tokenizer and model
model_name = "microsoft/biogpt"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
).to(device)

# Log file setup
# log_file_path = "result_log/training_logs_biogpt.txt"
# os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# def log_message(message, file_path=log_file_path):
#     with open(file_path, "a") as log_file:
#         log_file.write(message + "\n")
#     print(message)

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

train_tokens, train_labels = load_data('/home/akshatkrishna/chemprot-relexner-pipeline-main/ner_files/bio_ner_training_data.txt')

# Split into training and validation sets
train_tokens, val_tokens, train_labels, val_labels = train_test_split(
    train_tokens, train_labels, test_size=0.1, random_state=42
)

def tokenize_and_align_labels(examples):
    tokenized_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

    for tokens, ner_tags in zip(examples["tokens"], examples["ner_tags"]):
        input_ids = []
        aligned_labels = []
        attention_mask = []

        # Process each word and its corresponding label
        for word, label in zip(tokens, ner_tags):
            word_tokens = tokenizer.tokenize(word)  # Tokenize the word into subwords
            if not word_tokens:  # Skip empty tokenized words
                continue

            word_ids = tokenizer.convert_tokens_to_ids(word_tokens)
            input_ids.extend(word_ids)
            attention_mask.extend([1] * len(word_ids))

            # Align labels: first subword gets the label, others get -100
            aligned_labels.append(label)
            aligned_labels.extend([-100] * (len(word_ids) - 1))

        # Add special tokens ([BOS] and [EOS]) manually
        input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
        attention_mask = [1] + attention_mask + [1]
        aligned_labels = [-100] + aligned_labels + [-100]

        # Ensure max_length truncation and padding
        max_length = 128
        input_ids = input_ids[:max_length] + [tokenizer.pad_token_id] * max(0, max_length - len(input_ids))
        attention_mask = attention_mask[:max_length] + [0] * max(0, max_length - len(attention_mask))
        aligned_labels = aligned_labels[:max_length] + [-100] * max(0, max_length - len(aligned_labels))

        # Append results
        tokenized_inputs["input_ids"].append(input_ids)
        tokenized_inputs["attention_mask"].append(attention_mask)
        tokenized_inputs["labels"].append(aligned_labels)

    return tokenized_inputs

# Create datasets
train_dataset = Dataset.from_dict({"tokens": train_tokens, "ner_tags": train_labels})
val_dataset = Dataset.from_dict({"tokens": val_tokens, "ner_tags": val_labels})


# Tokenize and align labels
train_dataset = train_dataset.map(
    tokenize_and_align_labels, batched=True, remove_columns=train_dataset.column_names
)
val_dataset = val_dataset.map(
    tokenize_and_align_labels, batched=True, remove_columns=val_dataset.column_names
)

# Data collator for dynamic padding
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=data_collator)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 50
num_training_steps = len(train_loader) * num_epochs
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Training loop
best_val_loss = float("inf")
best_model_path = "./best_biogpt_ner_model"

if __name__ == "__main__":
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1} - Training")
        # import ipdb; ipdb.set_trace()
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

            # train_progress_bar.set_postfix(loss=loss.item())

        # log_message(f"Epoch {epoch + 1} completed, Average Loss: {total_loss / len(train_loader):.4f}")
        avg_train_loss = total_loss / len(train_loader)
        print(f"[INFO] Epoch {epoch + 1} completed. Avg Training Loss: {avg_train_loss:.4f}")
        # Evaluation
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
        print(f"[INFO] Validation Loss after Epoch {epoch + 1}: {avg_val_loss:.4f}")
        # log_message(f"Validation Loss after Epoch {epoch + 1}: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"[INFO] New best model saved with validation loss: {best_val_loss:.4f}")
            # log_message(f"New best model saved with validation loss: {best_val_loss:.4f}")

# log_message("Training completed!")
print("Training completed!")