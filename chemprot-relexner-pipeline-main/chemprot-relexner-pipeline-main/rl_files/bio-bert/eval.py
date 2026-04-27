import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
import pandas as pd

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the log_message function
log_file_path = "result_logs/evaluation_logs.txt"
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
def log_message(message, file_path=log_file_path):
    """Logs a message to a file and prints it to the console."""
    with open(file_path, "a") as log_file:
        log_file.write(message + "\n")
    print(message)

# Reverse label mapping (ensure this matches the mapping used in training)
label_mapping = {3: 0, 4: 1, 5: 2, 6: 3, 9: 4}
id2label = {v: k for k, v in label_mapping.items()}

# Load data
def load_data(file_path):
    """Loads text and label data from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    data = pd.read_csv(file_path)
    if "text" not in data.columns or "label" not in data.columns:
        raise ValueError(f"File {file_path} must contain 'text' and 'label' columns.")
    return data["text"].tolist(), data["label"].tolist()

# Tokenization function
def tokenize_function(texts, labels, tokenizer, max_length=512):
    """Tokenizes texts and aligns labels."""
    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True,
        return_token_type_ids=False,
    )
    return encodings, torch.tensor(labels)

# Function to evaluate the model
def evaluate_model(model, test_loader, device):
    """Evaluates the model and returns predictions and true labels."""
    model.eval()
    predictions, true_labels = [], []
    eval_progress_bar = tqdm(test_loader, desc="Evaluating", leave=True)
    
    with torch.no_grad():
        for batch in eval_progress_bar:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            # Collect predictions and true labels
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

    return np.array(predictions), np.array(true_labels)

# Load the best model and tokenizer
best_model_path = "./best_biobert_relation_extraction_model"
if not os.path.exists(best_model_path):
    raise FileNotFoundError(f"Trained model not found at: {best_model_path}")
model = AutoModelForSequenceClassification.from_pretrained(best_model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(best_model_path)

# Load test data
test_data_path = "../preprocessed_data/test_relation_extraction_dataset.csv"
test_texts, test_labels = load_data(test_data_path)
test_labels = [label_mapping[label] for label in test_labels]

# Tokenize test data
test_encodings, test_labels = tokenize_function(test_texts, test_labels, tokenizer)

# Create DataLoader for test data
test_data = torch.utils.data.TensorDataset(
    test_encodings["input_ids"], test_encodings["attention_mask"], test_labels
)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

# Evaluate the model
log_message("Starting evaluation...")
predictions, true_labels = evaluate_model(model, test_loader, device)

# Calculate metrics
precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels, predictions, average="weighted"
)
accuracy = accuracy_score(true_labels, predictions)

# Log results
log_message("\n--- Evaluation Results ---")
log_message(f"Accuracy: {accuracy:.4f}")
log_message(f"Precision: {precision:.4f}")
log_message(f"Recall: {recall:.4f}")
log_message(f"F1 Score: {f1:.4f}")

# Save evaluation results to a file
evaluation_results_path = "result_logs/evaluation_results.txt"
os.makedirs(os.path.dirname(evaluation_results_path), exist_ok=True)
with open(evaluation_results_path, "w") as f:
    f.write("--- Evaluation Results ---\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")

log_message("\n--- Evaluation Completed! ---")
log_message(f"Results saved in {evaluation_results_path}")
print("Evaluation results logged successfully.")