import torch
from transformers import BertTokenizer, BertForTokenClassification, pipeline
import pandas as pd
import re
import nltk
from tqdm import tqdm

# Download the sentence tokenizer
nltk.download('punkt')

# Load NER Model and Tokenizer
path = './best_ner_model'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained(path)
ner_model = BertForTokenClassification.from_pretrained(path).to(device)

# Create NER pipeline with GPU support
ner_pipeline = pipeline(
    "ner", model=ner_model, tokenizer=tokenizer, aggregation_strategy="simple",
    device=0 if torch.cuda.is_available() else -1
)

# Combine NER results to handle subwords (e.g., ##token)
def combine_ner_results(results):
    combined_results = []
    current_entity = None

    for entry in results:
        word = entry['word']
        entity_group = entry['entity_group']
        score = entry['score']

        if word.startswith("##"):  # Combine with the previous entity
            if current_entity:
                current_entity['word'] += word[2:]  # Remove ##
                current_entity['score'] = max(current_entity['score'], score)
        else:
            if current_entity:
                combined_results.append(current_entity)
            current_entity = {
                'entity_group': entity_group,
                'score': score,
                'word': word
            }

    if current_entity:
        combined_results.append(current_entity)

    return combined_results

# Load datasets and drop rows with NaN values
abstracts = pd.read_csv('/home/nateshreddy/igs/data/ChemProt_Corpus/split_data/test/abstracts.tsv', 
                        sep='\t', encoding='utf-8', 
                        names=['ID', 'Title', 'Abstract']).dropna()

entities = pd.read_csv('/home/nateshreddy/igs/data/ChemProt_Corpus/split_data/test/entities.tsv', 
                       sep='\t', encoding='utf-8', 
                       names=['ID', 'Entity term number', 'Entity type', 
                              'Start offset', 'End offset', 'Entity text']).dropna()

# Merge abstracts and entities based on ID
merged_data = pd.merge(abstracts, entities, on='ID')

# Group data by abstract
abstract_groups = merged_data.groupby('ID')

# Specify log file
log_file_path = "result_log/eval-bert-base-cased.txt"

# Function to log messages to a file
def log_message(message):
    with open(log_file_path, "a") as log_file:
        log_file.write(message + "\n")

# Initialize metrics
total_true_values = 0
total_true_positive = 0
total_false_positive = 0

# Log start of the evaluation
log_message("Starting NER Evaluation...\n")

# Process each abstract
for abstract_id, group in tqdm(abstract_groups):
    abstract_text = group['Abstract'].iloc[0]
    
    ## Original
    # true_entities = set([item.lower() for item in group['Entity text']])

    # # Perform NER on the abstract
    # ner_results = ner_pipeline(abstract_text)
    # predicted_entities = set([item['word'].lower() for item in combine_ner_results(ner_results)])

    ## Remove spaces
    # Preprocess true entities: remove spaces and convert to lowercase
    true_entities = set([item.replace(" ", "").lower() for item in group['Entity text']])

    # Perform NER on the abstract
    ner_results = ner_pipeline(abstract_text)

    # Preprocess predicted entities: remove spaces and convert to lowercase
    predicted_entities = set([item['word'].replace(" ", "").lower() for item in combine_ner_results(ner_results)])

    # Calculate true positives, false negatives, and false positives
    true_positive = true_entities.intersection(predicted_entities)
    false_positive = predicted_entities.difference(true_entities)
    false_negative = true_entities.difference(predicted_entities)

    # Update metrics
    total_true_values += len(true_entities)
    total_true_positive += len(true_positive)
    total_false_positive += len(false_positive)

    # import ipdb; ipdb.set_trace()

# Calculate precision, recall, and F1-score
precision = total_true_positive / (total_true_positive + total_false_positive) if (total_true_positive + total_false_positive) > 0 else 0
recall = total_true_positive / total_true_values if total_true_values > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Log final metrics
log_message("Final Evaluation Metrics:")
log_message(f"Total True Values: {total_true_values}")
log_message(f"Correctly Predicted (True Positives): {total_true_positive}")
log_message(f"Extra Predicted (False Positives): {total_false_positive}")
log_message(f"Precision: {precision:.4f}")
log_message(f"Recall: {recall:.4f}")
log_message(f"F1-Score: {f1_score:.4f}")


log_message("NER Evaluation Completed!")