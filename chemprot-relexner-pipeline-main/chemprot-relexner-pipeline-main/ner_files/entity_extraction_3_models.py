import os
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from itertools import combinations
import nltk
import re
import transformers
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

labels = ['O', 'B-GENE-Y', 'I-GENE-Y', 'B-CHEMICAL', 'I-CHEMICAL', 'B-GENE-N', 'I-GENE-N']
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}

# Multiple models
model_paths = {
    "bert-base-cased": "/data/akshatkrishna/chemprot-relexner-pipeline-main/ner_files/bert-base-cased/best_ner_model",
    "biobert": "/data/akshatkrishna/chemprot-relexner-pipeline-main/ner_files/bio-bert/best_biobert_ner_model",
    "biogpt": "/data/akshatkrishna/chemprot-relexner-pipeline-main/ner_files/biogpt/best_biogpt_ner_model"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_entity_text(text):
    words = text.strip().split()
    deduped = []
    for word in words:
        if not deduped or deduped[-1].lower() != word.lower():
            deduped.append(word)
    return re.sub(r"^[^\w\d]+|[^\w\d]+$", "", " ".join(deduped))

def load_ner_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(
        model_path, num_labels=len(labels), id2label=id2label, label2id=label2id
    )
    is_fast = isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
    return tokenizer, model.to(device).eval(), is_fast

def predict_entities(sentence, tokenizer, model, is_fast):
    if not is_fast:
        return []
    tokens = sentence.split()
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,  # or 256 if you want to be conservative
        padding="max_length"  # optional, if batching later
    ).to(device)
    with torch.no_grad():
        outputs = model(**encoding).logits

    predictions = torch.argmax(outputs, dim=2).squeeze().tolist()
    word_ids = encoding.word_ids()

    entities = []
    current_entity = None

    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        label = id2label[predictions[idx]]
        token_text = tokens[word_idx]

        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "text": token_text,
                "type": label[2:],
                "start": word_idx,
                "end": word_idx
            }
        elif label.startswith("I-") and current_entity and label[2:] == current_entity["type"]:
            current_entity["text"] += " " + token_text
            current_entity["end"] = word_idx
        elif label == "O":
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    # Clean and return
    cleaned = []
    for ent in entities:
        cleaned_text = clean_entity_text(ent["text"])
        cleaned.append({
            "text": cleaned_text,
            "type": ent["type"],
            "start": ent["start"],
            "end": ent["end"]
        })
    return cleaned

def generate_entity_pairs(entities):
    return list(combinations(entities, 2)) if len(entities) >= 2 else []

def load_sentences_from_abstracts(file_paths, max_sentences=500):
    all_sentences = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, sep="\t", names=["ID", "Title", "Abstract"])
        for _, row in df.iterrows():
            combined = f"{row['Title']} {row['Abstract']}"
            sentences = sent_tokenize(combined)
            all_sentences.extend(sentences)
            if max_sentences and len(all_sentences) >= max_sentences:
                return all_sentences[:max_sentences]
    return all_sentences

def merge_entities_across_models(model_outputs):
    entity_map = {}

    for model_name, entities in model_outputs.items():
        for ent in entities:
            key = (ent["text"].lower(), ent["type"], ent["start"], ent["end"])
            if key not in entity_map:
                entity_map[key] = {
                    "text": ent["text"],
                    "type": ent["type"],
                    "start": ent["start"],
                    "end": ent["end"],
                    "models": set()
                }
            entity_map[key]["models"].add(model_name)

    merged_entities = []
    for key, ent_data in entity_map.items():
        merged_entities.append({
            "text": ent_data["text"],
            "type": ent_data["type"],
            "start": ent_data["start"],
            "end": ent_data["end"],
            "models": list(ent_data["models"]),
            "confidence": round(len(ent_data["models"]) / len(model_paths), 2)
        })
    return merged_entities

def process_and_merge_all_models(sentences, output_json_path):
    all_results = []

    for sentence in sentences:
        model_outputs = {}
        for model_name, model_path in model_paths.items():
            tokenizer, model, is_fast = load_ner_model(model_path)
            entities = predict_entities(sentence, tokenizer, model, is_fast)
            model_outputs[model_name] = entities

        merged_entities = merge_entities_across_models(model_outputs)
        entity_pairs = generate_entity_pairs(merged_entities)

        all_results.append({
            "sentence": sentence,
            "entities": merged_entities,
            "entity_pairs": [
                {
                    "head": {"text": e1["text"], "type": e1["type"]},
                    "tail": {"text": e2["text"], "type": e2["type"]}
                }
                for e1, e2 in entity_pairs
            ]
        })

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"[✅ SUCCESS] Merged entity results saved to {output_json_path}")

if __name__ == "__main__":
    chemprot_files = [
        "/data/akshatkrishna/chemprot-relexner-pipeline-main/ChemProt_Corpus/chemprot_test_gs/chemprot_test_abstracts_gs.tsv"
    ]
    sentences = load_sentences_from_abstracts(chemprot_files, max_sentences=1000)
    process_and_merge_all_models(sentences, output_json_path="merged_entity_output.json")