import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fine-tuned RE models
model_paths = {
    "bert-base-cased": "/data/akshatkrishna/chemprot-relexner-pipeline-main/ner_files/bert-base-cased/best_ner_model",
    "biobert": "/data/akshatkrishna/chemprot-relexner-pipeline-main/ner_files/bio-bert/best_biobert_ner_model",
    "biogpt": "/data/akshatkrishna/chemprot-relexner-pipeline-main/ner_files/biogpt/best_biogpt_ner_model"
}

label_map = {
    0: "CPR:3",
    1: "CPR:4",
    2: "CPR:5",
    3: "CPR:6",
    4: "CPR:9",
    5: "other",         # ← just examples
    6: "no_relation"    # ← default rejection class
}

def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model.to(device).eval()

def format_input(sentence, head_entity, tail_entity):
    return f"[CLS] {head_entity.strip()} [SEP] {tail_entity.strip()} [SEP] {sentence.strip()}"

def predict_relation(tokenizer, model, sentence, head, tail):
    input_text = format_input(sentence, head, tail)
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()

    return label_map[pred]

def merge_relation_predictions(all_model_outputs):
    merged = {}

    for model_name, predictions in all_model_outputs.items():
        for item in predictions:
            sentence = item["sentence"]
            for rel in item["relations"]:
                key = (sentence, rel["head"], rel["tail"], rel["relation"])
                if key not in merged:
                    merged[key] = {
                        "sentence": sentence,
                        "head": rel["head"],
                        "tail": rel["tail"],
                        "relation": rel["relation"],
                        "models": [model_name]
                    }
                else:
                    merged[key]["models"].append(model_name)

    # Add confidence
    for val in merged.values():
        val["confidence"] = round(len(val["models"]) / len(model_paths), 2)

    return list(merged.values())

def run_relation_extraction(input_json_path, output_json_path):
    with open(input_json_path, "r") as f:
        entity_data = json.load(f)

    # Using NER results from bert-base-cased (common base)
    base_data = entity_data

    all_model_outputs = {}

    for model_name, model_path in model_paths.items():
        print(f"[INFO] Running {model_name}")
        tokenizer, model = load_model_and_tokenizer(model_path)
        model_output = []

        for sample in tqdm(base_data):
            sentence = sample["sentence"]
            predictions = []

            for pair in sample["entity_pairs"]:
                head = pair["head"]["text"]
                tail = pair["tail"]["text"]
                relation = predict_relation(tokenizer, model, sentence, head, tail)

                if relation != "no_relation":
                    predictions.append({
                        "head": head,
                        "tail": tail,
                        "relation": relation
                    })

            if predictions:
                model_output.append({
                    "sentence": sentence,
                    "relations": predictions
                })

        all_model_outputs[model_name] = model_output

    merged_output = merge_relation_predictions(all_model_outputs)

    with open(output_json_path, "w") as f:
        json.dump(merged_output, f, indent=2)
        print(f"[SUCCESS] Saved to {output_json_path}")

if __name__ == "__main__":
    run_relation_extraction(
        input_json_path="/data/akshatkrishna/chemprot-relexner-pipeline-main/ner_files/merged_entity_output.json",
        output_json_path="merged_relation_predictions_all_models.json"
    )
# import json
# with open("/data/akshatkrishna/chemprot-relexner-pipeline-main/ner_files/merged_entity_output.json", "r") as f:
#     entity_data = json.load(f)

# print(type(entity_data))
# print(entity_data[0])