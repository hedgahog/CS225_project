import os
import pandas as pd
from tqdm import tqdm

def preprocess_relation_data(base_path, mode, output_dir):
    # Define file paths
    relations_path = os.path.join(base_path, f"{mode}/relations.tsv")
    entities_path = os.path.join(base_path, f"{mode}/entities.tsv")
    abstracts_path = os.path.join(base_path, f"{mode}/abstracts.tsv")

    # Load datasets
    relations = pd.read_csv(relations_path, sep='\t', names=["PMID", "CPR", "Evaluation", "Relation Group", "Arg1", "Arg2"], encoding='utf-8')
    entities = pd.read_csv(entities_path, sep='\t', names=["PMID", "Entity ID", "Type", "Start", "End", "Text"], encoding='utf-8')
    abstracts = pd.read_csv(abstracts_path, sep='\t', names=["PMID", "Title", "Abstract"], encoding='utf-8')
    # import ipdb; ipdb.set_trace()
    # Step 1: Filter rows with 'Y' in the Evaluation column
    relations = relations[relations["Evaluation"] == "Y "]

    # Step 2: Transform Arg1 and Arg2 to only contain entity IDs
    relations["Arg1"] = relations["Arg1"].str.split(":").str[1]
    relations["Arg2"] = relations["Arg2"].str.split(":").str[1]

    # Step 3: Map Arg1 and Arg2 to their corresponding entity text
    def map_entity_text(row, entities_df):
        arg1_text = entities_df[(entities_df["PMID"] == row["PMID"]) & (entities_df["Entity ID"] == row["Arg1"])]["Text"]
        arg2_text = entities_df[(entities_df["PMID"] == row["PMID"]) & (entities_df["Entity ID"] == row["Arg2"])]["Text"]
        return arg1_text.values[0] if not arg1_text.empty else None, arg2_text.values[0] if not arg2_text.empty else None

    
    # relations[["Arg1_Text", "Arg2_Text"]] = relations.apply(lambda row: pd.Series(map_entity_text(row, entities)), axis=1)
    # Create empty lists to store Arg1_Text and Arg2_Text
    arg1_text_list = []
    arg2_text_list = []

    # Iterate through each row in the relations DataFrame
    for index, row in tqdm(relations.iterrows()):
        # Map Arg1 and Arg2 to their corresponding entity text
        arg1_text, arg2_text = map_entity_text(row, entities)
        
        # Append the results to the lists
        arg1_text_list.append(arg1_text)
        arg2_text_list.append(arg2_text)

    # Add the lists as new columns to the DataFrame
    relations["Arg1_Text"] = arg1_text_list
    relations["Arg2_Text"] = arg2_text_list

    # Step 4: Merge abstracts to get the context
    relations = relations.merge(abstracts, on="PMID")

    # Step 5: Create the dataset for training
    def create_training_sample(row):
        return {
            "text": f"[CLS] {row['Arg1_Text']} [SEP] {row['Arg2_Text']} [SEP] {row['Abstract']}",
            "label": int(row["CPR"].split(":")[1])
        }

    training_samples = relations.apply(create_training_sample, axis=1)

    # Convert to DataFrame for easier handling
    train_data = pd.DataFrame(list(training_samples))

    # Save preprocessed data for model training
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{mode}_relation_extraction_dataset.csv")
    train_data.to_csv(output_path, index=False)
    print(f"Preprocessing complete for {mode}. Data saved to '{output_path}'.")

# Define base path and output directory
base_path = "/home/nateshreddy/igs/data/ChemProt_Corpus/split_data"  # Replace with your base path
output_dir = "./preprocessed_data"

# Preprocess and save data for train and test
preprocess_relation_data(base_path, "train", output_dir)
preprocess_relation_data(base_path, "test", output_dir)