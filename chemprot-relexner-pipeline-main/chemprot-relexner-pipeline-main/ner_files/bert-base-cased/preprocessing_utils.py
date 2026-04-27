# preprocessing_utils.py

import random
from typing import List, Tuple
from nltk.corpus import stopwords
from collections import Counter

# Download stopwords if not already available
import nltk
nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

def remove_stopwords(tokens_list: List[List[str]], labels_list: List[List[int]]) -> Tuple[List[List[str]], List[List[int]]]:
    """Remove stopwords from tokens and align corresponding labels."""
    filtered_tokens_list, filtered_labels_list = [], []

    for tokens, labels in zip(tokens_list, labels_list):
        filtered_tokens, filtered_labels = [], []
        for token, label in zip(tokens, labels):
            if token.lower() not in STOPWORDS:  # Keep non-stopword tokens
                filtered_tokens.append(token)
                filtered_labels.append(label)
        if filtered_tokens:  # Ensure non-empty samples are retained
            filtered_tokens_list.append(filtered_tokens)
            filtered_labels_list.append(filtered_labels)

    return filtered_tokens_list, filtered_labels_list

def downsample_o_labels(tokens_list: List[List[str]], labels_list: List[List[int]], drop_prob: float = 0.9) -> Tuple[List[List[str]], List[List[int]]]:
    """Downsample 'O' labels to reduce class imbalance."""
    filtered_tokens_list, filtered_labels_list = [], []

    for tokens, labels in zip(tokens_list, labels_list):
        filtered_tokens, filtered_labels = [], []
        for token, label in zip(tokens, labels):
            if label != 0 or random.random() > drop_prob:  # Keep non-'O' or drop 'O' with probability
                filtered_tokens.append(token)
                filtered_labels.append(label)

        if filtered_tokens:  # Ensure non-empty samples are retained
            filtered_tokens_list.append(filtered_tokens)
            filtered_labels_list.append(filtered_labels)

    return filtered_tokens_list, filtered_labels_list

def filter_by_frequency(tokens_list: List[List[str]], labels_list: List[List[int]], min_freq: int = 2, max_freq: int = 10) -> Tuple[List[List[str]], List[List[int]]]:
    """Filter tokens based on their frequency in the dataset."""
    token_freq = Counter([token for tokens in tokens_list for token in tokens])

    filtered_tokens_list, filtered_labels_list = [], []
    for tokens, labels in zip(tokens_list, labels_list):
        filtered_tokens, filtered_labels = [], []
        for token, label in zip(tokens, labels):
            if min_freq <= token_freq[token] <= max_freq:  # Keep tokens within frequency range
                filtered_tokens.append(token)
                filtered_labels.append(label)

        if filtered_tokens:  # Ensure non-empty samples are retained
            filtered_tokens_list.append(filtered_tokens)
            filtered_labels_list.append(filtered_labels)

    return filtered_tokens_list, filtered_labels_list

def filter_o_chunks(tokens_list: List[List[str]], labels_list: List[List[int]], max_o_ratio: float = 0.6) -> Tuple[List[List[str]], List[List[int]]]:
    """Filter out samples with too many 'O' labels."""
    filtered_tokens_list, filtered_labels_list = [], []

    for tokens, labels in zip(tokens_list, labels_list):
        o_count = sum(1 for label in labels if label == 0)
        if o_count / len(labels) <= max_o_ratio:  # Keep if 'O' ratio is below threshold
            filtered_tokens_list.append(tokens)
            filtered_labels_list.append(labels)

    return filtered_tokens_list, filtered_labels_list

def get_weighted_loss(labels: List[int], num_labels: int):
    """Create a weighted loss to handle class imbalance."""
    from torch.nn import CrossEntropyLoss

    label_counts = Counter(labels)
    total = sum(label_counts.values())
    weights = [1.0 - (label_counts[i] / total) for i in range(num_labels)]
    return CrossEntropyLoss(weight=torch.tensor(weights))