#  ------------------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------------------

from sentence_transformers import evaluation
from typing import Tuple, Generator, List
import json
import os
import glob
import random
import itertools
from datasets import Dataset, concatenate_datasets
from ruamel.yaml import CommentedMap

def yield_data(filename: str) -> Generator[Tuple[int, str], None, None]:
    """Generator of tuples of label, text

    Args:
        filename (str): input file

    Yields:
        Generator[Tuple[int, str], None, None]: Generator of tuples of label and text
    """
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            
            # Assign numerical labels based on subject
            label = 0
            if "Physics" == data["label"]:
                label = 1
            elif "History" == data["label"]:
                label = 2
                
            yield label, data["text"]

def get_train_test_lists(cfg: CommentedMap, split: float = 0.8) -> Tuple[List[Tuple[int, str]]]:
    """Get the train and test lists

    Args:
        cfg (CommentedMap): configuration read from config file. 
        split (float): split ratio for train and test data

    Returns:
        Tuple[List[Tuple[int, str]]]: The train and test lists of tuples containing label and text
    """
    data_dir = cfg["paths"]["data_dir"]

    # Use glob to find all JSONL files in the chunks directory
    jsonl_files = glob.glob(os.path.join(f"{data_dir}/chunks", "*.json"))

    # Collect examples from all JSONL files
    examples = [(label, text) for jsonl_file in jsonl_files for label, text in yield_data(jsonl_file)]
    examples = [example for example in examples if len(example[1]) > 100]
    random.shuffle(examples)
    train_size = int(len(examples) * split)
    train_list = examples[:train_size]
    test_list = examples[train_size:]
    return train_list, test_list

def tuples_list_to_dataset(tuples_list: List[Tuple[int, str]]) -> Dataset:
    """Converts list of tuples of label, text to Hugging face dataset

    Args:
        tuples_list (List[Tuple[int, str]]): A list of tuples containing subject label along with the text

    Returns:
        Dataset: The hugging face dataset with columns "label" and "text" with these two values
    """
    # Separate the labels and texts
    labels, texts = zip(*tuples_list)
    # Create a dictionary
    data_dict = {"label": labels, "text": texts}
    # Convert to Dataset
    return Dataset.from_dict(data_dict)

def get_evaluator(test_dataset: Dataset, truncate_dim: int = None) -> evaluation.SentenceEvaluator:
    """Gets the embeddings similarity evaluator from a sample of test dataset

    Args:
        test_dataset (Dataset): Test dataset 
        truncate_dim (int): For use with Matryoshka loss training - defaults to None

    Returns:
        evaluation.SentenceEvaluator: The evaluator returned
    """
    # Sample 5000 random entries from the test_dataset
    random.seed(42)
    sample_indices = random.sample(range(len(test_dataset)), 5000)
    sampled_dataset = test_dataset.select(sample_indices)
    sentences1 = sampled_dataset["sentence1"]
    sentences2 = sampled_dataset["sentence2"]
    labels = sampled_dataset["label"]

    # Set up the evaluator with binary similarity scores
    return evaluation.BinaryClassificationEvaluator(
        sentences1=sentences1,
        sentences2=sentences2,
        labels=labels,
        truncate_dim=truncate_dim,
    )

def convert_to_pair_dataset(sentence_label_dataset: Dataset) -> Dataset:
    """Generate pair dataset from sentence-label dataset

    Args:
        sentence_label_dataset (Dataset): sentence to class

    Returns:
        Dataset: pairs of sentences with a label indicating if they are same or different
    """
    sentences = sentence_label_dataset["text"]
    labels = sentence_label_dataset["label"]
    sentence_pair_dataset = {"sentence1":[], "sentence2":[], "label":[]}
    for (i, sent1), (j, sent2) in itertools.combinations(enumerate(sentences), 2):
        # If labels are the same, it's a similar pair (1), else dissimilar (0)
        score = 1 if labels[i] == labels[j] else 0
        sentence_pair_dataset["sentence1"].append(sent1)
        sentence_pair_dataset["sentence2"].append(sent2)
        sentence_pair_dataset["label"].append(score)  
    
    return Dataset.from_dict(sentence_pair_dataset) 

def convert_to_triplet_dataset(sentence_label_dataset: Dataset) -> Dataset:
    """Generate triplet dataset from sentence-label dataset for MultipleNegativesRankingLoss

    Args:
        sentence_label_dataset (Dataset): sentence to class dataset

    Returns:
        Dataset: triplets of (anchor, positive, negative) where anchor and positive 
                are from same subject and negative is from different subject
    """
    sentences = sentence_label_dataset["text"]
    labels = sentence_label_dataset["label"]
    
    # Group sentences by label
    label_to_sentences = {}
    for i, label in enumerate(labels):
        if label not in label_to_sentences:
            label_to_sentences[label] = []
        label_to_sentences[label].append((i, sentences[i]))
    
    triplet_dataset = {"anchor": [], "positive": [], "negative": []}
    
    # Generate triplets for each label
    for current_label in label_to_sentences:
        current_sentences = label_to_sentences[current_label]
        other_labels = [label for label in label_to_sentences if label != current_label]
        
        # For each sentence in current label, create triplets
        for i, (anchor_idx, anchor_sent) in enumerate(current_sentences):
            # Find positive examples (other sentences from same label)
            positive_candidates = [(idx, sent) for j, (idx, sent) in enumerate(current_sentences) if i != j]
            
            # Find negative examples (sentences from different labels)
            negative_candidates = []
            for other_label in other_labels:
                negative_candidates.extend(label_to_sentences[other_label])
            
            # Create triplets with each positive candidate
            for pos_idx, pos_sent in positive_candidates:
                # Randomly select a negative example
                if negative_candidates:
                    neg_idx, neg_sent = random.choice(negative_candidates)
                    triplet_dataset["anchor"].append(anchor_sent)
                    triplet_dataset["positive"].append(pos_sent)
                    triplet_dataset["negative"].append(neg_sent)
    
    return Dataset.from_dict(triplet_dataset)

def convert_to_anchor_positive_dataset(sentence_label_dataset: Dataset) -> Dataset:
    """Generate (anchor, positive) pair dataset from sentence-label dataset for MultipleNegativesRankingLoss

    Args:
        sentence_label_dataset (Dataset): sentence to class dataset

    Returns:
        Dataset: pairs of (anchor, positive) where both are from same subject
    """
    sentences = sentence_label_dataset["text"]
    labels = sentence_label_dataset["label"]
    
    # Group sentences by label
    label_to_sentences = {}
    for i, label in enumerate(labels):
        if label not in label_to_sentences:
            label_to_sentences[label] = []
        label_to_sentences[label].append((i, sentences[i]))
    
    pair_dataset = {"anchor": [], "positive": []}
    
    # Generate pairs for each label
    for current_label in label_to_sentences:
        current_sentences = label_to_sentences[current_label]
        
        # For each sentence in current label, create pairs with other sentences from same label
        for i, (anchor_idx, anchor_sent) in enumerate(current_sentences):
            # Find positive examples (other sentences from same label)
            positive_candidates = [(idx, sent) for j, (idx, sent) in enumerate(current_sentences) if i != j]
            
            # Create pairs with each positive candidate
            for pos_idx, pos_sent in positive_candidates:
                pair_dataset["anchor"].append(anchor_sent)
                pair_dataset["positive"].append(pos_sent)
    
    return Dataset.from_dict(pair_dataset)

def get_triplet_evaluator(test_dataset: Dataset, truncate_dim: int = None) -> evaluation.SentenceEvaluator:
    """Gets the embeddings similarity evaluator from a triplet test dataset

    Args:
        test_dataset (Dataset): Test dataset with triplet format (anchor, positive, negative)
        truncate_dim (int): For use with Matryoshka loss training - defaults to None

    Returns:
        evaluation.SentenceEvaluator: The evaluator returned
    """
    # Sample 5000 random entries from the test_dataset
    random.seed(42)
    sample_size = min(5000, len(test_dataset))
    sample_indices = random.sample(range(len(test_dataset)), sample_size)
    sampled_dataset = test_dataset.select(sample_indices)
    
    # Convert triplet dataset to pair format for evaluation
    # Create positive pairs (anchor, positive) with label 1
    sentences1 = list(sampled_dataset["anchor"]) 
    sentences2 = list(sampled_dataset["positive"]) 
    labels = [1] * len(sampled_dataset["anchor"]) 
    
    # Create negative pairs (anchor, negative) with label 0
    sentences1.extend(sampled_dataset["anchor"])
    sentences2.extend(sampled_dataset["negative"])
    labels.extend([0] * len(sampled_dataset["anchor"]))

    # Set up the evaluator with binary similarity scores
    return evaluation.BinaryClassificationEvaluator(
        sentences1=sentences1,
        sentences2=sentences2,
        labels=labels,
        truncate_dim=truncate_dim,
    )

def sampled_dataset(dataset: Dataset) -> Dataset:
    """Sample incoming dataset to max of 500 per label

    Args:
        dataset (Dataset): incoming dataset

    Returns:
        Dataset: sampled dataset containing 500 per label
    """
    # Initialize an empty list to hold sampled subsets
    samples = []

    # Loop through each label (assuming labels are 0, 1, and 2)
    for label in set(dataset['label']):
        # Filter rows with the current label
        label_subset = dataset.filter(lambda x, label=label : x['label'] == label)
        # Shuffle and select the first 500 samples (or fewer if less are available)
        label_sample = label_subset.shuffle(seed=42).select(range(min(500, len(label_subset))))
        # Append to samples list
        samples.append(label_sample)
    
    return concatenate_datasets(samples)  