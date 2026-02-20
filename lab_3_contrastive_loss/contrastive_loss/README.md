# Encoder Embeddings and Anisotropy

This module explores the inherent anisotropy problem in BERT-based embeddings and demonstrates how different encoder models perform at creating well-separated representations for different subject domains.

## Overview

The encoder embeddings notebook (`docs/notebooks/01-encoder-embeddings.ipynb`) demonstrates:

- **Anisotropy in BERT embeddings**: How standard BERT models produce embeddings that cluster in a narrow angular region
- **Comparison with Sentence Transformers**: Using BAAI/bge-base-en-v1.5 model trained with contrastive loss
- **Fine-tuned model evaluation**: Loading and testing a fine-tuned model specifically trained for subject classification
- **PCA visualization**: Linear 2D projection using PCA of high-dimensional embeddings to visualize the embeddings across three different models

## Prerequisites

- Python 3.8+
- PyTorch
- sentence-transformers
- scikit-learn
- matplotlib
- HuggingFace datasets

## Setup Instructions

### 1. Install Dependencies

```bash
uv sync
```

### 2. Download `chunks.tar.gz` Data from the course portal

Download the `chunks.tar.gz` from the course portal, inflate it into a folder on your local.

### 3. Update Configuration

The data path and the model results path are configured in `config.yaml`. Update the `data_dir` path to point to where you've placed the `chunks` folder.  Likewise update the `results_dir` to point to a folder where you have write access to save the fine-tuned models:

```yaml
paths:
  data_dir: /path/to/your/chunks/folder
  results_dir: /path/to/your/results/folder
```

## Expected Results

The notebook will demonstrate:

1. **BERT Model Anisotropy**: PCA visualization showing embeddings clustered in a narrow angular region, illustrating the inherent anisotropy problem in standard BERT models
2. **Sentence Transformer Improvement**: Better separation of embeddings using BAAI/bge-base-en-v1.5 model trained with contrastive loss
3. **Fine-tuned Model Performance**: Remarkable separation of embeddings into distinct clusters after fine-tuning with contrastive loss, making it very unlikely for different subjects to be mistaken for each other

## Key Concepts Demonstrated

- **Anisotropy**: The tendency of BERT embeddings to occupy a narrow cone in the embedding space
- **Subject Classification**: How well embeddings can distinguish between different academic subjects (Physics, Biology, History)
- **Contrastive Learning**: How contrastive loss training improves embedding separation
- **Model Comparison**: Side-by-side evaluation of BERT, Sentence Transformer, and fine-tuned models
- **Fine-tuning Process**: Loading and using pre-trained fine-tuned models for specific domain tasks

## References

1. [https://sbert.net/](https://sbert.net/)
2. [https://sbert.net/docs/quickstart.html](https://sbert.net/docs/quickstart.html) (for quickstart)
3. [https://sbert.net/docs/sentence_transformer/usage/usage.html](https://sbert.net/docs/sentence_transformer/usage/usage.html) (overview of usage)
4. [https://sbert.net/docs/sentence_transformer/training_overview.html](https://sbert.net/docs/sentence_transformer/training_overview.html) (training overview)
5. [https://sbert.net/docs/sentence_transformer/loss_overview.html](https://sbert.net/docs/sentence_transformer/loss_overview.html) (loss functions overview for sentence transformers)

