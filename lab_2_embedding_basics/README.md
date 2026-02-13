# Embedding Basics

This project contains utilities and notebooks for learning embedding concepts and
interacting with the SupportVectors Ray cluster embedding and chat APIs.

## Project Goals

This project is multi-fold:

1. Understand counter-intuitive properties of high-dimensional embeddings.
2. Learn embedding basics and visualize embeddings with PCA, t-SNE, and UMAP,
   including how sentence-BERT uses contrastive loss to address BERT anisotropy.
3. Study how multimodal embeddings of images cluster by class labels, with
   t-SNE and UMAP revealing structure more clearly than linear PCA.
4. Access the SV Ray cluster for text and image embeddings without downloading
   models locally or requiring direct GPU access.
5. Access the SV Ray cluster LLM `openai/gpt-oss-20b` to avoid external OpenAI
   token costs.

## Source Code (`src/embedding_basics`)

- `__init__.py` loads environment variables and reads configuration into `config`.
- `hf_text_utils.py` provides dataset utilities for sentence-label data, including
  train/test splitting from JSONL chunks, pair/triplet dataset generation, and
  evaluator helpers for sentence embeddings.
- `sv_ray_cluster_api.py` wraps the SupportVectors Ray cluster API for text/image
  embedding requests and exposes an OpenAI-compatible chat client helper.

## Notebooks (`docs/notebooks`)

- `00-conc-of-measures.ipynb` explores concentration of measure in high-dimensional
  embeddings with experiments and plots.
- `01-encoder-embeddings.ipynb` demonstrates BERT embedding anisotropy and PCA-based
  visualization across subject-labeled text.
- `02-siglip-embeddings.ipynb` walks through multimodal embeddings using sampled
  images and text inputs.
- `03-openai-chat-sample.ipynb` shows a chat completion example using the
  SupportVectors Ray cluster OpenAI-compatible client.
