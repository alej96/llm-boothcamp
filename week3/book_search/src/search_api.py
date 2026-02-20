import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CHUNKS_PATH = DATA_DIR / "text_chunks.csv"
DEFAULT_ARTICLES_PATH = DATA_DIR / "articles.csv"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SearchRequest(BaseModel):
    search_text: str = Field(..., min_length=1, description="User search query")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")


class SearchResult(BaseModel):
    chunk_id: str = Field(..., description="Unique ID of the matched chunk")
    chunk_text: str = Field(..., description="Chunk text content")
    doc_id: str = Field(..., description="Source article ID")
    score: float = Field(
        ..., description="Cosine similarity score between query and chunk embedding"
    )


class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: List[SearchResult]


class ChunkEmbedding(BaseModel):
    id: str
    chunk: str
    article_id: str
    vector_embedding: List[float]


app = FastAPI(title="Chunk Search Service", version="1.0.0")


@app.on_event("startup")
def startup() -> None:
    app.state.model = SentenceTransformer(MODEL_NAME)
    app.state.records, app.state.embedding_matrix = load_chunk_embeddings(DEFAULT_CHUNKS_PATH)
    app.state.article_map = load_articles(DEFAULT_ARTICLES_PATH)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "chunks_loaded": len(getattr(app.state, "records", [])),
        "articles_loaded": len(getattr(app.state, "article_map", {})),
        "model": MODEL_NAME,
    }


@app.post("/reload")
def reload_index() -> dict:
    records, embedding_matrix = load_chunk_embeddings(DEFAULT_CHUNKS_PATH)
    app.state.records = records
    app.state.embedding_matrix = embedding_matrix
    app.state.article_map = load_articles(DEFAULT_ARTICLES_PATH)
    return {
        "status": "reloaded",
        "chunks_loaded": len(records),
        "articles_loaded": len(app.state.article_map),
    }


@app.get("/articles/{doc_id}")
def get_article(doc_id: str) -> dict:
    article_map: Dict[str, str] = getattr(app.state, "article_map", {})
    article_text = article_map.get(doc_id)
    if article_text is None:
        raise HTTPException(status_code=404, detail=f"Article not found: {doc_id}")
    return {"id": doc_id, "text": article_text}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    if not hasattr(app.state, "records") or not hasattr(app.state, "embedding_matrix"):
        raise HTTPException(status_code=503, detail="Search index is not initialized")

    records: List[ChunkEmbedding] = app.state.records
    matrix: np.ndarray = app.state.embedding_matrix

    if matrix.size == 0:
        return SearchResponse(query=request.search_text, top_k=request.top_k, results=[])

    query_embedding = app.state.model.encode(
        request.search_text,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)

    top_indices, top_scores = knn_search(matrix, query_embedding, request.top_k)

    results = [
        SearchResult(
            chunk_id=records[idx].id,
            chunk_text=records[idx].chunk,
            doc_id=records[idx].article_id,
            score=float(score),
        )
        for idx, score in zip(top_indices, top_scores)
    ]

    return SearchResponse(query=request.search_text, top_k=len(results), results=results)


def load_articles(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}

    article_map: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_columns = {"id", "text"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

        for row in reader:
            article_id = (row.get("id") or "").strip()
            article_text = (row.get("text") or "").strip()
            if article_id:
                article_map[article_id] = article_text

    return article_map


def load_chunk_embeddings(path: Path) -> tuple[List[ChunkEmbedding], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")

    records: List[ChunkEmbedding] = []
    vectors: List[List[float]] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_columns = {"id", "chunk", "article_id", "vector_embedding"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

        for row in reader:
            vector_raw = row.get("vector_embedding", "")
            if not vector_raw:
                continue

            try:
                vector = json.loads(vector_raw)
            except json.JSONDecodeError:
                continue

            if not isinstance(vector, list) or not vector:
                continue

            try:
                vector_floats = [float(v) for v in vector]
            except (TypeError, ValueError):
                continue

            record = ChunkEmbedding(
                id=str(row["id"]),
                chunk=str(row["chunk"]),
                article_id=str(row["article_id"]),
                vector_embedding=vector_floats,
            )
            records.append(record)
            vectors.append(vector_floats)

    if not vectors:
        return records, np.empty((0, 0), dtype=np.float32)

    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    matrix = matrix / norms

    return records, matrix


def knn_search(
    normalized_matrix: np.ndarray,
    query_vector: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    query_norm = np.linalg.norm(query_vector)
    if query_norm <= 1e-12:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    normalized_query = query_vector / query_norm
    similarity_scores = normalized_matrix @ normalized_query

    k = min(top_k, similarity_scores.shape[0])
    if k == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    candidate_idx = np.argpartition(-similarity_scores, k - 1)[:k]
    ranked_idx = candidate_idx[np.argsort(-similarity_scores[candidate_idx])]

    return ranked_idx, similarity_scores[ranked_idx]
