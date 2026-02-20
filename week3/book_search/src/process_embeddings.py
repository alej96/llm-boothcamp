import argparse
import csv
import json
import re
import uuid
from pathlib import Path
from typing import Dict, List

from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraph chunks by blank lines."""
    if not text:
        return []

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n\s*\n+", normalized)
    return [p.strip() for p in parts if p and p.strip()]


def resolve_data_path(path_value: str) -> Path:
    """Resolve a user path, defaulting relative paths to the data directory."""
    path = Path(path_value)
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    return DATA_DIR / path


def load_articles_from_input(input_csv: Path) -> List[Dict[str, str]]:
    articles: List[Dict[str, str]] = []

    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "text" not in (reader.fieldnames or []):
            raise ValueError(f"Input file must contain a 'text' column: {input_csv}")

        for row in reader:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            articles.append({"id": str(uuid.uuid4()), "text": text})

    return articles


def write_articles_csv(articles: List[Dict[str, str]], articles_csv: Path) -> None:
    articles_csv.parent.mkdir(parents=True, exist_ok=True)
    with articles_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text"])
        writer.writeheader()
        writer.writerows(articles)


def build_chunks(articles: List[Dict[str, str]]) -> List[Dict[str, str]]:
    chunks: List[Dict[str, str]] = []

    for article in articles:
        article_id = article["id"]
        paragraphs = split_into_paragraphs(article["text"])

        for paragraph in paragraphs:
            chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "chunk": paragraph,
                    "article_id": article_id,
                }
            )

    return chunks


def embed_chunks(chunks: List[Dict[str, str]], model_name: str, batch_size: int) -> None:
    if not chunks:
        return

    model = SentenceTransformer(model_name)
    texts = [c["chunk"] for c in chunks]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    for chunk_row, vector in zip(chunks, embeddings):
        chunk_row["vector_embedding"] = json.dumps(vector.tolist())


def write_text_chunks_csv(chunks: List[Dict[str, str]], chunks_csv: Path) -> None:
    chunks_csv.parent.mkdir(parents=True, exist_ok=True)
    with chunks_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "chunk", "article_id", "vector_embedding"],
        )
        writer.writeheader()
        writer.writerows(chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create article + paragraph-chunk embedding tables from input CSV"
    )
    parser.add_argument(
        "--input",
        default="week-3-experiment.csv",
        help="Input CSV filename or path (must include a text column)",
    )
    parser.add_argument(
        "--articles",
        default="articles.csv",
        help="Output filename or path for articles table",
    )
    parser.add_argument(
        "--chunks",
        default="text_chunks.csv",
        help="Output filename or path for chunks table",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_csv = resolve_data_path(args.input)
    articles_csv = resolve_data_path(args.articles)
    chunks_csv = resolve_data_path(args.chunks)

    articles = load_articles_from_input(input_csv)
    write_articles_csv(articles, articles_csv)

    chunks = build_chunks(articles)
    embed_chunks(chunks, model_name=args.model, batch_size=args.batch_size)
    write_text_chunks_csv(chunks, chunks_csv)

    print(f"Data directory: {DATA_DIR}")
    print(f"Articles written: {len(articles)} -> {articles_csv}")
    print(f"Chunks written: {len(chunks)} -> {chunks_csv}")


if __name__ == "__main__":
    main()
