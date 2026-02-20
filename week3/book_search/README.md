# Book Search Microservice

This project builds paragraph-level embeddings from CSV data, serves semantic search via FastAPI, and provides a NiceGUI frontend.

## Project Structure

- `data/week-3-experiment.csv`: Input dataset (`label,text`)
- `data/articles.csv`: Article table (`id,text`)
- `data/text_chunks.csv`: Chunk table (`id,chunk,article_id,vector_embedding`)
- `src/process_embeddings.py`: Data processing + embedding generation
- `src/search_api.py`: FastAPI semantic search service
- `src/search_ui.py`: NiceGUI frontend app

## 1) Install Dependencies

```bash
pip install -r requirements.txt
```

## 2) Build Article and Chunk Tables with Embeddings

From `book_search/`:

```bash
python src/process_embeddings.py
```

This reads from `data/week-3-experiment.csv` and writes:

- `data/articles.csv`
- `data/text_chunks.csv`

## 3) Run the Search API

```bash
uvicorn src.search_api:app --reload
```

API docs:

- `http://127.0.0.1:8000/docs`

### API Endpoints

- `GET /health`: Service health + loaded counts
- `POST /reload`: Reloads `data/text_chunks.csv` and `data/articles.csv`
- `POST /search`: Returns top-k chunk matches for a query
- `GET /articles/{doc_id}`: Returns article text by document ID

`/search` request body:

```json
{
  "search_text": "gravity and spacetime",
  "top_k": 10
}
```

`/search` response includes:

- `chunk_id`
- `chunk_text`
- `doc_id`
- `score` (cosine similarity, higher is better)

## 4) Run the NiceGUI UI

Keep the API running, then in another terminal:

```bash
python src/search_ui.py
```

Open:

- `http://127.0.0.1:8080`

Features:

- Search box + Search button
- Adjustable top-k value (default 10)
- Rabbit-running loading animation while searching
- Results table with:
  - `Chunk`
  - `Score`
  - `Link of Article` (opens `/articles/{doc_id}`)
- Results are sorted by highest score first

## Score Meaning

- `score` is cosine similarity between query embedding and chunk embedding.
- Higher score means a stronger semantic match.
- Typical range is approximately `[-1, 1]`.
