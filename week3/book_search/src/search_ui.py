import asyncio
import json
import os
import urllib.error
import urllib.request

from nicegui import ui


API_BASE_URL = os.getenv("SEARCH_API_URL", "http://127.0.0.1:8000")


def call_search_api(search_text: str, top_k: int) -> list[dict]:
    payload = json.dumps({"search_text": search_text, "top_k": top_k}).encode("utf-8")
    request = urllib.request.Request(
        url=f"{API_BASE_URL}/search",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))

    results = data.get("results", [])
    return sorted(results, key=lambda r: float(r.get("score", -1.0)), reverse=True)


ui.add_head_html(
    """
    <style>
    .rabbit-track {
      position: relative;
      width: 100%;
      height: 34px;
      overflow: hidden;
      border-radius: 8px;
      background: linear-gradient(90deg, #f8fafc, #eef2ff);
      border: 1px solid #e2e8f0;
    }

    .rabbit-runner {
      position: absolute;
      left: -40px;
      top: 3px;
      font-size: 24px;
      animation: rabbit_run 1.1s linear infinite;
    }

    @keyframes rabbit_run {
      0% { transform: translateX(0px) translateY(0px); }
      25% { transform: translateX(25vw) translateY(-2px); }
      50% { transform: translateX(50vw) translateY(1px); }
      75% { transform: translateX(75vw) translateY(-1px); }
      100% { transform: translateX(110vw) translateY(0px); }
    }
    </style>
    """
)

ui.page_title("Book Search UI")

with ui.column().classes("w-full max-w-6xl mx-auto p-6 gap-4"):
    ui.label("Semantic Search").classes("text-3xl font-bold")
    ui.label("Search for the most relevant chunks and inspect score + source article.")

    with ui.row().classes("w-full items-end gap-3"):
        query_input = ui.input("Search text", placeholder="Type your query...").classes(
            "w-full"
        )
        top_k_input = ui.number("Top results", value=10, min=1, max=100, step=1).classes(
            "w-32"
        )
        search_button = ui.button("Search")

    loading_box = ui.column().classes("w-full")
    with loading_box:
        ui.html('<div class="rabbit-track"><div class="rabbit-runner">&#128007;</div></div>')
        ui.label("Searching...").classes("text-sm text-gray-600")
    loading_box.visible = False

    status_label = ui.label("").classes("text-sm")

    columns = [
        {"name": "chunk", "label": "Chunk", "field": "chunk", "align": "left"},
        {
            "name": "score",
            "label": "Score",
            "field": "score",
            "align": "right",
            "sortable": True,
        },
        {
            "name": "article_link",
            "label": "Link of Article",
            "field": "article_link",
            "align": "left",
        },
    ]
    table = ui.table(columns=columns, rows=[], row_key="chunk_id").classes("w-full")
    table.add_slot(
        "body-cell-article_link",
        """
        <q-td :props="props">
          <a :href="props.row.article_link" target="_blank" rel="noopener noreferrer">Open Article</a>
        </q-td>
        """,
    )


async def run_search() -> None:
    query = (query_input.value or "").strip()
    if not query:
        ui.notify("Type a search query first", type="warning")
        return

    try:
        top_k = int(top_k_input.value or 10)
    except (TypeError, ValueError):
        top_k = 10

    top_k = max(1, min(top_k, 100))

    loading_box.visible = True
    status_label.text = ""
    search_button.disable()

    try:
        results = await asyncio.to_thread(call_search_api, query, top_k)

        rows = [
            {
                "chunk_id": item.get("chunk_id", ""),
                "chunk": item.get("chunk_text", ""),
                "score": f"{float(item.get('score', 0.0)):.4f}",
                "score_value": float(item.get("score", 0.0)),
                "article_link": f"{API_BASE_URL}/articles/{item.get('doc_id', '')}",
            }
            for item in results
        ]

        rows.sort(key=lambda r: r["score_value"], reverse=True)
        for row in rows:
            row.pop("score_value", None)

        table.rows = rows
        table.update()
        status_label.text = f"Returned {len(rows)} result(s), sorted by highest score first."
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        status_label.text = "Search failed."
        ui.notify(f"API error {e.code}: {detail}", type="negative")
    except urllib.error.URLError:
        status_label.text = "Search failed."
        ui.notify(
            f"Cannot reach API at {API_BASE_URL}. Start FastAPI first.", type="negative"
        )
    except Exception as e:
        status_label.text = "Search failed."
        ui.notify(f"Unexpected error: {e}", type="negative")
    finally:
        loading_box.visible = False
        search_button.enable()


search_button.on_click(run_search)
query_input.on("keydown.enter", lambda _: run_search())


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="Book Search UI", reload=True)
