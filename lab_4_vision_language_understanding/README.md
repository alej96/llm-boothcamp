# Vision–Language Understanding

This topic covers vision–language models and related concepts through a sequence of notebooks and supporting library code. Notebooks are ordered by prefix (`00_` through `07_`) and should be followed in numerical order. All notebooks live under `docs/notebooks/`.

---

## Notebooks and underlying code

### 00 — ViT attention visualization (`00_vit_attention.ipynb`)

**What it covers:** How a Vision Transformer (ViT) “looks” at an image. The notebook loads a pre-trained ViT (e.g. `google/vit-base-patch16-224`), runs a forward pass with `output_attentions=True`, and visualizes attention from the [CLS] token to spatial patches—both as a grid of attention heads and as an overlay on the input image.

**Underlying code:** `src/svlearn_vlu/vit_attention_maps.py`

- **`load_image`** — Loads an image from a path or URL and returns preprocessed `pixel_values` and the PIL image.
- **`get_attention_maps`** — Runs the ViT encoder and returns attention weights for a chosen layer (with fallbacks for CPU and eager attention when the default backend does not return weights).
- **`plot_attention_heads`** — Plots a grid of attention maps (one per head) from [CLS] to patches.
- **`plot_attention_overlay`** — Overlays mean attention (across heads) onto the image to show which regions the model attends to.

---

### 01 — CLIP (`01_clip.ipynb`)

**What it covers:** Contrastive Language–Image Pre-training (CLIP): a dual-encoder model that maps images and text into a shared embedding space. The notebook explains the architecture (ViT image encoder + text transformer), contrastive loss (e.g. multiclass N-pair / batch negatives), and demonstrates **image retrieval**—encoding a text query and finding the most similar images in a pre-computed embedding collection.

**Underlying code:** `src/svlearn_vlu/clip/`

- **`embed_images.py`** — Defines `CLIPEncoder`: loads CLIP model and processor, connects to Qdrant, creates a collection (e.g. `tree_embeddings` from config), and embeds images from a directory into vectors stored in Qdrant.
- **`query_images.py`** — Encodes a text query with CLIP, normalizes the embedding, and runs a similarity search in the Qdrant collection (`search_similar_images`) to return the top-k matching images.

---

### 02 — BLIP (`02_blip.ipynb`)

**What it covers:** Bootstrapping Language–Image Pre-training (BLIP): a model that unifies image-grounded text encoding and caption generation. The notebook walks through the dual-encoder + multimodal encoder (with cross-attention) and the causal text decoder used for **image captioning**, and shows how to generate captions for a set of images.

**Underlying code:** `src/svlearn_vlu/blip/`

- **`captioner.py`** — Defines `BlipCaptioner`: loads `BlipForConditionalGeneration` and `BlipProcessor` (e.g. `Salesforce/blip-image-captioning-base`), and provides `_generate_caption` (single image with optional prompt) and `generate_captions` (batch over an image directory, with optional JSON export).

---

### 03 — BLIP-2 (`03_blip-2.ipynb`)

**What it covers:** BLIP-2’s Q-Former: a lightweight module that sits between a **frozen** image encoder and a **frozen** LLM. The notebook explains the Image Transformer (learnable query embeddings + cross-attention to visual features) and the Text Transformer (encoder/decoder), and the vision–language representation learning objectives (with different attention masks). It demonstrates captioning using the same pattern as BLIP but with a larger, more capable setup (e.g. OPT 2.7B).

**Underlying code:** `src/svlearn_vlu/blip2/`

- **`captioner.py`** — Defines `Blip2Captioner`: loads `Blip2ForConditionalGeneration` and `Blip2Processor` (e.g. `Salesforce/blip2-opt-2.7b`), with optional 8-bit quantization on CUDA. Exposes `_generate_caption` and `generate_captions` for single and batch image captioning.

---

### 04 — LLaVA (`04_llava.ipynb`)

**What it covers:** Large Language and Vision Assistant (LLaVA): a vision–language model that connects a visual encoder to an LLM via a projection layer and is trained for **instruction-following** and detailed image description. The notebook compares captions from BLIP, BLIP-2, and LLaVA (e.g. “Describe the image in detail”) and shows LLaVA’s richer, conversational outputs.

**Underlying code:** `src/svlearn_vlu/llava/`

- **`captioner.py`** — Defines `LlavaCaptioner`: loads `LlavaForConditionalGeneration` and `AutoProcessor` (e.g. `llava-hf/llava-1.5-7b-hf`), with optional 4-bit quantization on CUDA and `device_map="auto"`. Uses a conversation-style prompt (user message with image + “Describe the image in detail”) and returns only the assistant’s reply.
- **`chat_utils.py`** — `ChatHelper.chat_format`: formats raw model output (USER:/ASSISTANT: style) for display.

**Shared:** `src/svlearn_vlu/utils/display_captions_table.py` — `CaptionDisplay`: loads captions from JSON and displays image–caption tables (used for comparing BLIP, BLIP-2, LLaVA, etc.).

---

### 05 — SigLIP-2 (`05_siglip2.ipynb`)

**What it covers:** SigLIP-2 as a contrastive vision–language model (similar in spirit to CLIP): image and text encoders with a shared embedding space. The notebook demonstrates **embedding a set of images** into a vector store and **text-to-image retrieval** (query by text, get top-k similar images).

**Underlying code:** `src/svlearn_vlu/siglip2/`

- **`embed_images.py`** — Defines `SIGLIP2Encoder`: loads the SigLIP-2 model and processor from config (e.g. `google/siglip2-base-patch16-224`), connects to Qdrant, creates a collection (e.g. `tree_siglip2_embeddings`), and embeds images from a directory into the collection.
- **`query_images.py`** — Encodes a text query with SigLIP-2 (normalized embedding) and runs `search_similar_images` against the Qdrant collection to return the top-k images.

---

### 06 — Face detection and recognition (`06_view_faces.ipynb`)

**What it covers:** The pipeline from **face detection** to **face recognition** using embeddings: DeepFace (embedding-based recognition), RetinaFace (detection and landmarks), and **InsightFace** as a unified toolkit (e.g. buffalo_l: RetinaFace/SCRFD + ArcFace). The notebook explains how reference faces are extracted from an image folder, embedded, and stored in Qdrant, then shows how to **view** the extracted reference faces and the original images.

**Underlying code:** `src/svlearn_vlu/face_detection/`

- **`detect_and_embed.py`** — Uses InsightFace `FaceAnalysis` (e.g. `buffalo_l`: detector + ArcFace embedder). `get_face_embeddings(image_path)` returns a list of dicts (bbox, landmarks, 512-D embedding, detection score). `find_or_insert_face` and related helpers upsert face embeddings into Qdrant with metadata (image path, bbox), supporting deduplication and threshold-based matching. Assumes Qdrant is already running (config: host/port).
- **`collection_create.py`** — Creates the Qdrant collection for face embeddings (e.g. `faces_collection` from config), with 512-dimensional vectors and cosine distance.

---

### 07 — Hosted VL API demo (`07_vllm_vl_call.ipynb`)

**What it covers:** Using a **hosted** vision–language model (e.g. Qwen VL served via vLLM) through standard APIs. The notebook shows how to call the model for **image understanding** and **high-quality transcriptions/descriptions** using:

- **OpenAI-compatible API** — `openai` client pointing at the vLLM server (e.g. `Qwen/Qwen2.5-VL-72B-Instruct`).
- **LiteLLM** — Same model or endpoint abstracted through LiteLLM for a unified interface.

There is no local library in `src/svlearn_vlu` for this notebook; it relies on the hosted endpoint and the OpenAI/LiteLLM clients.

---

## Source code layout (summary)

| Notebook | Source |
|----------|--------|
| `00_vit_attention.ipynb` | `src/svlearn_vlu/vit_attention_maps.py` |
| `01_clip.ipynb` | `src/svlearn_vlu/clip/` (`embed_images.py`, `query_images.py`) |
| `02_blip.ipynb` | `src/svlearn_vlu/blip/captioner.py` |
| `03_blip-2.ipynb` | `src/svlearn_vlu/blip2/captioner.py` |
| `04_llava.ipynb` | `src/svlearn_vlu/llava/captioner.py`, `chat_utils.py`; `utils/display_captions_table.py` |
| `05_siglip2.ipynb` | `src/svlearn_vlu/siglip2/` (`embed_images.py`, `query_images.py`) |
| `06_view_faces.ipynb` | `src/svlearn_vlu/face_detection/` (`detect_and_embed.py`, `collection_create.py`) |
| `07_vllm_vl_call.ipynb` | Hosted API only (OpenAI client, LiteLLM) |

Configuration (model names, collection names, dataset paths, vector DB host/port) is centralized in **`config.yaml`** at the project root.
