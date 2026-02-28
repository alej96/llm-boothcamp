#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------

from transformers import AutoModel, AutoProcessor
from PIL import Image
import requests
import torch
import numpy as np
from svlearn_vlu import config
# ----------------------------
# 1️⃣ Load model and processor
# ----------------------------
#ckpt = "google/siglip2-so400m-patch14-384"
ckpt = config["siglip2"]["model"]
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("mps") if torch.backends.mps.is_available() else device
model = AutoModel.from_pretrained(ckpt).to(device).eval()
processor = AutoProcessor.from_pretrained(ckpt)

# ----------------------------
# 2️⃣ One sample image
# ----------------------------
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

# ----------------------------
# 3️⃣ Encode image
# ----------------------------
with torch.no_grad():
    image_inputs = processor(images=image, return_tensors="pt").to(device)
    image_embeds = model.get_image_features(**image_inputs)
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

# ----------------------------
# 4️⃣ Encode two text queries
# ----------------------------
texts = ["photo of a cat", "photo of a dog"]

with torch.no_grad():
    text_inputs = processor(
        text=texts,
        padding="max_length",
        max_length=64,
        return_tensors="pt"
    ).to(device)
    text_embeds = model.get_text_features(**text_inputs)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

# ----------------------------
# 5️⃣ Compute cosine similarities
# ----------------------------
similarities = (text_embeds @ image_embeds.T).squeeze().cpu().numpy()

# ----------------------------
# 6️⃣ Display results
# ----------------------------
print(f"Comparing text queries against the image: {image_url}\n")
for t, score in zip(texts, similarities):
    print(f"Similarity for '{t}': {score:.3f}")

best_match = texts[int(np.argmax(similarities))]
print(f"\n✅ Highest similarity → '{best_match}'")
