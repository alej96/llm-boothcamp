#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------
"""
Fine-tune SigLIP2 on a custom image-caption dataset.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModel, 
    AutoProcessor,
    get_cosine_schedule_with_warmup,
    AdamW,
)

# -------------------------------
# 1. Dataset
# -------------------------------
class ImageCaptionDataset(Dataset):
    def __init__(self, csv_path, image_folder, processor):
        self.df = pd.read_csv(csv_path)
        self.image_folder = image_folder
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_folder, row["image_path"])
        caption = row["caption"]

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=caption, images=image, padding="max_length", max_length=64, return_tensors="pt")
        # Each field has shape [1, ...] — remove batch dimension
        return {k: v.squeeze(0) for k, v in inputs.items()}


# -------------------------------
# 2. Training setup
# -------------------------------
def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        img_embeds = outputs.image_embeds
        txt_embeds = outputs.text_embeds

        # Normalize
        img_embeds = img_embeds / img_embeds.norm(dim=1, keepdim=True)
        txt_embeds = txt_embeds / txt_embeds.norm(dim=1, keepdim=True)

        logits = txt_embeds @ img_embeds.t()
        labels = torch.arange(len(logits)).to(device)

        loss_i2t = torch.nn.functional.cross_entropy(logits, labels)
        loss_t2i = torch.nn.functional.cross_entropy(logits.t(), labels)
        loss = (loss_i2t + loss_t2i) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    # -------------------------------
    # Configuration
    # -------------------------------
    model_name = "google/siglip2-so400m-patch14-384"
    dataset_csv = "dataset/captions.csv"
    image_folder = "dataset/images"
    output_dir = "finetuned_siglip2"
    batch_size = 8
    num_epochs = 5
    lr = 5e-6
    warmup_ratio = 0.05

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("mps") if torch.backends.mps.is_available() else device

    # -------------------------------
    # Load model and processor
    # -------------------------------
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)

    # -------------------------------
    # Dataset and DataLoader
    # -------------------------------
    ds = ImageCaptionDataset(dataset_csv, image_folder, processor)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)

    # -------------------------------
    # Optimizer and scheduler
    # -------------------------------
    num_training_steps = len(dataloader) * num_epochs
    num_warmup_steps = int(num_training_steps * warmup_ratio)

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    # -------------------------------
    # Training loop
    # -------------------------------
    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(model, dataloader, optimizer, scheduler, device)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")

        # Save checkpoint
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(os.path.join(output_dir, f"epoch_{epoch+1}"))
        processor.save_pretrained(output_dir)

    print("✅ Fine-tuning complete! Model saved in", output_dir)


if __name__ == "__main__":
    main()
