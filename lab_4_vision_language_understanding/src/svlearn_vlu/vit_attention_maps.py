#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------

import io
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor


#  -------------------------------------------------------------------------------------------------

DEFAULT_MODEL = "google/vit-base-patch16-224"
# Number of attention heads to show in the grid (ViT-base has 12 heads)
NUM_HEADS_TO_SHOW = 12
# Which layer's attention to visualize (0 = first, -1 = last)
LAYER_INDEX = -1


def load_image(image_path: str, processor: ViTImageProcessor) -> tuple[torch.Tensor, Image.Image]:
    """Load and preprocess image for the ViT. image_path may be a local path or http(s) URL."""
    if image_path.startswith(("http://", "https://")):
        req = urllib.request.Request(
            image_path,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            image = Image.open(io.BytesIO(resp.read())).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    return inputs["pixel_values"], image


def _get_attention_maps_manual(
    model: ViTForImageClassification,
    pixel_values: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """
    Run embeddings, then encoder layers 0..layer_idx-1, then call the target layer's
    self-attention with eager backend so we get attention weights. Use when the model
    does not return attentions (e.g. ViTEncoder does not pass output_attentions, or
    SDPA does not return weights).
    """
    vit = model.vit
    with torch.no_grad():
        embedding_output = vit.embeddings(pixel_values)
        hidden_states = embedding_output
        for i in range(layer_idx):
            hidden_states = vit.encoder.layer[i](hidden_states)
        hidden_norm = vit.encoder.layer[layer_idx].layernorm_before(hidden_states)
        self_attn = vit.encoder.layer[layer_idx].attention.attention

        # Try once; SDPA/Flash may return only (context,) so no weights
        result = self_attn(hidden_norm)
        if isinstance(result, tuple) and len(result) >= 2 and result[1] is not None:
            return result[1].squeeze(0).cpu()

        print("No attentions returned; forcing eager attention...")
        # Force eager attention so we get (context, attention_probs)
        config = vit.config
        original_implementation = getattr(config, "_attn_implementation", None)
        print(f"Original implementation: {original_implementation}")
        try:
            config._attn_implementation = "eager"
            result = self_attn(hidden_norm)
            if isinstance(result, tuple) and len(result) >= 2 and result[1] is not None:
                return result[1].squeeze(0).cpu()
        finally:
            if original_implementation is not None:
                config._attn_implementation = original_implementation

    raise RuntimeError(
        "Could not obtain attention weights from the target layer (manual forward)."
    )


def get_attention_maps(
    model: ViTForImageClassification,
    pixel_values: torch.Tensor,
    layer_idx: int,
    device: torch.device,
):
    """
    Run forward pass with output_attentions=True and return attention weights
    for the specified layer. Shape: (num_heads, num_patches+1, num_patches+1).
    Uses model.vit() because ViTForImageClassification.forward() does not return attentions.
    Falls back to CPU then to a forward-hook if the model does not return attentions.
    """
    with torch.no_grad():
        outputs = model.vit(pixel_values, output_attentions=True)
    attentions = outputs.attentions
    if not attentions and device.type != "cpu":
        print("No attentions returned; retrying on CPU...")
        model.vit.to("cpu")
        with torch.no_grad():
            outputs = model.vit(pixel_values.cpu(), output_attentions=True)
        attentions = outputs.attentions
        model.vit.to(device)
    if attentions:
        return attentions[layer_idx].squeeze(0).cpu()
    # Model did not return attentions; run encoder manually and call target layer with eager attn
    print("Running encoder manually and calling target layer with eager attn...")
    if device.type != "cpu":
        model.vit.to("cpu")
        pixel_values = pixel_values.cpu()
    try:
        return _get_attention_maps_manual(model, pixel_values, layer_idx)
    finally:
        if device.type != "cpu":
            model.vit.to(device)

def plot_attention_heads(
    attention_weights: torch.Tensor,
    num_heads: int = 12,
    layer_idx: int = -1,
    save_path: str | None = None,
) -> None:
    """
    Plot the image and a grid of attention maps, one per head (from [CLS] to spatial patches).
    """
    cls_to_patches = attention_weights[:, 0, 1:]  # (num_heads, num_patches)
    num_patches = cls_to_patches.shape[1]
    side = int(num_patches**0.5)

    n_heads = min(num_heads, cls_to_patches.shape[0])
    ncols = 4
    nrows = (n_heads + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i in range(n_heads):
        ax = axes[i]
        head_map = cls_to_patches[i].reshape(side, side).numpy()
        im = ax.imshow(head_map, cmap="viridis")
        ax.set_title(f"Head {i}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, shrink=0.6)

    for j in range(n_heads, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"ViT attention from [CLS] to spatial patches (layer {layer_idx})",
        fontsize=12,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved head-wise attention map to {save_path}")
    plt.show()


def plot_attention_overlay(
    attention_weights: torch.Tensor,
    image: Image.Image,
    layer_idx: int = -1,
    save_path: str | None = None,
) -> None:
    """
    Overlay mean attention (over heads) from [CLS] onto the image to see which regions the model attends to.
    """
    cls_to_patches = attention_weights[:, 0, 1:]  # (num_heads, num_patches)
    mean_attention = cls_to_patches.mean(dim=0)  # (num_patches,)
    num_patches = mean_attention.shape[0]
    side = int(num_patches**0.5)
    attention_map = mean_attention.reshape(side, side).numpy()

    # Resize attention map to image size for overlay (using torch to avoid scipy)
    h, w = image.size[1], image.size[0]
    att = torch.from_numpy(attention_map).float().unsqueeze(0).unsqueeze(0)
    attention_resized = torch.nn.functional.interpolate(
        att, size=(h, w), mode="bilinear", align_corners=False
    ).squeeze().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Input image")
    axes[0].axis("off")

    axes[1].imshow(image)
    axes[1].imshow(attention_resized, cmap="hot", alpha=0.5)
    axes[1].set_title(f"Mean attention overlay (layer {layer_idx})")
    axes[1].axis("off")

    im = axes[2].imshow(attention_map, cmap="hot")
    axes[2].set_title(f"Mean attention only (layer {layer_idx})")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], shrink=0.7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved overlay to {save_path}")
    plt.show()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize ViT attention maps for a natural image (objects vs textures)."
    )
    parser.add_argument(
        "image_path",
        type=str,
        nargs="?",
        default="https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Cat_August_2010-4.jpg/500px-Cat_August_2010-4.jpg",
        help="Path to a natural image. If not set, tries config datasets.images or datasets.trees.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"ViT model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Layer index for attention (default: -1 = last)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to save attention visualizations",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not call plt.show(); only save if --out-dir is set",
    )
    args = parser.parse_args()

    image_path = args.image_path
    if not image_path:
        try:
            from svlearn_vlu import config as vlu_config
            images_dir = (
                vlu_config.get("datasets", {}).get("trees")
            )
            if images_dir and Path(images_dir).exists():
                first_image = next(Path(images_dir).rglob("*.[pj][np]g"), None)
                if first_image:
                    image_path = str(first_image)
        except Exception:
            pass
        if not image_path:
            print("Please provide image_path or set datasets.trees in config.yaml.")
            return

    path = Path(image_path)
    if not image_path.startswith(("http://", "https://")) and not path.exists():
        print(f"Image not found: {image_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps") if torch.backends.mps.is_available() else device
    print(f"Using device: {device}")

    print(f"Loading model {args.model}...")
    model = ViTForImageClassification.from_pretrained(args.model).to(device)
    processor = ViTImageProcessor.from_pretrained(args.model)

    print(f"Loading image: {image_path}")
    pixel_values, image = load_image(image_path, processor)
    pixel_values = pixel_values.to(device)

    layer_idx = args.layer if args.layer >= 0 else model.config.num_hidden_layers + args.layer
    print(f"Extracting attention for layer {layer_idx}...")
    attention_weights = get_attention_maps(model, pixel_values, layer_idx, device)

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem

    save_heads = str(out_dir / f"{stem}_vit_attention_heads.png") if out_dir else None
    save_overlay = str(out_dir / f"{stem}_vit_attention_overlay.png") if out_dir else None

    plot_attention_heads(
        attention_weights,
        num_heads=model.config.num_attention_heads,
        layer_idx=layer_idx,
        save_path=save_heads,
    )
    plot_attention_overlay(
        attention_weights,
        image,
        layer_idx=layer_idx,
        save_path=save_overlay,
    )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
