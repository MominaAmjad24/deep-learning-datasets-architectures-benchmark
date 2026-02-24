# analysis/visualize_weights.py
from __future__ import annotations
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from models.cnn_image import CIFAR100CNN
from models.cnn_pcam import PCamCNN
from models.vit import VisionTransformer


def save_conv1_filters(conv_weight: torch.Tensor, out_path: Path, title: str):
    """
    conv_weight: (out_channels, in_channels, kH, kW)
    We'll visualize the first N filters as small images.
    """
    w = conv_weight.detach().cpu()
    n = min(w.size(0), 16)  # show up to 16 filters
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(title)

    for i in range(n):
        ax = fig.add_subplot(4, 4, i + 1)
        # Take RGB channels of filter i and normalize for display
        f = w[i]
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        f = f.permute(1, 2, 0)  # (kH, kW, inC)
        ax.imshow(f)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()
    print(f"Saved: {out_path}")


def main():
    runs = Path("runs")

    # Example: visualize CIFAR CNN conv1
    cifar_ckpt = runs / "cifar100_cnn" / "best_model.pth"
    if cifar_ckpt.exists():
        model = CIFAR100CNN(num_classes=100)
        model.load_state_dict(torch.load(cifar_ckpt, map_location="cpu"))
        conv1 = model.features[0].weight
        save_conv1_filters(conv1, runs / "cifar100_cnn_conv1.png", "CIFAR-100 CNN: Conv1 Filters")

    # Example: visualize PCam CNN conv1
    pcam_ckpt = runs / "pcam_cnn" / "best_model.pth"
    if pcam_ckpt.exists():
        model = PCamCNN(num_classes=2)
        model.load_state_dict(torch.load(pcam_ckpt, map_location="cpu"))
        conv1 = model.features[0].weight
        save_conv1_filters(conv1, runs / "pcam_cnn_conv1.png", "PCam CNN: Conv1 Filters")

    # Example: visualize ViT patch embedding (it's a Conv2d)
    vit_ckpt = runs / "cifar100_vit" / "best_model.pth"
    if vit_ckpt.exists():
        model = VisionTransformer(img_size=32, patch_size=4, num_classes=100)
        model.load_state_dict(torch.load(vit_ckpt, map_location="cpu"))
        conv = model.patch_embed.proj.weight  # (E, 3, P, P)
        save_conv1_filters(conv, runs / "cifar100_vit_patch_embed.png", "ViT: Patch Embedding Filters")

    print("Done.")


if __name__ == "__main__":
    main()

