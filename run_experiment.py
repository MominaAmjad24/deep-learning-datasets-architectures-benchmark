# run_experiment.py
from __future__ import annotations

import json
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from utils.config import load_config

# Datasets
from datasets.adult import AdultDataset
from datasets.cifar import get_cifar100_loaders
from datasets.pcam import get_pcam_loaders

# Models
from models.mlp import MLP
from models.cnn_tabular import TabularCNN
from models.cnn_image import CIFAR100CNN
from models.cnn_pcam import PCamCNN
from models.vit import VisionTransformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_pref: str) -> torch.device:
    if device_pref == "cpu":
        return torch.device("cpu")
    if device_pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_pref == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def eval_epoch(model, loader, device, arch: str, binary: bool):
    model.eval()
    preds, labels = [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        # flatten only for MLP on images/tabular already flat
        if arch == "mlp" and X.ndim == 4:
            X = X.view(X.size(0), -1)

        logits = model(X)
        p = torch.argmax(logits, dim=1)

        preds.extend(p.detach().cpu().tolist())
        labels.extend(y.detach().cpu().tolist())

    acc = sum(int(a == b) for a, b in zip(preds, labels)) / max(len(labels), 1)
    f1 = f1_score(labels, preds) if binary else None
    return acc, f1


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    cfg = load_config("configs/config.yaml")
    exp = cfg["experiment"]
    trn = cfg["training"]

    set_seed(int(exp.get("seed", 42)))
    device = choose_device(exp.get("device", "auto"))

    dataset = exp["dataset"].lower()
    arch = exp["architecture"].lower()

    out_dir = Path(exp.get("output_dir", "runs")) / f"{dataset}_{arch}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Dataset: {dataset} | Architecture: {arch}")
    print(f"Logs: {out_dir}")

    # -------------------------
    # Build loaders + model
    # -------------------------
    binary = False

    if dataset == "adult":
        binary = True
        train_ds = AdultDataset(split="train")
        val_ds = AdultDataset(split="val")
        test_ds = AdultDataset(split="test")

        bs = int(trn["batch_size"])
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=int(trn["num_workers"]))
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=int(trn["num_workers"]))
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=int(trn["num_workers"]))

        input_dim = train_ds.X.shape[1]
        num_classes = int(cfg["adult"]["num_classes"])

        if arch == "mlp":
            model = MLP(input_dim=input_dim, num_classes=num_classes).to(device)
        elif arch == "cnn":
            model = TabularCNN(input_dim=input_dim, num_classes=num_classes, dropout=float(trn["dropout"])).to(device)
        else:
            raise ValueError("For adult, architecture must be mlp or cnn.")

    elif dataset == "cifar100":
        num_classes = int(cfg["cifar100"]["num_classes"])
        val_ratio = float(cfg["cifar100"].get("val_ratio", 0.1))

        train_loader, val_loader, test_loader = get_cifar100_loaders(
            batch_size=int(trn["batch_size"]),
            num_workers=int(trn["num_workers"]),
            val_ratio=val_ratio,
        )

        if arch == "mlp":
            model = MLP(input_dim=3 * 32 * 32, num_classes=num_classes).to(device)
        elif arch == "cnn":
            model = CIFAR100CNN(num_classes=num_classes, dropout=float(trn["dropout"])).to(device)
        elif arch == "vit":
            vitcfg = cfg["cifar100"]["vit"]
            model = VisionTransformer(
                img_size=int(vitcfg["img_size"]),
                patch_size=int(vitcfg["patch_size"]),
                num_classes=num_classes,
                embed_dim=int(vitcfg["embed_dim"]),
                depth=int(vitcfg["depth"]),
                num_heads=int(vitcfg["num_heads"]),
                mlp_ratio=float(vitcfg["mlp_ratio"]),
                dropout=float(vitcfg["dropout"]),
            ).to(device)
        else:
            raise ValueError("For cifar100, architecture must be mlp, cnn, or vit.")

    elif dataset == "pcam":
        binary = True
        num_classes = int(cfg["pcam"]["num_classes"])
        subset = cfg["pcam"]["subset"]

        train_loader, val_loader, test_loader = get_pcam_loaders(
            batch_size=int(trn["batch_size"]),
            num_workers=int(trn["num_workers"]),
            max_train=subset.get("max_train"),
            max_val=subset.get("max_val"),
            max_test=subset.get("max_test"),
        )

        if arch == "mlp":
            model = MLP(input_dim=3 * 96 * 96, num_classes=num_classes).to(device)
        elif arch == "cnn":
            model = PCamCNN(num_classes=num_classes, dropout=float(trn["dropout"])).to(device)
        else:
            raise ValueError("For pcam, architecture must be mlp or cnn.")

    else:
        raise ValueError("dataset must be one of: adult, cifar100, pcam")

    # -------------------------
    # Training setup
    # -------------------------
    lr = float(trn["lr"])
    epochs = int(trn["epochs"])
    patience = int(trn["early_stopping_patience"])

    # Override for ViT (common)
    if dataset == "cifar100" and arch == "vit":
        vitcfg = cfg["cifar100"]["vit"]
        lr = float(vitcfg.get("lr", lr))
        epochs = int(vitcfg.get("epochs", epochs))

    print(f"Train params: lr={lr}, epochs={epochs}, batch={trn['batch_size']}")
    print(f"Parameter count: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = -1.0
    best_state = None
    no_improve = 0
    history = {"train_loss": [], "val_acc": [], "val_f1": []}

    start = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            if arch == "mlp" and X.ndim == 4:
                X = X.view(X.size(0), -1)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        val_acc, val_f1 = eval_epoch(model, val_loader, device, arch, binary)

        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if binary:
            print(f"Epoch {ep:02d}: Loss={avg_loss:.4f} | Val Acc={val_acc:.4f} | Val F1={val_f1:.4f}")
        else:
            print(f"Epoch {ep:02d}: Loss={avg_loss:.4f} | Val Acc={val_acc:.4f}")

        if val_acc > best_val + 1e-6:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping triggered (no val acc improvement for {patience} epochs).")
            break

    train_time = time.time() - start

    if best_state is not None and exp.get("save_best", True):
        model.load_state_dict(best_state)

    test_acc, test_f1 = eval_epoch(model, test_loader, device, arch, binary)

    result = {
        "dataset": dataset,
        "architecture": arch,
        "device": str(device),
        "test_accuracy": float(test_acc),
        "test_f1": None if test_f1 is None else float(test_f1),
        "training_time_seconds": float(train_time),
        "param_count": int(count_parameters(model)),
        "history": history,
        "config": cfg,
    }

    # Save artifacts
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if exp.get("save_best", True) and best_state is not None:
        torch.save(model.state_dict(), out_dir / "best_model.pth")

    print("\nFINAL TEST RESULTS")
    print(f"Dataset: {dataset} | Architecture: {arch}")
    print(f"Test Accuracy: {test_acc:.4f}")
    if binary:
        print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Training Time (seconds): {train_time:.1f}")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()

