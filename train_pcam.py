import time
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from datasets.pcam import get_pcam_loaders
from models.mlp import MLP
from models.cnn_pcam import PCamCNN


# -----------------------
# Config (PCam)
# -----------------------
ARCH = "mlp"          # "mlp" or "cnn"
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 15
EARLY_STOPPING_PATIENCE = 4
DROPOUT = 0.3

# Subset mode (recommended on laptop):
MAX_TRAIN = 50000    # set to None for full train (huge)
MAX_VAL   = 10000
MAX_TEST  = 10000

NUM_CLASSES = 2


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )


@torch.no_grad()
def evaluate(model, loader, device, arch):
    model.eval()
    preds, labels = [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        if arch == "mlp":
            X = X.view(X.size(0), -1)

        logits = model(X)
        p = torch.argmax(logits, dim=1)

        preds.extend(p.cpu().tolist())
        labels.extend(y.cpu().tolist())

    acc = sum(int(p == l) for p, l in zip(preds, labels)) / max(len(labels), 1)
    f1 = f1_score(labels, preds)
    return acc, f1


def main():
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_pcam_loaders(
        batch_size=BATCH_SIZE,
        num_workers=0,
        max_train=MAX_TRAIN,
        max_val=MAX_VAL,
        max_test=MAX_TEST
    )

    if ARCH == "mlp":
        model = MLP(input_dim=3 * 96 * 96, num_classes=NUM_CLASSES).to(device)
    elif ARCH == "cnn":
        model = PCamCNN(num_classes=NUM_CLASSES, dropout=DROPOUT).to(device)
    else:
        raise ValueError('ARCH must be "mlp" or "cnn"')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = -1.0
    best_state = None
    epochs_no_improve = 0

    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for X, y in tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False):
            X, y = X.to(device), y.to(device)

            if ARCH == "mlp":
                X = X.view(X.size(0), -1)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        val_acc, val_f1 = evaluate(model, val_loader, device, ARCH)

        print(f"Epoch {epoch:02d}: Loss={avg_loss:.4f} | Val Acc={val_acc:.4f} | Val F1={val_f1:.4f}")

        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered (no val acc improvement for {EARLY_STOPPING_PATIENCE} epochs).")
            break

    train_time = time.time() - start

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, test_f1 = evaluate(model, test_loader, device, ARCH)

    print("\nFINAL TEST RESULTS")
    print(f"Dataset: pcam | Architecture: {ARCH}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Training Time (seconds): {train_time:.1f}")
    print(f"Subset sizes: train={MAX_TRAIN}, val={MAX_VAL}, test={MAX_TEST}")


if __name__ == "__main__":
    main()

