import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from datasets.adult import AdultDataset
from models.mlp import MLP
from models.cnn_tabular import TabularCNN


# -----------------------
# Config (simple for now)
# -----------------------
DATASET = "adult"        # only adult in this script
ARCH = "cnn"             # "mlp" or "cnn"
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5

DROPOUT = 0.3
HIDDEN_DIM = 128  # used only by MLP if you later choose to extend it


def accuracy_from_lists(preds, labels) -> float:
    correct = sum(int(p == l) for p, l in zip(preds, labels))
    return correct / max(len(labels), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        pred = torch.argmax(logits, dim=1)

        preds.extend(pred.cpu().tolist())
        labels.extend(y.cpu().tolist())

    acc = accuracy_from_lists(preds, labels)
    f1 = f1_score(labels, preds)
    return acc, f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if DATASET != "adult":
        raise ValueError("This train.py currently supports only the Adult dataset.")

    # -----------------------
    # Data
    # -----------------------
    train_dataset = AdultDataset(split="train")
    val_dataset = AdultDataset(split="val")
    test_dataset = AdultDataset(split="test")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = train_dataset.X.shape[1]

    # -----------------------
    # Model
    # -----------------------
    if ARCH == "mlp":
        model = MLP(input_dim=input_dim, num_classes=2).to(device)
    elif ARCH == "cnn":
        model = TabularCNN(input_dim=input_dim, num_classes=2, dropout=DROPOUT).to(device)
    else:
        raise ValueError('ARCH must be "mlp" or "cnn"')

    # -----------------------
    # Train setup
    # -----------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = -1.0
    best_state = None
    epochs_no_improve = 0

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)

        val_acc, val_f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d}: Loss={avg_loss:.4f} | Val Acc={val_acc:.4f} | Val F1={val_f1:.4f}")

        # Early stopping on validation accuracy
        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered (no val acc improvement for {EARLY_STOPPING_PATIENCE} epochs).")
            break

    train_time = time.time() - start_time

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, test_f1 = evaluate(model, test_loader, device)

    print("\nFINAL TEST RESULTS")
    print(f"Dataset: {DATASET} | Architecture: {ARCH}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Training Time (seconds): {train_time:.1f}")


if __name__ == "__main__":
    main()

