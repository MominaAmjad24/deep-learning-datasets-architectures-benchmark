import time
import torch
import torch.nn as nn
from tqdm import tqdm

from datasets.cifar import get_cifar100_loaders
from models.mlp import MLP
from models.cnn_image import CIFAR100CNN


# -----------------------
# Config (CIFAR-100)
# -----------------------
ARCH = "cnn"          # "mlp" or "cnn"
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
DROPOUT = 0.3

NUM_CLASSES = 100

@torch.no_grad()
def evaluate_accuracy(model, loader, device, arch):
    model.eval()
    correct = 0
    total = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        # Flatten only for MLP
        if arch == "mlp":
            X = X.view(X.size(0), -1)

        logits = model(X)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return correct / max(total, 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_cifar100_loaders(batch_size=BATCH_SIZE)

    # Model
    if ARCH == "mlp":
        # CIFAR images are 3x32x32 = 3072 features after flatten
        model = MLP(input_dim=3 * 32 * 32, num_classes=NUM_CLASSES).to(device)
    elif ARCH == "cnn":
        model = CIFAR100CNN(num_classes=NUM_CLASSES, dropout=DROPOUT).to(device)
    else:
        raise ValueError('ARCH must be "mlp" or "cnn"')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = -1.0
    best_state = None
    epochs_no_improve = 0

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for X, y in tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False):
            X, y = X.to(device), y.to(device)

            # Flatten only for MLP
            if ARCH == "mlp":
                X = X.view(X.size(0), -1)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)

        val_acc = evaluate_accuracy(model, val_loader, device, ARCH)
        print(f"Epoch {epoch:02d}: Loss={avg_loss:.4f} | Val Acc={val_acc:.4f}")

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

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc = evaluate_accuracy(model, test_loader, device, ARCH)

    print("\nFINAL TEST RESULTS")
    print(f"Dataset: cifar100 | Architecture: {ARCH}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Training Time (seconds): {train_time:.1f}")


if __name__ == "__main__":
    main()

