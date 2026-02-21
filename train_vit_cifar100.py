import time
import torch
import torch.nn as nn
from tqdm import tqdm

from datasets.cifar import get_cifar100_loaders
from models.vit import VisionTransformer


BATCH_SIZE = 128
LR = 3e-4
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return correct / max(total, 1)


def main():
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_cifar100_loaders(
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    model = VisionTransformer(
        img_size=32,
        patch_size=4,
        num_classes=100,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = -1
    best_state = None
    patience = 0

    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for X, y in tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d}: Loss={total_loss/len(train_loader):.4f} | Val Acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if patience >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    train_time = time.time() - start

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc = evaluate(model, test_loader, device)

    print("\nFINAL TEST RESULTS")
    print(f"Dataset: cifar100 | Architecture: vit")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Training Time (seconds): {train_time:.1f}")


if __name__ == "__main__":
    main()

