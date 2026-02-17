import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from datasets.adult import AdultDataset
from models.mlp import MLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
train_dataset = AdultDataset(split="train")
val_dataset = AdultDataset(split="val")
test_dataset = AdultDataset(split="test")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

input_dim = train_dataset.X.shape[1]
model = MLP(input_dim).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 15

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Validation
    model.eval()
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, preds = torch.max(outputs, 1)

            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(y.cpu().numpy())

    val_acc = sum([p == l for p, l in zip(val_preds, val_labels)]) / len(val_labels)
    val_f1 = f1_score(val_labels, val_preds)

    print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f} | Val Acc={val_acc:.4f} | Val F1={val_f1:.4f}")

# Test evaluation
model.eval()
test_preds = []
test_labels = []

with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        _, preds = torch.max(outputs, 1)

        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(y.cpu().numpy())

test_acc = sum([p == l for p, l in zip(test_preds, test_labels)]) / len(test_labels)
test_f1 = f1_score(test_labels, test_preds)

print("\nFINAL TEST RESULTS")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")

