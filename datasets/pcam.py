import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_pcam_loaders(batch_size=128, num_workers=0, max_train=None, max_val=None, max_test=None):
    """
    PCam (PatchCamelyon) loaders using torchvision.datasets.PCAM.
    Splits are predefined: 'train', 'val', 'test'.

    max_* let you train on a subset for speed (recommended on laptop).
    """
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    # PCAM images are 96x96 RGB, labels are 0/1
    train_set = datasets.PCAM(root="data", split="train", download=True, transform=transform_train)
    val_set   = datasets.PCAM(root="data", split="val", download=True, transform=transform_eval)
    test_set  = datasets.PCAM(root="data", split="test", download=True, transform=transform_eval)

    # Optional subsets for speed
    if max_train is not None:
        train_set = Subset(train_set, list(range(min(max_train, len(train_set)))))
    if max_val is not None:
        val_set = Subset(val_set, list(range(min(max_val, len(val_set)))))
    if max_test is not None:
        test_set = Subset(test_set, list(range(min(max_test, len(test_set)))))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

