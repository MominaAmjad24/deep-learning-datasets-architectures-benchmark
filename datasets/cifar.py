import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms


def get_cifar100_loaders(batch_size=128, num_workers=2, val_ratio=0.1):
    """
    Returns train/val/test DataLoaders for CIFAR-100.
    Uses torchvision built-in dataset + a random split for validation.
    """
    # Standard CIFAR normalization
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761)
        )
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761)
        )
    ])

    train_full = datasets.CIFAR100(root="data", train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR100(root="data", train=False, download=True, transform=transform_test)

    val_size = int(len(train_full) * val_ratio)
    train_size = len(train_full) - val_size
    train_set, val_set = random_split(train_full, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # Important: validation should not use random augmentation (crop/flip)
    # So we overwrite val transform to test-style transform
    val_set.dataset.transform = transform_test

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
    )

    return train_loader, val_loader, test_loader

