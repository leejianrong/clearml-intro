"""
train.py — Baseline CNN training script for Fashion-MNIST
========================================================
Run from the project root (e.g. `python src/train.py --epochs 20`).

This script
1. Downloads Fashion-MNIST to <project-root>/data/ (unless already present)
2. Splits the original 60,000-image training set into Train (55,000) & Validation (5,000)
3. Trains a simple CNN, saving the best-validation checkpoint to outputs/checkpoints/best_model.pth

(Optionally integrate ClearML by uncommenting the two marked lines.)
"""
import os, random, argparse, numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Optional experiment-tracker — uncomment if using ClearML
from clearml import Task

task = Task.init(project_name="FashionMNIST-CNN", task_name="baseline-training")

def seed_everything(seed: int = 42):
    """Reproducibility helper."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class CNNFashion(nn.Module):
    """A light(~1.2M params) CNN for 28x28 grayscale images."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Final spatial size: 3×3
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def dataloaders(data_dir: str, batch_size: int, val_ratio: float = 0.0833):
    """Return Train, Val, Test loaders (val_ratio 5k/60k)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    full_train = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_set   = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    val_size   = int(len(full_train) * val_ratio)   # 5 000
    train_size = len(full_train) - val_size         # 55 000
    train_set, val_set = random_split(full_train, [train_size, val_size])

    kw = dict(num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, **kw)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, **kw)
    return train_loader, val_loader, test_loader

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct    += logits.argmax(1).eq(y).sum().item()
        total      += y.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        correct    += logits.argmax(1).eq(y).sum().item()
        total      += y.size(0)
    return total_loss / total, correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",        type=str,   default="data")
    parser.add_argument("--batch_size",      type=int,   default=64)
    parser.add_argument("--epochs",          type=int,   default=20)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--checkpoint_dir",  type=str,   default="outputs/checkpoints")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    seed_everything()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = dataloaders(args.data_dir, args.batch_size)

    model = CNNFashion().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} train_acc={tr_acc*100:.2f}% "
              f"| val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")

        # Checkpoint best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)

    # Final evaluation on test set
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test accuracy: {test_acc*100:.2f}% (loss={test_loss:.4f})")

if __name__ == "__main__":
    main()
