#17M params 

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import v2


class CifarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))


def run_epoch(model, loader, criterion, optimizer=None, *, device="cpu"):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_correct, total_n = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            loss = criterion(logits, yb)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            n = yb.size(0)
            total_loss += loss.item() * n
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_n += n

    return total_loss / total_n, total_correct / total_n


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    epochs = 100
    patience = 20
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    train_tfs = v2.Compose([
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])
    valid_tfs = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])

    data_root = Path(__file__).resolve().parent / "cifar-10"
    train_source = datasets.CIFAR10(
        root=data_root, train=True, transform=train_tfs, download=True
    )
    valid_source = datasets.CIFAR10(
        root=data_root, train=True, transform=valid_tfs, download=True
    )

    n = len(train_source)
    n_train = int(0.9 * n)
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(n, generator=generator).tolist()
    train_dataset = Subset(train_source, indices[:n_train])
    valid_dataset = Subset(valid_source, indices[n_train:])

    num_workers = 2
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_options)

    model = CifarModel().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_state = None
    best_loss = float("inf")
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, optimizer, device=device
        )
        valid_loss, valid_accuracy = run_epoch(
            model, valid_loader, criterion, device=device
        )
        scheduler.step()

        if valid_loss < best_loss:
            best_loss = valid_loss
            bad_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            bad_epochs += 1

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train loss {train_loss:.4f}, acc {train_accuracy:.2%} | "
            f"valid loss {valid_loss:.4f}, acc {valid_accuracy:.2%} | "
            f"lr {scheduler.get_last_lr()[0]:.5f}"
        )

        if bad_epochs >= patience:
            print(f"Early stopping after epoch {epoch}.")
            break

    if best_state is not None:
        checkpoint_path = Path(__file__).resolve().parent / "model" / "cifar10_weights.pth"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, checkpoint_path)
        print(f"Best validation loss: {best_loss:.4f}")
        print(f"Weights saved to {checkpoint_path}")


if __name__ == "__main__":
    main()

