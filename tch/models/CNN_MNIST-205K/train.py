import torch 
import torch.nn as nn
from torchvision import datasets, transforms
import os

class CnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_linear_split = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        ) 
    def forward(self, x):
        logits = self.conv2d_linear_split(x)
        return logits

def run_epoch(model, loader, criterion, optimizer=None, *, device="cpu"):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_n = 0.0, 0
    total_correct = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for xb, yb in loader:
            xb = xb.to(device=device, non_blocking=True)
            yb = yb.to(device=device, non_blocking=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            n = yb.size(0)
            total_loss += loss.item() * n 
            total_n += n 
            
            pred = logits.argmax(dim=1)
            total_correct += (pred == yb).sum().item()

    return (total_loss / total_n), (total_correct / total_n) 

def main():
    best_state, best_valid_loss, bad_epoch = None, float("inf"), 0
    patience = 30 
    epochs = 30 + patience
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"powered by {device}\n")

    model = CnnModel().to(device)

    print(f"the structure of the model:", end='\n\n')
    print(model)

    tfs = transforms.ToTensor()
    train_valid_dataset = datasets.MNIST(
        root="data", train=True, transform=tfs, target_transform=None, download=True
    )
    
    n = len(train_valid_dataset)
    n_train = int(0.8 * n)
    n_valid = n - n_train
    
    train_dataset, valid_dataset = torch.utils.data.random_split(train_valid_dataset, 
                                                                 [n_train, n_valid])
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               num_workers=2, pin_memory=True,
                                               batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, 
                                               num_workers=2, pin_memory=True,
                                               batch_size=32, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=1e-7)


    print("\ntrain/valid:")
    for e in range(epochs):
        is_save = "model is saved"

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device=device)
        valid_loss, valid_acc = run_epoch(model, valid_loader, criterion, device=device)

        scheduler.step()
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            bad_epoch = 0 
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            
        else:
            bad_epoch += 1
            if bad_epoch >= patience:
                print(f"Early stopping: bad_epoch({bad_epoch}) == patience({patience})")
                print(f"epoch = {e}")
                break 
            is_save = "model isn't saved"
        
        print(
            f"Epoch {e:2d}/{epochs}\n"
            f"train loss => {train_loss1:.10f} | train accuracy => {train_acc:2%}\n"
            f"valid loss => {valid_loss:.10f} | valid accuracy => {valid_acc:%2}\n"
            f"{is_save}: {bad_epoch}/{patience}\n"
            f"LR => {scheduler.get_last_lr()[0]:.10f}\n"
                )
    path_model = "model/mnist_cnn_model2.pth"
    os.makedirs("model", exist_ok=True)
    model.load_state_dict(best_state)
    torch.save(best_state, path_model)
    print(f"\nModel saved to path: \"{path_model}\"")
    
if __name__ == "__main__":
    main()
