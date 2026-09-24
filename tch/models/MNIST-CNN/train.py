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
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            n = yb.numel()
            total_loss += loss.item() * n 
            total_n += n 

    return total_loss / total_n 

def main():
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
                                               num_workers=0, 
                                               batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               num_workers=0,
                                               batch_size=32, shuffle=False)
    
    criterion = nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-2)
    
    best_state, best_valid_loss, bad_epoch = model.state_dict(), float("inf"), 0
    patience = 15 

    total_train = 30
    epochs = total_train + patience

    print("\ntrain/valid:")
    for e in range(epochs):
        is_checkpoint_model = "model is saved"

        train_loss = run_epoch(model, train_loader, criterion, optimizer, device=device)
        valid_loss = run_epoch(model, valid_loader, criterion, None, device=device)

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
            is_checkpoint_model = "model isn't saved"
        
        print(
            f"({e}):\tbest loss = {best_valid_loss:.10f}\n"
            f"\ttrain loss = {train_loss:.10f}\n"
            f"\tbad epoch = {bad_epoch}\t({is_checkpoint_model})\n"
        )

    path_model = "model/mnist_cnn_model2.pth"
    os.makedirs("model", exist_ok=True)
    model.load_state_dict(best_state)
    torch.save(best_state, path_model)
    print(f"\nModel saved to path: \"{path_model}\"")
    
if __name__ == "__main__":
    main()
