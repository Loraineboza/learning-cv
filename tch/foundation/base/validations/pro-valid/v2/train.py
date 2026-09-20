import os
import torch
import torch.utils.data as data

class MyModel(torch.nn.Module):
    def __init__(self, in_features=3, out_features=1):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(in_features, out_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        return x @ self.w + self.bias

class MyDataset(data.Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def run_epoch(model, loader, loss_func, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_n = 0.0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_func(pred, yb)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * yb.numel()
            total_n += yb.numel()
    return total_loss / total_n

def main():
    torch.manual_seed(0)

    x1, x2, x3 = (torch.randn(10_000) for _ in range(3))
    x = torch.stack([x1, x2, x3], dim=1)


    eps = torch.finfo(torch.float32).eps
    y = (3 * torch.sin(x1) + 0.5 * x2**2 - 2 * x3 + 0.7 * x1 * x2 + eps
         ).unsqueeze(1)                                 

    ds = MyDataset(x, y)
    n_train = int(0.8 * len(ds))
    train_ds, val_ds = data.random_split(ds, [n_train, len(ds) - n_train])

    train_loader = data.DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = data.DataLoader(val_ds,   batch_size=16, shuffle=False)

    model = MyModel(in_features=3, out_features=1)
    mse = torch.nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    epochs, patience = 10_000, 15
    best_val, best_state_dict, bad = float("inf"), None, 0

    for epoch in range(epochs):
        train_loss = run_epoch(model, train_loader, mse, opt)
        val_loss   = run_epoch(model, val_loader,   mse)

        if val_loss < best_val:
            best_val = val_loss
            best_state_dict = {key: value.detach().clone() for key, value in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % max(1, epochs // 20) == 0:
            print(f"{epoch}: train={train_loss:.6f} val={val_loss:.6f} best={best_val:.6f}")
    model.load_state_dict(best_state_dict)
    os.makedirs("model", exist_ok=True)
    torch.save(best_state_dict, "model/model155.pth")
    print("saved best model, best_val =", best_val)

if __name__ == "__main__":
    main()
