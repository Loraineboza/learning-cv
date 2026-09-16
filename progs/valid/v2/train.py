import torch
import torch.utils.data as data
import numpy as np 
import os 
import random

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(3, 1))
        self.bias = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        return x @ self.w + self.bias

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x: torch.tensor, true: torch.tensor):
        self.x = x
        self.true = true
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx], self.true[idx]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_epoch(
        model, loader, loss_func, optimizer=None
):
    is_train = optimizer is not None
    model.train(is_train) #is_train=True => model.train(); is_train=False => model.eval()
    
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    total_n, total_loss = 0, 0.0

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
    set_seed(42)
    
    model = MyModel()
    
    n = 10_000
     
    x1, x2, x3 = torch.randn(n), torch.randn(n), torch.randn(n)

    x = torch.stack([x1, x2, x3], dim=1)

    epsilion = torch.randn(n) * 0.1
    y = (
        3 * torch.sin(x1)
        + 0.5 * x2 ** 2 
        - 2 * x3 
        + 0.7 * x1 * x2 
        + epsilion
    ).unsqueeze(1)

        
    m_dataset = MyDataset(x, y)

    train_n = int(0.7 * len(m_dataset))   
    valid_n = int(0.15 * len(m_dataset))
    test_n = len(m_dataset) - train_n - valid_n 

    g = torch.Generator().manual_seed(42)
    train_dataset, valid_dataset, test_loader = torch.utils.data.random_split(m_dataset, [train_n, valid_n, test_n], generator=g)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, num_workers=0, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_loader, num_workers=0, batch_size=32, shuffle=False)
    
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=5e-2)
    
    bad_epoch, best_loss, patience, best_state = 0, float("inf"), 15, model.state_dict()
    epochs = 10_000
        
    for e in range(epochs):
        train_loss = run_epoch(model, train_loader, loss_func, optimizer)
        valid_loss = run_epoch(model, valid_loader, loss_func, None)

        if valid_loss < best_loss:
            best_loss = valid_loss
            bad_epoch = 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            bad_epoch += 1
            if bad_epoch >= patience:
                print("early stopping to epoch =", e)
                break
        
        if e % max(1, epochs//30) == 0:
            print(f"{e}: train loss = {train_loss}; valid loss = {valid_loss}; best loss = {best_loss}")

    os.makedirs("model", exist_ok=True) 
    torch.save(best_state, "model/model300.pth")
    model.load_state_dict(best_state)

    print("\ntest:")
    test_loss = run_epoch(model, test_loader, loss_func, None)
    print(f"best valid loss = {best_loss}; test loss = {test_loss}")


if __name__ == "__main__":
    main()
