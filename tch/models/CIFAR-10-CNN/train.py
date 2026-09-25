import torch 
import torch.nn as nn 
from torchvision import datasets, transforms 
import os 


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
     
    tfs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    train_valid_dataset = datasets.CIFAR10(
        root="cifar-10", train=True, transform=tfs, target_transform=None, download=True
    )
        
    n = len(train_valid_dataset)
    n_train = int(0.8 * n); n_valid = n - n_train 
    train_dataset, valid_dataset = torch.utils.data.random_split(train_valid_dataset, [n_train, n_valid])
    
    sz_batch = 32
    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=sz_batch, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, num_workers=0, batch_size=sz_batch, shuffle=False)
 
    c, h, w = train_dataset[0][0].shape[-3:]
    print("Размер изображения:", ' x '.join(map(str, [h, w])))
    print("img is RGB" if c == 3 else f"img has {c} channel(-s)")
    print()

    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, 1, 1),
        nn.ReLU(), 
        nn.MaxPool2d(2), #16x16px
        nn.Conv2d(32, 64, 3, 1, 1),
        nn.ReLU(),
        nn.MaxPool2d(2), #8x8px
        nn.Conv2d(64, 128, 3, 1, 1), #out.shape = (batch_size, 128, 8, 8)
        nn.ReLU(),
        nn.Flatten(), #out.shape = (batch_size, 128 * 8 * 8)
        nn.Linear(128 * 8 * 8, sz_batch), #out.shape = (32, 32)
        nn.ReLU(),
        nn.Linear(32, 10),
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-2)

    patience, bad_epoch = 10, 0
    best_state, best_loss = model.state_dict(), float("inf")
    epochs = 10 + patience

    for e in range(epochs):
        is_save = False
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device=device)
        valid_loss = run_epoch(model, valid_loader, criterion, None, device=device)

        if valid_loss < best_loss:
            best_loss = valid_loss 
            bad_epoch = 0 
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            is_save = True
        else:
            bad_epoch += 1 
            if bad_epoch >= patience: 
                print("Early stopping")
                print("epoch =", e)

        print(
            f"{e}:\tbest loss = {best_loss}\n"
            f"\ttrain loss = {train_loss}\n"
            f"\tvalid loss = {valid_loss}\n"
            f"\tbad epoch = {bad_epoch}\n"
            f"\t>> {"model is save" if is_save else "model isn't save"}\n"
        )

    os.makedirs("model", exist_ok=True)

    model.load_state_dict(best_state)

   # checkpoint = {
   #     "state_dict": best_state,
    #    "optim_state": optimizer.state_dict(),
     #   "epochs": epochs 
    #}
    torch.save(obj=best_state, f="model/cifar10_weigths.pth")

if __name__ == "__main__":
    main()
