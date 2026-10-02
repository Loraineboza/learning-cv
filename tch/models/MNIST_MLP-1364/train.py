import torch, torchvision
import os
class MnistModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(28 * 28 * 1, 512, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 10, bias=True)
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def run_epoch(
    model, loader, criterion, optimizer=None
) -> float:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_n = 0.0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for xb, yb in loader:
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
    tfs = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])  

    #datasets
    main_dataset = torchvision.datasets.MNIST(
        root="data_mnist", 
        train=True, 
        transform=tfs, 
        target_transform=None,
        download=True
    )
    
    # split MNIST training dataset into train and valid
    n_train = int(0.8 * len(main_dataset))
    n_valid = len(main_dataset) - n_train
    train_dataset, valid_dataset = torch.utils.data.random_split(
        main_dataset, [n_train, n_valid]
    )
    
    print(f"len of main_dataset: {len(main_dataset)}")
    print(f"len of train_dataset = {len(train_dataset)}; valid_dataset = {len(valid_dataset)}\n")

    image, label = train_dataset[0] 
    print(f"1 object(img) shape = {image.shape}\n")

    #iterators / loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=0,
        batch_size=32,
        shuffle=True,
        drop_last=False
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        num_workers=0,
        batch_size=32,
        shuffle=False,
        drop_last=False
    )
    

    model = MnistModel()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=5e-2)

    print(f"structure of the model:\n{model}")
    print()   

    pred_avg_loss, best_state = float("inf"), model.state_dict()
    bad_epoch, patience = 0, 15
    epochs = 1000
    

    for e in range(epochs):
        train_avg_loss = run_epoch(model, train_loader, criterion, optimizer)
        valid_avg_loss = run_epoch(model, valid_loader, criterion, None)

        if valid_avg_loss < pred_avg_loss:
            pred_avg_loss = valid_avg_loss
            bad_epoch = 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            bad_epoch += 1
            if bad_epoch >= patience:
                print("early stopping train to epoch=", e)
                break
        
        if e % max(1, epochs//30) == 0:
            print(f"{e}: valid avg loss: {valid_avg_loss}")
        
    os.makedirs("model", exist_ok=True)
    torch.save(best_state, "model/model300.pth")
    model.load_state_dict(best_state)

    print("\ndownload state_dict of the model")
    

if __name__ == "__main__":
    main()
