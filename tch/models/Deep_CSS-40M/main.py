import torch 
import torch.nn as nn
import torch.utils.data as data 
import torchvision 
import torchvision.transforms.v2 as v2 
import os 

class TransformedDataset(data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        x = self.transform(x) if self.transform is not None else x 
        y = torch.tensor(y, dtype=torch.int64)
        return x, y 
    def __len__(self):
        return len(self.subset)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2,2), #16px
            
            nn.Conv2d(64, 128, 3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2,2), #8px

            nn.Conv2d(128, 256, 3, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 128, 3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Flatten(1,-1),
            nn.Linear(128 * 8 * 8, 4096),
            nn.Dropout(0.5),
            nn.ReLU(),
            
            nn.Linear(4096, 2048),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.Linear(2048, 1024),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.Linear(1024, 32),
            nn.ReLU(),
        ) 
        self.out = nn.Linear(32, 40)

    def __call__(self, x):
        x = self.layers(x)
        out = self.out(x)
        return out

def run_epoch(model, loader, criterion, optimizer=None, *, device="cpu"):
    is_train = optimizer is not None 
    model.train(is_train)
    total_loss, total_correct, total_n = 0.0, 0, 400 
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

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


    return total_loss / total_n, total_correct / total_n

def save_state_dict(best_state, model, folder, filename):
    os.makedirs(folder, exist_ok=True)
    model.load_state_dict(best_state)
    path = os.path.join(folder, filename)
    torch.save(obj=best_state, f=path)

    return model, path 

if __name__ == "__main__":
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    lr = 5e-3
    patience = 10 
    bad_epoch = 0 
    best_state = None 
    best_valid_loss = float("inf")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 9_990+ patience 

        
    obj = 20_000
    c = 3 
    h, w = 32, 32
    num_classes = 10
    
    tfs_train = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomCrop(size=32, padding=4),
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std)
    ])

    tfs_valid = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std)
    ])
    

    tfs_test = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std)
    ])

    true = torch.randint(0, num_classes, (obj,))
    main_dataset = torch.utils.data.TensorDataset(torch.randn(obj, c, h, w), true)
    n = len(main_dataset)
    n_train = int(0.8 * n)
    n_valid = n - n_train 

    train_subs, valid_subs = data.random_split(main_dataset, [n_train, n_valid])
    train_dataset, valid_dataset = TransformedDataset(train_subs, tfs_train), TransformedDataset(valid_subs, tfs_valid)

    test_dataset = torch.utils.data.TensorDataset(torch.randn(obj, c, h, w), true)
    train_loader = data.DataLoader(
        train_dataset, 
        num_workers=4, pin_memory=True, 
        batch_size=32, shuffle=True)
    valid_loader = data.DataLoader(
        valid_dataset, 
        num_workers=4, pin_memory=True, 
        batch_size=32, shuffle=True
    )
    test_loader = data.DataLoader(
        test_dataset, 
        num_workers=4, pin_memory=True, 
        batch_size=32, shuffle=False
    )

    model = Model().to(device)
    print(model)
    print(f"работает на {device}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    in_channels, x, w = train_dataset[0][0].shape[::]
    print(f"Каждое изображение имеет {x} x {w} разрешение")
    print(f"содержит {in_channels} channels(-s)\n")

    for e in range(1, epochs+1):
        is_save = True
        train_loss, train_acc, = run_epoch(model, train_loader, criterion, optimizer, device=device)
        valid_loss, valid_acc = run_epoch(model, valid_loader, criterion, device=device)
        
        scheduler.step()
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            bad_epoch = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epoch += 1 
            if bad_epoch >= patience:
                print("Early stopping epoch =", e)
                break 
            is_save = False
        print(
            f"Epoch {e:02d}/{epochs} ({e/epochs:.1%})\n"
            f"train loss = {train_loss:.10f} | train acc = {train_acc:.2%} | "
            f"valid loss = {valid_loss:.10f} | valid acc = {valid_acc:.2%} | "
            f"model is {"save" if is_save else "not save"} | "
            f"before early stop => {bad_epoch}/{patience}\n"
            f"LR: {scheduler.get_last_lr()[0]:.10f}\n"
        )
    print()
    model, path = save_state_dict(best_state, model, folder="model", filename="model_weights.pth")
    if path is not None:
        print("Параметры модели успешно сохранены по пути \"{path}\"")
    else:
        print(f"Параметры не были сохранены. Ошибка = {path}")

    total_corerct = 0 
    total_n = 0 
    
    #testing
    model.load_state_dict(torch.load(f="model/model_weights.pth", weights_only=True))
    
    total_parameters = sum([param.numel() for param in model.parameters()])
    print(f"Количество параметров модели: {total_parameters}")
    

    loss, acc = run_epoch(
        model, 
        test_loader, 
        criterion, 
        None, 
        device=device
    )

    print(f"test loss = {loss:.10f} | test acc = {acc:.2%}")
