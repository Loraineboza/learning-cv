import torch
import torch.utils.data as data

class MyDataset(data.Dataset):

    def __init__(self, x, true):
        self.x = x
        self.true = true

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.true[idx]

class MyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.w = torch.nn.Parameter(torch.randn(5, 10))
        self.bias = torch.nn.Parameter(torch.randn(10))

    def forward(self, x):
        return x @ self.w + self.bias

def main():
    x = torch.randn(100, 10, 5)

    true_w = torch.randn(5, 10)
    true_bias = torch.randn(10)
    torch.save(
            {
                "true_w":true_w,
                "true_bias":true_bias,
            },
            "data/ground_truth.pt"
    )

    true = x @ true_w + true_bias

    dataset = MyDataset(x, true)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train

    train_dataset, val_dataset = data.random_split(
        dataset,
        [n_train, n_val]
    )
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False
    )


    model = MyModel()

    mse = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01
    )


    best_val_loss = float("inf")
    patience = 15
    epochs_without_improvement = 0
    epochs = 10000

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        total_train_elements = 0

        for x_train, y_train in train_loader:
            pred = model(x_train)
            loss = mse(pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            n_elements = y_train.numel()
            total_train_loss += loss.item() * n_elements
            total_train_elements += n_elements

        # средняя ошибка обучения за эпоху (пригодится для вывода)
        avg_train_loss = total_train_loss / total_train_elements        
        model.eval()
        total_val_loss = 0.0
        total_val_elements = 0

        with torch.no_grad():
            for x_val, y_val in val_loader:
                pred = model(x_val)
                loss = mse(pred, y_val)
                n_elements = y_val.numel()

                total_val_loss += loss.item() * n_elements
                total_val_elements += n_elements

        # средняя ошибка валидации за эпоху
        avg_val_loss = total_val_loss / total_val_elements

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(
                model.state_dict(),
                "model/model155.pth"
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping")
            print(f"epoch = {epoch}")
            break

        if epoch % max(1, epochs // 20) == 0:
            print(
                f"{epoch}: "
                f"train_loss={avg_train_loss:.10f}, "
                f"val_loss={avg_val_loss:.10f}, "
                f"best_val={best_val_loss:.10f}"
            )

if __name__ == "__main__":
    main()
