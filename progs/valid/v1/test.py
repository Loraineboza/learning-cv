import torch
import torch.utils.data as data
from train import MyDataset, MyModel

def main():
    model = MyModel()
    state_dict = torch.load("model/model155.pth")
    model.load_state_dict(state_dict)
    model.eval()

    ground_truth = torch.load("data/ground_truth.pt")

    x = torch.randn(100, 10, 5)
    true_w = ground_truth["true_w"]
    true_bias = ground_truth["true_bias"]

    true = x @ true_w + true_bias

    test_dataset = MyDataset(x, true)
    iter_test = data.DataLoader(test_dataset, num_workers=0, batch_size=32, shuffle=False)

    mse = torch.nn.MSELoss()

    print("test:")
    with torch.no_grad():
        for x_test, y_test in iter_test:
            pred = model(x_test)
            loss = mse(pred, y_test)
            print(f"loss: {loss}")

if __name__ == "__main__":
    main()
