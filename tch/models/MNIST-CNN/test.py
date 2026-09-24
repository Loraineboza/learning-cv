import torch 
from torchvision import datasets, transforms
from train import CnnModel

def main():
    model = CnnModel()
    state_dict = torch.load("model/mnist_cnn_model2.pth", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    print(model)
    
    tfs = transforms.ToTensor()
    test_dataset = datasets.MNIST(
        root="data", train=False, transform=tfs, target_transform=None, download=True
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                              num_workers=0,
                                              batch_size=32, 
                                              shuffle=False)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    total_correct, total_loss = 0, 0.0
    total_n = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb)
            loss = criterion(logits, yb)

            pred = logits.argmax(dim=1)
            correct = (pred == yb).sum().item()

            n = yb.numel()
            total_correct += correct
            total_loss += loss.item() * n 
            total_n += n

    avg_correct = total_correct / total_n 
    avg_loss = total_loss / total_n 

    print(f"average correct of testing: {avg_correct * 100:.2f}%")
    print(f"average loss of tesing: {avg_loss:.10f}")
    
    print()
    print(f"total correct: {total_correct} from ", end='')
    print(f"total n: {total_n}")
    print("total parameters:", sum([params.numel() for params in model.parameters()]))
if __name__ == "__main__":
    main()
