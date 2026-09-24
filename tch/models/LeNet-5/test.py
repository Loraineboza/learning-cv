import torch, torchvision   
from train import MnistModel
def main():
    model = MnistModel()
    state_dict = torch.load("model/model300.pth", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    tfs = torchvision.transforms.ToTensor()
    test_dataset = torchvision.datasets.MNIST(
        root="data_mnist",
        train=False,
        transform=tfs,
        target_transform=None,
        download=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        num_workers=0,
        batch_size=31,
        shuffle=False,
        drop_last=False
    )
    
    criterion = torch.nn.CrossEntropyLoss()

    total_correct, total_n = 0, 0
    total_loss = 0.0
    
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            
            pred = logits.argmax(dim=1)
            correct = (pred == yb).sum().item()
    
            n = yb.numel()
            total_loss += loss.item() * n
            total_correct += correct
            total_n += n

        
    avg_correct = total_correct / total_n 
    avg_loss = total_loss / total_n

    print(
        f"average correct = {avg_correct * 100:.2f}%\n"
            f"average loss = {avg_loss:.10f}"
    )
if __name__ == "__main__":
    main()


