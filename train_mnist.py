import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

class MNIST_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 784)
        self.fc2 = nn.Linear(784, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

        # Proper Kaiming initialization for ReLU layers
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        # Last layer: default Xavier or zero bias is fine, but many use normal init
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)          # (B,1,28,28) -> (B,784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)                    # No activation on final layer for CE Loss
        return x


def train_epoch(model, optimizer, criterion, train_loader, scaler, scheduler, use_amp, device):
    model.train()
    total_loss = 0.0
    correct = 0
    for data, target in train_loader:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            output = model(data)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # OneCycleLR should be stepped every batch
        scheduler.step()
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
    avg_loss = total_loss / len(train_loader.dataset)
    acc = 100. * correct / len(train_loader.dataset)
    return avg_loss, acc

@torch.no_grad()
def test(model, test_loader, use_amp, device):
    model.eval()
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            output = model(data)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
    acc = 100. * correct / len(test_loader.dataset)
    return acc

def main():
    Epochs = 15
    # Device & Mixed Precision
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    use_amp = device.type == "cuda"  # Automatic Mixed Precision only on GPU
    # Data
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)
    model = MNIST_Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        steps_per_epoch=len(train_loader),
        epochs=Epochs,
        pct_start=0.3,
        anneal_strategy='cos'
    )
    scaler = torch.amp.GradScaler(enabled=use_amp)
    print("\nTraining started...\n")
    start_time = time.time()
    for epoch in range(1, Epochs+1):
        train_loss, train_acc = train_epoch(model, optimizer, criterion, train_loader, scaler, scheduler, use_amp, device)
        test_acc = test(model, test_loader, use_amp, device)
        print(f"Epoch {epoch:2d} | Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:6.2f}% | Test Acc: {test_acc:6.3f}%")
    total_time = time.time() - start_time
    print(f"\nFinished! Total time: {total_time:.1f} sec")
    print(f"Test accuracy: {test_acc:.3f}%")
    torch.save(model.state_dict(), "mnist_784_256_10.pth")
    print("Model saved → mnist_784_256_10.pth")


if __name__ == "__main__":
    main()