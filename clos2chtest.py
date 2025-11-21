import torch
from clos import Clos
from train_mnist import MNIST_Net
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================
# 1. Тохиргоо
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Ашиглаж байгаа төхөөрөмж: {device}")
use_amp = True if device.type == "cuda" else False

# ==========================
# 2. Dataset (MNIST)
# ==========================
model = MNIST_Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
model = MNIST_Net().to(device)
model.load_state_dict(torch.load("mnist_784_256_10.pth", map_location=device))
print(f"Нийт linear параметр: {sum(p.numel() for p in model.fc1.parameters()):,}")
clos = Clos(784, 784, channel=2).to(device)
clos.load_state_dict(torch.load("clos_784_best_test.pth", map_location=device))
model.fc1 = clos
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)
model.eval()
print("Clone done:", model)
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Тестийн нарийвчлал: {100 * correct / total:.2f}%") # want to increase accuracy to >97%
print(f"Нийт clos параметр: {sum(p.numel() for p in clos.parameters()):,}")

