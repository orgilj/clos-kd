import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from clos import Clos, transfer_fc_to_clos          # Your custom CLOS module
from train_mnist import MNIST_Net, test                   # Your trained MLP from previous script

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    # Load pre-trained MLP
    model = MNIST_Net().to(device)
    model.load_state_dict(torch.load("mnist_784_256_10.pth", map_location=device))
    print("Original model loaded (mnist_784_256_10.pth)")
    # Original fc1 to use as teacher for CLOS distillation
    fc = model.fc1
    # Test loader (same as training script)
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False,
                             num_workers=4, pin_memory=True)
    accuracy_original = test(model, test_loader, use_amp=True, device=device)
    print("Accuracy of original Linear layer:", accuracy_original)
    # Start searching for a good CLOS replacement
    best_acc = 0.0
    for i in range(10):
        clos = transfer_fc_to_clos(
            fc=fc,
            channel=2,
            max_steps=10000,
            W_lr=0.1,
            B_lr=0.3,
            verbose=True
        )
        model.fc1 = clos
        model.eval()
        accuracy_clos = test(model, test_loader, use_amp=True, device=device)
        print(f"Accuracy after replacing with CLOS {i}th:", accuracy_clos)
        if accuracy_clos > best_acc:
            best_acc = accuracy_clos
            torch.save(clos.state_dict(), "clos_784_best_test.pth")
            print(f"  → NEW BEST! {best_acc:.3f}% → saved to clos_784_best_test.pth\n")

if __name__ == "__main__":
    main()