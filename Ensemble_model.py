import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from CNN_base_model import StandardCNN

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models A and A′
model_a = StandardCNN().to(device)
model_a.load_state_dict(torch.load("cnn_cifar10.pth"))
model_a.eval()

model_ap = StandardCNN().to(device)
model_ap.load_state_dict(torch.load("cnn_cifar10_aprime.pth"))
model_ap.eval()

# DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = DataLoader(trainset, batch_size=1, shuffle=False)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = DataLoader(testset, batch_size=1, shuffle=False)

# Ensemble prediction and selection ratio
def evaluate_ensemble(loader, name):
    correct = 0
    total = 0
    used_a = 0
    used_ap = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            model_a.eval()
            model_ap.eval()

            output_a = model_a(inputs)
            output_ap = model_ap(inputs)

            _, pred_a = output_a.max(1)
            _, pred_ap = output_ap.max(1)

            pred_a_item = pred_a.item()
            pred_ap_item = pred_ap.item()
            label_item = labels.item()

            if pred_a_item == pred_ap_item:
                final_pred = pred_a_item
                used_a += 1
            else:
                final_pred = pred_ap_item
                used_ap += 1

            if final_pred == label_item:
                correct += 1
            total += 1

    acc = 100 * correct / total
    a_ratio = 100 * used_a / total
    ap_ratio = 100 * used_ap / total
    print(f"{name} Accuracy (Ensemble): {acc:.2f}%")
    print(f"{name} - A predictions used: {a_ratio:.2f}%, A′ predictions used: {ap_ratio:.2f}%\n")

# Evaluate on train and test sets
evaluate_ensemble(trainloader, "Train")
evaluate_ensemble(testloader, "Test")