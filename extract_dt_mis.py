import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from CNN_base_model import StandardCNN

# 1. Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. CIFAR-10 train set
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=False)

# 3. Load trained model
model = StandardCNN().to(device)
model.load_state_dict(torch.load("cnn_cifar10.pth"))
model.eval()

# 4. Misclassified sample extraction
misclassified_inputs = []
misclassified_labels = []

total = 0
misclassified = 0

with torch.no_grad():
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        incorrect = predicted != targets

        total += targets.size(0)
        misclassified += incorrect.sum().item()

        if incorrect.any():
            misclassified_inputs.append(inputs[incorrect].cpu())
            misclassified_labels.append(targets[incorrect].cpu())

# 5. Save misclassified samples
Dt_mis_inputs = torch.cat(misclassified_inputs, dim=0)
Dt_mis_labels = torch.cat(misclassified_labels, dim=0)

torch.save(Dt_mis_inputs, "Dt_mis_inputs.pt")
torch.save(Dt_mis_labels, "Dt_mis_labels.pt")

# 6. Output
misclassified_ratio = 100 * misclassified / total
print(f"Misclassified ratio in training set: {misclassified_ratio:.2f}%")
print("Saved: Dt_mis_inputs.pt, Dt_mis_labels.pt")