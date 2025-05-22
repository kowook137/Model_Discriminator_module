import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from CNN_base_model import StandardCNN

def train_aprime():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load base A model
    model_aprime = StandardCNN().to(device)
    model_aprime.load_state_dict(torch.load("cnn_cifar10.pth"))

    # 2. Freeze all layers
    for param in model_aprime.parameters():
        param.requires_grad = False

    # 3. Unfreeze only classifier (fine-tuning target)
    for param in model_aprime.classifier.parameters():
        param.requires_grad = True

    model_aprime.train()

    # 4. Load Dt_mis
    Dt_mis_inputs = torch.load("Dt_mis_inputs.pt")
    Dt_mis_labels = torch.load("Dt_mis_labels.pt")

    Dt_mis_dataset = TensorDataset(Dt_mis_inputs, Dt_mis_labels)
    Dt_mis_loader = DataLoader(Dt_mis_dataset, batch_size=16, shuffle=True)

    # 5. Optimizer for fine-tuning
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_aprime.parameters()), lr=0.0005)

    # 6. Fine-tuning loop
    for epoch in range(5):
        correct = 0
        total = 0
        for inputs, labels in Dt_mis_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model_aprime(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        acc = 100 * correct / total
        print(f"[Epoch {epoch+1}] Accuracy on Dt_mis: {acc:.2f}%")

    # 7. Evaluate on full train set
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    trainset = datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=False)

    model_aprime.eval()
    correct_train = 0
    total_train = 0
    with torch.no_grad():
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_aprime(inputs)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
    print(f"Train Accuracy (Full Set): {100 * correct_train / total_train:.2f}%")

    # 8. Evaluate on test set
    testset = datasets.CIFAR10(root="./data", train=False, download=False, transform=transform)
    testloader = DataLoader(testset, batch_size=128, shuffle=False)

    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_aprime(inputs)
            _, predicted = outputs.max(1)
            total_test += labels.size(0)
            correct_test += predicted.eq(labels).sum().item()
    print(f"Test Accuracy: {100 * correct_test / total_test:.2f}%")

    # 9. Save A′ model
    torch.save(model_aprime.state_dict(), "cnn_cifar10_aprime.pth")
    print("A′ model saved: cnn_cifar10_aprime.pth")

if __name__ == "__main__":
    train_aprime()