import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from CNN_base_model import StandardCNN

# 1. Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load A and A′ models
model_a = StandardCNN().to(device)
model_a.load_state_dict(torch.load("cnn_cifar10.pth"))
model_a.eval()

model_ap = StandardCNN().to(device)
model_ap.load_state_dict(torch.load("cnn_cifar10_aprime.pth"))
model_ap.eval()

# 3. Load CIFAR-10 test set
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform)
testloader = DataLoader(testset, batch_size=1, shuffle=False)

# 4. PETM confidence comparison
total = 0
a_wrong = 0
a_wrong_and_low_conf = 0
low_conf_and_a_wrong = 0
low_conf_total = 0

with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)

        output_a = model_a(inputs)
        output_ap = model_ap(inputs)

        _, pred_a = output_a.max(1)
        _, pred_ap = output_ap.max(1)

        is_correct_a = pred_a.item() == labels.item()
        is_same = pred_a.item() == pred_ap.item()

        total += 1

        if not is_correct_a:
            a_wrong += 1
            if not is_same:
                a_wrong_and_low_conf += 1  # A 틀림 + low confidence

        if not is_same:
            low_conf_total += 1
            if not is_correct_a:
                low_conf_and_a_wrong += 1  # low confidence + A 틀림

# 5. 결과 출력
if a_wrong > 0:
    p1 = 100 * a_wrong_and_low_conf / a_wrong
    print(f"Among A's incorrect predictions, {p1:.2f}% were low-confidence.")
else:
    print("A model made no mistakes on the test set.")

if low_conf_total > 0:
    p2 = 100 * low_conf_and_a_wrong / low_conf_total
    print(f"Among low-confidence predictions, {p2:.2f}% were incorrect by A.")
else:
    print("No low-confidence predictions were found.")