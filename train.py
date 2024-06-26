import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as transforms

from model import ConvNet

# Selecting device to train on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Hyperparameters
num_epochs = 10
batch_size = 64
learning_rate = 0.001
num_classes = 15

# Transformations for the training set
train_transform = transforms.Compose(
    [
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Transformations for the test set
test_transform = transforms.Compose(
    [
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Loading the dataset
dataset = torchvision.datasets.ImageFolder(root="./data/animal_data", transform=None)


# Splitting the dataset into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Apply the transformations to the datasets
train_dataset.dataset.transform = train_transform
test_dataset.dataset.transform = test_transform

# Creating DataLoader for train and test datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Init Model
model = ConvNet(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


n_total_steps = len(train_loader)


def test_model(model, train_loader, criterion, device):
    model.eval()  # Переключение в режим оценки (без градиентов)
    n_correct = 0
    n_samples = 0

    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

    accuracy = 100.0 * n_correct / n_samples
    return accuracy


# Использование функции test_model после каждой эпохи обучения
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # После каждой эпохи считаем точность на тестовом наборе данных
    train_accuracy = test_model(model, train_loader, criterion, device)
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.2f}%, \
            loss: {loss:.2f}"
    )


print("Finished training")

FILE = "./models/model.pth"
torch.save(model.state_dict(), FILE)


model = ConvNet().to(device)
model.load_state_dict(torch.load("./models/model.pth"))

model.eval()

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(num_classes)]
    n_class_samples = [0 for i in range(num_classes)]

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for label, pred in zip(labels, predicted):
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1


acc = 100.0 * n_correct / n_samples
print(f"Accuracy of the network: {acc}")
