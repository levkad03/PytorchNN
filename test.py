import cv2
import torch

from model import ConvNet

# Selecting device to train on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_classes = 15
batch_size = 64

class_names = [
    "Bear",
    "Bird",
    "Cat",
    "Cow",
    "Deer",
    "Dog",
    "Dolphin",
    "Elephant",
    "Giraffe",
    "Horse",
    "Kangaroo",
    "Lion",
    "Panda",
    "Tiger",
    "Zebra",
]

image_path = "./data/test_images/istockphoto-140157656-612x612.jpg"

image = cv2.imread(image_path)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = cv2.resize(image, (227, 227))

image = torch.from_numpy(image).float()

# Изменяем оси для приведения к формату [batch_size, channels, height, width]
image = image.permute(2, 0, 1)

# Добавляем размер батча
image = image.unsqueeze(0)


model = ConvNet().to(device)
model.load_state_dict(torch.load("./models/model.pth"))

model.eval()

with torch.no_grad():
    image = image.to(device)

    outputs = model(image)

    _, predicted = torch.max(outputs, 1)

    print(f"Predicted class: {class_names[predicted]}")
