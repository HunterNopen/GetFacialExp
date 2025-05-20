import numpy as np
import cv2
import os
import shutil
import kagglehub

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

train_path = "./train"
val_path = "./test"

if not os.path.exists(train_path) and not os.path.exists(val_path):
    os.environ['KAGGLEHUB_CACHE'] = './'
    kagglehub.dataset_download("msambare/fer2013")

    shutil.move("./datasets/msambare/fer2013/versions/1/train", "./")
    shutil.move("./datasets/msambare/fer2013/versions/1/test", "./")

    shutil.rmtree("./datasets")

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataloader = DataLoader(ImageFolder(train_path, transform=transform), batch_size=16, shuffle = True)

val_dataloader = DataLoader(ImageFolder(val_path, transform=transform), batch_size=16)

model = nn.Sequential(
    nn.Conv2d(1, 32, (3,3)),
    nn.ReLU(),
    nn.Conv2d(32, 64, (3,3)),
    nn.ReLU(),
    nn.MaxPool2d((2,2)),
    nn.Dropout(0.2),

    nn.Conv2d(64, 128, (3,3)),
    nn.ReLU(),
    nn.Conv2d(128, 128, (3,3)),
    nn.ReLU(),
    nn.MaxPool2d((2,2)),
    nn.Dropout(0.2),

    nn.Flatten(),
    nn.Linear(128 * 9 * 9, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 7),
    nn.Softmax(dim=1)
).to(device)

loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr = 2e-2)

def train(model, dataloader, loss_fn, optim, epochs = 5):
    model.train()

    for epoch in range(epochs):
        loss_epoch = 0.0
        print("!!!!!!!!!!!!!!!!!!!!!!!")

        for inputs, labels in dataloader:
            optim.zero_grad()
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optim.step()

            loss_epoch += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_epoch/len(dataloader):.4f}")

train(model, train_dataloader, loss_fn, optim)

# torch.save(model, 'model.pth')
# model = torch.load('model.pth')
# model.eval()