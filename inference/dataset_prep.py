import torch
import shutil
import os
import kagglehub
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

class Dataset():

    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f'\n{device}')

        self.train_path = "./train"
        self.val_path = "./test"

        self.transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])

    def setup(self):
        if not os.path.exists(self.train_path) and not os.path.exists(self.val_path):
            os.environ['KAGGLEHUB_CACHE'] = './'
            kagglehub.dataset_download("msambare/fer2013")

            shutil.move("./datasets/msambare/fer2013/versions/1/train", "./")
            shutil.move("./datasets/msambare/fer2013/versions/1/test", "./")

            shutil.rmtree("./datasets")

    def get_train_dataloader(self):
        return DataLoader(ImageFolder(self.train_path, transform=self.transform), batch_size=8, shuffle = True)

    def get_val_dataloader(self):
        return DataLoader(ImageFolder(self.val_path, transform=self.transform), batch_size=8)