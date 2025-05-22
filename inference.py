import torch
from torch import nn
from tqdm import tqdm

from dataset_prep import Dataset

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'\n{self.device}')

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
        ).to(self.device)

        self.classifier = nn.Sequential(
            nn.Linear(64 * 6 * 6, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 7)
        ).to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optim = torch.optim.AdamW(self.backbone.parameters(), lr = 1e-4)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
    
        return x

    def train_model(self, dataloader, epochs = 10):
        self.backbone.train()

        for epoch in range(epochs):
            loss_epoch = 0.0
            print(f"Epoch {epoch+1}/{epochs}")

            with tqdm(dataloader, desc=f"Epoch {epoch+1}", unit="batch") as pbar:
                for inputs, labels in pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.optim.zero_grad()
                    outputs = self.forward(inputs)

                    loss = self.loss_fn(outputs, labels)
                    loss.backward()
                    self.optim.step()

                    loss_epoch += loss.item()
                    pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
                
            print(f"\nEpoch {epoch+1}/{epochs}, Loss: {loss_epoch/len(dataloader)}\n")

if __name__ == "__main__":
    dataset = Dataset()
    dataset.setup()

    train_dataloader = dataset.get_train_dataloader()
    print(f'Train Len: {len(train_dataloader)}')

    val_dataloadert = dataset.get_val_dataloader()
    print(f'Val Len: {len(val_dataloadert)}')

    model = Model()
    model.train_model(train_dataloader)

    torch.save(model.state_dict(), 'model.pth')