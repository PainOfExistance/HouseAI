import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, TensorDataset
from torchvision import transforms, datasets
from torchvision.utils import save_image
import os
import numpy as np
import matplotlib.pyplot as plt

IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 32
NOISE_DIM = 100
EPOCHS = 50
CLASSIFIER_EPOCHS = 20
AUGMENT_MULTIPLIER = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "Data/train"
save_dir = "generated_images"
os.makedirs(save_dir, exist_ok=True)

gan_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
])

train_dataset = datasets.ImageFolder(root=data_dir, transform=gan_transform)
dcgan_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# DCGAN
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, NOISE_DIM)

        self.main = nn.Sequential(
            nn.Linear(NOISE_DIM, 8*8*256),
            nn.BatchNorm1d(8*8*256),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 3, 5, stride=2, padding=2, output_padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# DCGAN
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(128*16*16, 1)
        )

    def forward(self, x):
        return self.main(x)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

optimizerG = optim.Adam(generator.parameters(), lr=1e-4)
optimizerD = optim.Adam(discriminator.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

def train_gan(dataloader, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # Train Discriminator
            discriminator.zero_grad()

            real_output = discriminator(real_images)
            real_loss = criterion(real_output, torch.full((batch_size, 1), 1.0, device=device))

            noise = torch.randn(batch_size, NOISE_DIM, device=device)
            fake_images = generator(noise)
            fake_output = discriminator(fake_images.detach())
            fake_loss = criterion(fake_output, torch.full((batch_size, 1), 0.0, device=device))

            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizerD.step()

            # Train Generator
            generator.zero_grad()
            fake_output = discriminator(fake_images)
            g_loss = criterion(fake_output, torch.full((batch_size, 1), 1.0, device=device))
            g_loss.backward()
            optimizerG.step()

        # Save images
        with torch.no_grad():
            noise = torch.randn(16, NOISE_DIM, device=device)
            generated = generator(noise).cpu()
            generated = (generated + 1) / 2  # [0,1]
            save_image(generated, os.path.join(save_dir, f"epoch_{epoch+1}.png"), nrow=4)

train_gan(dcgan_loader, EPOCHS)

def generate_synthetic_images(num_images):
    generator.eval()
    noise = torch.randn(num_images, NOISE_DIM, device=device)
    with torch.no_grad():
        synthetic_images = generator(noise).cpu()
        synthetic_images = (synthetic_images + 1) / 2  # [0,1]
    return synthetic_images

total_train_images = sum([len(files) for _, _, files in os.walk(data_dir)])
num_synthetic = int(total_train_images * AUGMENT_MULTIPLIER)
synthetic_images = generate_synthetic_images(num_synthetic)

classifier_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder("Data/train", transform=classifier_transform)
valid_dataset = datasets.ImageFolder("Data/valid", transform=classifier_transform)
test_dataset = datasets.ImageFolder("Data/test", transform=classifier_transform)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.Flatten(),
            nn.Linear(128*12*12, 128), nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.main(x)
def train_classifier(model, train_loader, valid_loader, epochs):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch in train_loader:
            if len(batch) == 2:
                inputs, labels = batch
            else:
                inputs, labels, _ = batch

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        print(f"Epoch {epoch+1}: Loss: {total_loss/len(train_loader):.4f}, "
              f"Train Acc: {correct/total:.4f}, Val Acc: {val_correct/val_total:.4f}")

classifier_no_aug = Classifier().to(device)
train_classifier(classifier_no_aug,
                DataLoader(train_dataset, BATCH_SIZE, shuffle=True),
                DataLoader(valid_dataset, BATCH_SIZE),
                CLASSIFIER_EPOCHS)

if not isinstance(synthetic_images, torch.Tensor):
    synthetic_images = torch.tensor(synthetic_images.numpy(), dtype=torch.float32)

# Create synthetic dataset with proper labels (assuming class 0 for synthetic)
synthetic_labels = torch.zeros(len(synthetic_images), dtype=torch.long)
synthetic_dataset = TensorDataset(synthetic_images, synthetic_labels)

augmented_dataset = ConcatDataset([train_dataset, synthetic_dataset])

def custom_collate(batch):
    imgs = []
    labels = []
    for item in batch:
        if len(item) == 2:
            img, label = item
        else:
            img, label, _ = item
        imgs.append(img)
        labels.append(label)
    return torch.stack(imgs, 0), torch.tensor(labels)

augmented_loader = DataLoader(augmented_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            collate_fn=custom_collate)

classifier_with_aug = Classifier().to(device)
train_classifier(classifier_with_aug,
                augmented_loader,
                DataLoader(valid_dataset, BATCH_SIZE),
                CLASSIFIER_EPOCHS)

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

test_acc_no_aug = evaluate(classifier_no_aug, DataLoader(test_dataset, BATCH_SIZE))
test_acc_with_aug = evaluate(classifier_with_aug, DataLoader(test_dataset, BATCH_SIZE))

print(f"\nTest Accuracy without GAN: {test_acc_no_aug:.4f}")
print(f"Test Accuracy with GAN: {test_acc_with_aug:.4f}")
