import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, TensorDataset
from torchvision import transforms, datasets
from torchvision.utils import save_image
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 32
NOISE_DIM = 100
EPOCHS = 100
CLASSIFIER_EPOCHS = 20
AUGMENT_MULTIPLIER = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "Data/train/normal"
save_dir = "generated_images"
os.makedirs(save_dir, exist_ok=True)


class SingleClassDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) 
                          if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # Dummy label (not used by GAN)

gan_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
])

train_dataset = SingleClassDataset(root_dir=data_dir, transform=gan_transform)
dcgan_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
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
    g_losses = []
    d_losses = []
    d_real_accs = []
    d_fake_accs = []

    for epoch in range(epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        real_correct = 0
        fake_correct = 0
        total_samples = 0

        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            total_samples += batch_size

            discriminator.zero_grad()

            real_output = discriminator(real_images)
            real_loss = criterion(real_output, torch.full((batch_size, 1), 1.0, device=device))

            real_preds = torch.sigmoid(real_output) > 0.5
            real_correct += real_preds.sum().item()

            noise = torch.randn(batch_size, NOISE_DIM, device=device)
            fake_images = generator(noise)
            fake_output = discriminator(fake_images.detach())
            fake_loss = criterion(fake_output, torch.full((batch_size, 1), 0.0, device=device))

            fake_preds = torch.sigmoid(fake_output) < 0.5
            fake_correct += fake_preds.sum().item()

            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizerD.step()
            epoch_d_loss += d_loss.item()


            generator.zero_grad()
            fake_output = discriminator(fake_images)
            g_loss = criterion(fake_output, torch.full((batch_size, 1), 1.0, device=device))
            g_loss.backward()
            optimizerG.step()
            epoch_g_loss += g_loss.item()


        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)

        d_real_acc = real_correct / total_samples * 100
        d_fake_acc = fake_correct / total_samples * 100
        d_real_accs.append(d_real_acc)
        d_fake_accs.append(d_fake_acc)

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Generator Loss: {avg_g_loss:.4f}")
        print(f"Discriminator Loss: {avg_d_loss:.4f}")
        print(f"Discriminator Real Accuracy: {d_real_acc:.2f}%")
        print(f"Discriminator Fake Accuracy: {d_fake_acc:.2f}%")

        with torch.no_grad():
            noise = torch.randn(16, NOISE_DIM, device=device)
            generated = generator(noise).cpu()
            generated = (generated + 1) / 2  # Scale to [0,1]
            save_image(generated, os.path.join(save_dir, f"epoch_{epoch+1}.png"), nrow=4)

    return g_losses, d_losses, d_real_accs, d_fake_accs

train_gan(dcgan_loader, EPOCHS)

def generate_synthetic_images(num_images):
    generator.eval()
    noise = torch.randn(num_images, NOISE_DIM, device=device)
    with torch.no_grad():
        synthetic_images = generator(noise).cpu()
        synthetic_images = (synthetic_images + 1) / 2  # [0,1]
    return synthetic_images

total_train_images = len(train_dataset)
num_synthetic = int(total_train_images * AUGMENT_MULTIPLIER)
synthetic_images = generate_synthetic_images(num_synthetic)

classifier_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor()
])

target_transform = transforms.Lambda(lambda y: torch.tensor(y, dtype=torch.long))

train_dataset_classifier = datasets.ImageFolder(
    "Data/train",
    transform=classifier_transform,
    target_transform=target_transform
)
valid_dataset = datasets.ImageFolder(
    "Data/valid",
    transform=classifier_transform,
    target_transform=target_transform
)
test_dataset = datasets.ImageFolder(
    "Data/test",
    transform=classifier_transform,
    target_transform=target_transform
)

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

class_to_idx = train_dataset_classifier.class_to_idx
normal_class_idx = class_to_idx['normal']

synthetic_labels = torch.full((num_synthetic,), normal_class_idx, dtype=torch.long)
synthetic_dataset = TensorDataset(synthetic_images, synthetic_labels)
augmented_dataset = ConcatDataset([train_dataset_classifier, synthetic_dataset])
augmented_loader = DataLoader(augmented_dataset, BATCH_SIZE, shuffle=True)

def train_classifier(model, train_loader, valid_loader, epochs):  # <-- Define FIRST
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
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
                DataLoader(train_dataset_classifier, BATCH_SIZE, shuffle=True),
                DataLoader(valid_dataset, BATCH_SIZE),
                CLASSIFIER_EPOCHS)

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

print(f"\nTest Accuracy without augmentation: {test_acc_no_aug:.4f}")
print(f"Test Accuracy with augmentation: {test_acc_with_aug:.4f}")
