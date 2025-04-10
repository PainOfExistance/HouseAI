import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from torchvision import transforms, datasets
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# -------------------------------
# Configuration
# -------------------------------
IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 32
EPOCHS = 200
CLASSIFIER_EPOCHS = 120
AUGMENT_MULTIPLIER = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "Data/train"
save_dir = "generated_images_vqgan"
os.makedirs(save_dir, exist_ok=True)

# -------------------------------
# 1. Dataset + DataLoader
# -------------------------------
transform_gan = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
])

train_dataset_gan = datasets.ImageFolder(root=data_dir, transform=transform_gan)
vqgan_loader = DataLoader(train_dataset_gan, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# -------------------------------
# 2. Define the VQGAN Components
# -------------------------------
class Encoder(nn.Module):
    """
    Encodes an image into a latent representation of shape (latent_dim, H/8, W/8).
    """
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            # Downsample 64 -> 32
            nn.Conv2d(in_channels, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 -> 16
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 -> 8
            nn.Conv2d(256, latent_dim, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """
    Decodes a latent tensor of shape (latent_dim, H/8, W/8) back to an image.
    """
    def __init__(self, out_channels=3, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            # 8 -> 16
            nn.ConvTranspose2d(latent_dim, 256, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 -> 32
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 -> 64
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),
            nn.Tanh()  # final image in [-1, 1]
        )

    def forward(self, z):
        return self.net(z)


class VectorQuantizer(nn.Module):
    """
    A simple vector-quantization module (non-EMA).  
    Codebook is [num_codes, latent_dim].
    """
    def __init__(self, num_codes=512, latent_dim=256, commitment_cost=0.25):
        super().__init__()
        self.num_codes = num_codes
        self.latent_dim = latent_dim
        self.commitment_cost = commitment_cost

        # Codebook: each code is a learnable embedding vector of size latent_dim
        self.codebook = nn.Embedding(num_codes, latent_dim)
        nn.init.normal_(self.codebook.weight, 0, 0.02)

    def forward(self, z):
        """
        z: shape [B, C, H, W]
        """
        B, C, H, W = z.shape
        # Flatten
        z_flattened = z.permute(0, 2, 3, 1).reshape(-1, C)  # [BHW, C]

        # Compute distances to each embedding
        distances = (z_flattened.unsqueeze(1) - self.codebook.weight.unsqueeze(0)).pow(2).sum(-1)
        # [BHW, num_codes]

        # Get the indices of the closest codes
        encoding_indices = distances.argmin(dim=1)  # [BHW]

        # Get corresponding embeddings
        z_q = self.codebook(encoding_indices).view(B, H, W, C)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        # Quantization Loss
        commitment_loss = self.commitment_cost * (z_q.detach() - z).pow(2).mean()
        embedding_loss = (z_q - z.detach()).pow(2).mean()

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        return z_q, encoding_indices, (commitment_loss + embedding_loss)

    def get_codebook_entry(self, indices, shape):
        """
        Given code indices, return the corresponding embeddings reshaped to `shape`.
        """
        z_q = self.codebook(indices)
        z_q = z_q.view(*shape)
        return z_q


class VQGANDiscriminator(nn.Module):
    """
    A simple CNN discriminator to provide adversarial feedback on reconstructed images.
    """
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels*2, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels*2, base_channels*4, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(base_channels*4*(IMG_HEIGHT//8)*(IMG_WIDTH//8), 1),
        )

    def forward(self, x):
        return self.main(x)

# -------------------------------
# 3. VQGAN Model Wrapper
# -------------------------------
class VQGAN(nn.Module):
    def __init__(self, latent_dim=256, num_codes=512):
        super().__init__()
        self.encoder = Encoder(in_channels=3, latent_dim=latent_dim)
        self.decoder = Decoder(out_channels=3, latent_dim=latent_dim)
        self.quantizer = VectorQuantizer(num_codes=num_codes, latent_dim=latent_dim)

    def forward(self, x):
        """
        Return:
          - reconstructed image
          - codebook indices
          - quantization loss
        """
        z_e = self.encoder(x)
        z_q, codes, q_loss = self.quantizer(z_e)
        x_rec = self.decoder(z_q)
        return x_rec, codes, q_loss

# Instantiate VQGAN + Discriminator
latent_dim = 256
num_codes = 512
vqgan = VQGAN(latent_dim=latent_dim, num_codes=num_codes).to(device)
vq_discriminator = VQGANDiscriminator().to(device)

# Optimizers
opt_vq = optim.Adam(vqgan.parameters(), lr=1e-4, betas=(0.5, 0.999))
opt_disc = optim.Adam(vq_discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

# Losses
l1_loss = nn.L1Loss()
bce_loss = nn.BCEWithLogitsLoss()

# -------------------------------
# 4. Train the VQGAN
# -------------------------------
def train_vqgan(dataloader, epochs):
    vqgan.train()
    vq_discriminator.train()

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for real_images, _ in dataloader:
            real_images = real_images.to(device)

            # -------------------
            # Train VQGAN (Encoder+Decoder+Quantizer)
            # -------------------
            opt_vq.zero_grad()
            x_rec, _, q_loss = vqgan(real_images)

            # Reconstruction loss
            recon_loss = l1_loss(x_rec, real_images)

            # Adversarial loss (generator side)
            fake_logits = vq_discriminator(x_rec)
            adv_loss = bce_loss(fake_logits, torch.ones_like(fake_logits, device=device))

            vq_loss = recon_loss + q_loss + 0.1 * adv_loss  # Weighted sum; tune as needed
            vq_loss.backward()
            opt_vq.step()

            # -------------------
            # Train Discriminator
            # -------------------
            opt_disc.zero_grad()

            # Real images
            real_logits = vq_discriminator(real_images)
            real_loss = bce_loss(real_logits, torch.ones_like(real_logits, device=device))

            # Fake images
            fake_logits = vq_discriminator(x_rec.detach())
            fake_loss = bce_loss(fake_logits, torch.zeros_like(fake_logits, device=device))

            disc_loss = real_loss + fake_loss
            disc_loss.backward()
            opt_disc.step()

        # Save some reconstructed samples each epoch
        with torch.no_grad():
            sample = x_rec[:16].cpu().clone()
            sample = (sample + 1) / 2  # scale to [0,1]
            save_image(sample, os.path.join(save_dir, f"epoch_{epoch+1}.png"), nrow=4)

# -------------------------------
# 5. Generate Synthetic Images
# -------------------------------
@torch.no_grad()
def generate_synthetic_images(num_images):
    """
    Sample random code indices from the codebook to generate images.
    """
    vqgan.eval()

    # We assume each latent is shape [latent_dim, 8, 8] for a 64x64 image
    # Actually, we store code *indices*, so shape [8*8] per image
    num_tokens = (IMG_HEIGHT // 8) * (IMG_WIDTH // 8)  # 8x8=64 tokens
    # Randomly choose codebook indices
    codes = torch.randint(0, num_codes, (num_images, num_tokens), device=device)

    # Reshape to (B, H*W), then expand to (B, C, H, W) once we embed them
    # The VectorQuantizer expects shape (B, C, H, W), but get_codebook_entry
    # returns (B*H*W, latent_dim), so we’ll reshape carefully.
    shape = (num_images, num_tokens)
    z_q = vqgan.quantizer.get_codebook_entry(codes.view(-1), (num_images, latent_dim, 8, 8))
    # Now decode
    x_rec = vqgan.decoder(z_q)
    synthetic_images = (x_rec + 1) / 2  # scale to [0,1]
    return synthetic_images.cpu()

# -------------------------------
# 6. Classifier + Datasets
# -------------------------------
target_transform = lambda x: torch.tensor(x)

classifier_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
])
train_dataset = datasets.ImageFolder("Data/train", transform=classifier_transform, target_transform=target_transform)
valid_dataset = datasets.ImageFolder("Data/valid", transform=classifier_transform, target_transform=target_transform)
test_dataset  = datasets.ImageFolder("Data/test",  transform=classifier_transform, target_transform=target_transform)

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
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
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
        
        # Validation
        val_correct = 0
        val_total = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        model.train()

        print(f"Epoch {epoch+1}: "
              f"Loss: {total_loss/len(train_loader):.4f}, "
              f"Train Acc: {correct/total:.4f}, "
              f"Val Acc: {val_correct/val_total:.4f}")

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

# -------------------------------
# 7. Main Execution
# -------------------------------
if __name__ == '__main__':
    # Train the VQGAN model
    train_vqgan(vqgan_loader, EPOCHS)

    # Generate synthetic images for augmentation
    total_train_images = 0
    for _, _, files in os.walk("Data/train"):
        total_train_images += len(files)
    num_synthetic = int(total_train_images * AUGMENT_MULTIPLIER)

    synthetic_images = generate_synthetic_images(num_synthetic)
    # synthetic_images in [0,1], shape [N, 3, 64, 64]

    # -------------------------------
    # Classifier without augmentation
    # -------------------------------
    train_loader_no_aug = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader_no_aug = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    classifier_no_aug = Classifier().to(device)
    train_classifier(classifier_no_aug, train_loader_no_aug, valid_loader_no_aug, CLASSIFIER_EPOCHS)
    test_acc_no_aug = evaluate(classifier_no_aug, test_loader)
    print(f"\nTest Accuracy without augmentation: {test_acc_no_aug:.4f}")

    # -------------------------------
    # Classifier with augmentation
    # -------------------------------
    # Create labels = 0 for all synthetic images (or adapt to your problem)
    synthetic_labels = torch.zeros(num_synthetic, dtype=torch.long)
    synthetic_dataset = TensorDataset(synthetic_images, synthetic_labels)

    augmented_dataset = ConcatDataset([train_dataset, synthetic_dataset])
    augmented_loader = DataLoader(augmented_dataset, batch_size=BATCH_SIZE, shuffle=True)

    classifier_with_aug = Classifier().to(device)
    train_classifier(classifier_with_aug, augmented_loader, valid_loader_no_aug, CLASSIFIER_EPOCHS)
    test_acc_with_aug = evaluate(classifier_with_aug, test_loader)
    print(f"Test Accuracy with augmentation: {test_acc_with_aug:.4f}")
