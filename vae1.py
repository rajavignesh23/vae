import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define a Convolutional VAE architecture
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(ConvVAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # Output: (32, 85, 100)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, 43, 50)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: (128, 21, 25)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output: (256, 10, 12)
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(256 * 10 * 12, latent_dim)  # Mean
        self.fc_logvar = nn.Linear(256 * 10 * 12, latent_dim)  # Log-variance

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 256 * 10 * 12)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: (128, 21, 25)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, 43, 50)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: (32, 85, 100)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # Output: (3, 170, 200) (may differ)
            nn.Sigmoid(),
            nn.Upsample(size=(170, 200))  # Ensure the output size is exactly (170, 200)
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten the output
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, 256, 10, 12)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Loss function with KLD weight
def loss_function(recon_x, x, mu, logvar, kld_weight=0.1):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + kld_weight * KLD  # Apply weight to KLD

# Training function
def train(epoch, model, train_loader, optimizer, kld_weight):
    model.train()
    train_loss = 0
    print(f"Starting epoch {epoch}...")
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, kld_weight)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(f"Epoch {epoch} completed. Loss: {train_loss / len(train_loader.dataset)}")

# Custom dataset to load images from a directory
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        print(f"Found {len(self.image_paths)} images in {image_dir}.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0

# Save model function
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def main():
    image_dir = "/mnt/c/gpy/testdata"

    transform = transforms.Compose([
        transforms.Resize((170, 200)),
        transforms.ToTensor()
    ])

    dataset = ImageDataset(image_dir, transform)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    if len(dataset) == 0:
        print("No images found in the directory.")
        return

    # Initialize VAE model, optimizer, and other parameters
    model = ConvVAE(latent_dim=20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    epochs = 20
    kld_weight = 0.0001  # Start with a smaller weight for KLD

    # Train the model
    for epoch in range(1, epochs + 1):
        train(epoch, model, data_loader, optimizer, kld_weight)

    # Save the trained model
    save_model(model, "vae_model.pth")

if __name__ == "__main__":
    main()
