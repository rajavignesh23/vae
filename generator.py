import os
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from torch_tut import ConvVAE  # Import the ConvVAE architecture

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load a saved model
def load_model(path, latent_dim=20):
    model = ConvVAE(latent_dim).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model

# Generate and display images
def generate_images(model, num_images=5):
    # Ensure the directory exists
    
    with torch.no_grad():
        for i in range(num_images):
            z = torch.randn(1, 20).to(device)  # Latent vector
            sample = model.decode(z).cpu()
            sample = sample.view(3, 170, 200).permute(1, 2, 0)  # Reshape to (170, 200, 3)
            sample = sample.numpy()
            sample = sample.clip(0, 1)

            # Display the generated image
            plt.figure(figsize=(5, 5))
            plt.imshow(sample)
            plt.axis('off')  # Hide axis
            plt.title(f'Generated Image {i + 1}')  # Title for the image
            plt.show()  # Show the image

if __name__ == "__main__":
    model = load_model("vae_model.pth")  # Load the saved model
    generate_images(model, num_images=5)  # Generate and display images
