''' taken from https://github.com/rohanreddy1201/CIFAR10-Image-Generator-using-GANs/tree/main -- MIT license'''

import torch
import torchvision
from gan_model import Generator

# Define device configuration (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
latent_dim = 100
num_images = 100

# Load the saved checkpoint
checkpoint = torch.load('generator.pth', map_location=torch.device('cpu'))

# Adjust the dimensions of the layers
checkpoint['main.0.weight'] = checkpoint['main.0.weight'][:latent_dim]  # Adjust the number of input channels

# Create a new instance of the Generator model
generator = Generator(latent_dim).to(device)

# Load the adjusted state dictionary
generator.load_state_dict(checkpoint)

# Set the generator to evaluation mode
generator.eval()

# Generate new images
with torch.no_grad():
    z = torch.randn(num_images, latent_dim, 1, 1).to(device)
    fake_images = generator(z)

# Save generated images
torchvision.utils.save_image(fake_images, 'generated_images.png', nrow=10, normalize=True)

# Visualize generated images
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
plt.imshow(plt.imread('generated_images.png'))
plt.axis('off')
plt.show()