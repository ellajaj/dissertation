import os
import torch
import torch.nn as nn
from torchvision.utils import save_image
from gan_model import Generator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

total_images = 100
batch_size = 100 
num_batches = total_images // batch_size
latent_dim = 100
num_classes = 10

generator = Generator(latent_dim, num_classes).to(device) 
generator.load_state_dict(torch.load("generator.pth"))
generator.eval()

os.makedirs("train_imgs", exist_ok=True)

image_count = 0

with torch.no_grad():
    for batch_idx in range(num_batches):
        # Sample noise
        z = torch.randn(batch_size, latent_dim, 1, 1).to(device)

        # Sample class labels
        labels = torch.randint(0, num_classes, (batch_size,)).to(device)
        labels_one_hot = nn.functional.one_hot(labels, num_classes).float()

        # Concatenate noise + labels
        z = torch.cat([z, labels_one_hot.unsqueeze(-1).unsqueeze(-1)], dim=1)

        # Generate images
        fake_images = generator(z)

        # Save each image individually
        for i in range(batch_size):
            save_image(
                fake_images[i],
                f"train_imgs/img_{image_count:05d}.png",
                normalize=True
            )
            image_count += 1

        print(f"Generated {image_count}/{total_images}")
