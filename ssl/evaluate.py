import torch
import torchvision
from utils_models import Generator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
latent_dim = 100
num_images = 100

# Load the saved checkpoint
checkpoint = torch.load('../Results/Model/CIFAR10_16_42_Drichlet_0.600/FedDC_0.1_S10_F0.250000_Lr0.000200_10_0.500000_B64_E6_W0.000100_a0.100000_seed42_lrdecay1.000000/checkpoint_round_500.pt', map_location=torch.device('cpu'))

print(type(checkpoint))
if isinstance(checkpoint, dict):
    print("Top-level keys:", list(checkpoint.keys())[:20])
    # If there is a nested state_dict:
    if "state_dict" in checkpoint:
        print("state_dict keys:", list(checkpoint["state_dict"].keys())[:20])
    else:
        print("checkpoint keys:", list(checkpoint.keys())[:20])


# Pick generator weights from the checkpoint wrapper
state = checkpoint['model_G_state'] if isinstance(checkpoint, dict) and 'model_G_state' in checkpoint else checkpoint

# Infer how many class channels the checkpoint expects
in_ch = state['main.0.weight'].shape[0]
num_classes = max(0, in_ch - latent_dim)

dataset_name = 'CIFAR10'  # change if needed
generator = Generator(dataset_name, z_dim=latent_dim, num_classes=num_classes).to(device)

generator.load_state_dict(state, strict=False)
generator.eval()

with torch.no_grad():
    z = torch.randn(num_images, latent_dim, 1, 1).to(device)

    if num_classes > 0:
        # sample random class labels
        y = torch.randint(0, num_classes, (num_images,), device=device)
        fake_images = generator(z, y)
    else:
        fake_images = generator(z)

# Debug stats
print("fake_images stats:",
      fake_images.min().item(),
      fake_images.max().item(),
      fake_images.mean().item())

# Convert tensor to CPU, clamp, and map from [-1,1] to [0,1]
vis = fake_images[:100].detach().cpu()
vis = (vis + 1) / 2
vis = vis.clamp(0, 1)

grid = torchvision.utils.make_grid(vis, nrow=10)
np_img = grid.permute(1, 2, 0).numpy()
torchvision.utils.save_image(grid, 'generated_10_images.png')


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
plt.imshow(np_img)
plt.axis('off')
plt.show()