import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
import torch.nn.functional as F
from torch.utils.data import DataLoader

def get_features(images, model, device):
    model.eval()
    with torch.no_grad():
        # Resize images to 299x299 as required by InceptionV3
        # CIFAR images are usually 32x32; we upscale them here
        '''if images.shape[2:] != (299, 299):
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        features = model(images.to(device))
        
        # InceptionV3 sometimes returns a tuple (output, aux_output) 
        # during training, but in .eval() it usually returns just the tensor.
        # This squeeze ensures we get a 2D array (batch, features)
        if isinstance(features, tuple):
            features = features[0]'''
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        # Inception also needs 299x299. If not already resized:
        if images.shape[2] < 299:
            images = torch.nn.functional.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)

        features = model(images.to(device))
        
        # If it's a tuple (Inception aux), take the first element
        if isinstance(features, tuple):
            features = features[0]

    #return model(images.to(device))
            
    return features.squeeze().cpu().numpy()

def calculate_stats(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

def precalculate_real_stats(dataset, model, device, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_features = []
    
    for images, _ in loader:
        feat = get_features(images, model, device)
        all_features.append(feat)
    
    all_features = np.concatenate(all_features, axis=0)
    return calculate_stats(all_features)

def compute_fid(generator, model, mu_real, sigma_real, device, num_fake=5000, batch_size=64):
    generator.eval()
    all_features = []
    
    # Generate fake images and get features
    for _ in range(0, num_fake, batch_size):
        z = torch.randn(batch_size, generator.latent_dim, 1, 1).to(device)
        
        # 2. Create labels
        fake_labels = torch.randint(0, generator.num_classes, (batch_size,)).to(device)
        fake_labels_oh = torch.nn.functional.one_hot(fake_labels, generator.num_classes).float().to(device)
        
        # 3. Concatenate (Ensure dimensions match: [batch, channels, 1, 1])
        # We reshape one_hot from [batch, 10] to [batch, 10, 1, 1]
        fake_labels_input = fake_labels_oh.unsqueeze(-1).unsqueeze(-1)
        z_input = torch.cat([z, fake_labels_input], dim=1)
        
        with torch.no_grad():
            fake_ims = generator(z_input)
            feat = get_features(fake_ims, model, device)
            all_features.append(feat)
            
    all_features = np.concatenate(all_features, axis=0)
    mu_fake, sigma_fake = calculate_stats(all_features)
    
    # FID Calculation
    diff = mu_real - mu_fake
    # Matrix square root
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid