import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_cifar10_batch(file_path):
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
    return data_dict

def load_cifar10_data(data_dir):
    data_batches = []
    for i in range(1, 6):
        batch_file = f"{data_dir}/data_batch_{i}"
        data_batch = load_cifar10_batch(batch_file)
        data_batches.append(data_batch)
    test_batch = load_cifar10_batch(f"{data_dir}/test_batch")
    return data_batches, test_batch

def preprocess_images(data_batches, test_batch):
    train_images = np.vstack([batch[b'data'] for batch in data_batches])
    train_images = train_images.reshape(-1, 3, 32, 32)  # Reshape to (num_samples, channels, height, width)
    train_images = train_images.transpose(0, 2, 3, 1)  # Transpose to (num_samples, height, width, channels)
    train_images = train_images.astype('float32') / 255.0  # Normalize pixel values to [0, 1]

    test_images = test_batch[b'data'].reshape(-1, 3, 32, 32)
    test_images = test_images.transpose(0, 2, 3, 1)
    test_images = test_images.astype('float32') / 255.0

    return train_images, test_images

# Load CIFAR-10 dataset
data_dir = "../ssl/Folder/Data/Raw/cifar-10-batches-py"
data_batches, test_batch = load_cifar10_data(data_dir)

# Preprocess images
train_images, test_images = preprocess_images(data_batches, test_batch)

# Optional: Visualize a few sample images
num_samples = 5
plt.figure(figsize=(10, 5))
for i in range(num_samples):
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(train_images[i])
    plt.axis('off')
plt.show()