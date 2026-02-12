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

data_dir = "../ssl/Folder/Data/Raw/cifar-10-batches-py"
data_batches, test_batch = load_cifar10_data(data_dir)