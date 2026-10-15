from torchvision import datasets
import torch
import numpy as np

def load_mnist(subset_size, train, data_dir, seed=42):
    dataset = datasets.MNIST(root=data_dir, train=train, download=True)

    # Convert to (N, )
    images = dataset.data.float().div(255.0).unsqueeze(1)
    labels = dataset.targets.long()

    n = len(images)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=subset_size, replace=subset_size > n)
    idx = torch.from_numpy(idx)
    images, labels = images[idx], labels[idx]

    return images, labels