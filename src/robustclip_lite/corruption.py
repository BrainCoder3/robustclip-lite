import torch


def add_gaussian_noise(images, sigma):
    """images in [0, 1]; noise is added then clipped back to [0, 1]."""
    return torch.clamp(images + sigma * torch.randn_like(images), 0, 1)


def apply_noise_randomly(images, sigma, probability):
    """Corrupt each image of the batch independently with the given probability."""
    corrupt = (torch.rand(images.shape[0], device=images.device) < probability)[:, None, None, None]
    return torch.where(corrupt, add_gaussian_noise(images, sigma), images)
