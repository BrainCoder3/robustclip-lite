import random

import numpy as np
import torch

from .corruption import apply_noise_randomly
from .model import RobustCLIP, contrastive_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validation_loss(model, loader, device):
    model.eval()
    total = 0.0
    for images, ids, mask in loader:
        total += contrastive_loss(model(images.to(device), ids.to(device), mask.to(device))).item()
    return total / len(loader)


def train_fixed_epochs(vocab_size, train_loader, val_loader, device, epochs, seed,
                       noise_sigma=0.0, noise_probability=0.0, log=print):
    """Train for exactly `epochs` epochs and return the last state (no checkpoint selection).

    With noise_probability > 0 each training image is corrupted independently by Gaussian noise of std
    `noise_sigma`. Validation is always on clean images.
    """
    set_seed(seed)
    model = RobustCLIP(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for images, ids, mask in train_loader:
            images, ids, mask = images.to(device), ids.to(device), mask.to(device)
            if noise_probability > 0:
                images = apply_noise_randomly(images, noise_sigma, noise_probability)
            optimizer.zero_grad()
            loss = contrastive_loss(model(images, ids, mask))
            loss.backward()
            optimizer.step()
            total += loss.item()
        val = validation_loss(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": total / len(train_loader), "val_loss": val})
        log(f"  seed {seed} epoch {epoch:2d}  train {history[-1]['train_loss']:.4f}  val {val:.4f}")
    return model, history
