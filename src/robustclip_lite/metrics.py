import torch

from .corruption import add_gaussian_noise


def recall_at_k(similarities, k):
    """Row i is correct when column i is among its top-k scores."""
    top = torch.topk(similarities, k=k, dim=1).indices
    target = torch.arange(similarities.shape[0], device=top.device)[:, None]
    return (top == target).any(dim=1).float().mean().item()


@torch.no_grad()
def evaluate_retrieval(model, loader, device, sigma=0.0, noise_seed=42):
    """Retrieval over the whole loader (1000 candidates for dev/test), optional Gaussian noise on images."""
    model.eval()
    torch.manual_seed(noise_seed)
    images_emb, texts_emb = [], []
    for images, ids, mask in loader:
        images, ids, mask = images.to(device), ids.to(device), mask.to(device)
        if sigma > 0:
            images = add_gaussian_noise(images, sigma)
        images_emb.append(model.embed_images(images).cpu())
        texts_emb.append(model.embed_texts(ids, mask).cpu())
    sim = torch.cat(images_emb) @ torch.cat(texts_emb).T
    return {
        "image_to_text": {f"R@{k}": recall_at_k(sim, k) for k in (1, 5, 10)},
        "text_to_image": {f"R@{k}": recall_at_k(sim.T, k) for k in (1, 5, 10)},
    }
