import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.relu(self.conv(x)))


class ImageEncoder(nn.Module):
    """Four conv blocks (3 -> 32 -> 64 -> 128 -> 256), then global average pooling."""

    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.layer = nn.ModuleList([
            ConvBlock(in_ch, base),
            ConvBlock(base, base * 2),
            ConvBlock(base * 2, base * 4),
            ConvBlock(base * 4, base * 8),
        ])
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        for block in self.layer:
            x = block(x)
        return torch.flatten(self.pool(x), start_dim=1)


class TextEncoder(nn.Module):
    """Token + learned positional embeddings, 2-layer Transformer, masked mean pooling."""

    def __init__(self, vocab_size, dim=256, max_len=32, heads=4, ff=1024, layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.positional = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=ff, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, ids, mask):
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.embedding(ids) + self.positional(positions)
        x = self.transformer(x, src_key_padding_mask=(mask == 0))
        mask = mask[:, :, None]
        return (x * mask).sum(dim=1) / mask.sum(dim=1)


class RobustCLIP(nn.Module):
    def __init__(self, vocab_size, feat_dim=256, embed_dim=128):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder(vocab_size, dim=feat_dim)
        self.projection_image = nn.Linear(feat_dim, embed_dim)
        self.projection_text = nn.Linear(feat_dim, embed_dim)
        self.logits_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

    def embed_images(self, images):
        return F.normalize(self.projection_image(self.image_encoder(images)))

    def embed_texts(self, ids, mask):
        return F.normalize(self.projection_text(self.text_encoder(ids, mask)))

    def forward(self, images, ids, mask):
        return self.embed_images(images) @ self.embed_texts(ids, mask).T * torch.exp(self.logits_scale)


def contrastive_loss(logits):
    """Symmetric cross-entropy; the diagonal holds the matching pairs."""
    target = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
