from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

MAX_LEN = 32
PAD, UNK = 0, 1


def build_vocab(captions_per_image, min_count=2):
    """Word-level vocabulary from every caption of the training images (lowercase, whitespace split)."""
    counts = Counter()
    for captions in captions_per_image:
        for caption in captions:
            counts.update(caption.lower().split())
    vocab = {"<PAD>": PAD, "<UNK>": UNK}
    for word, count in counts.items():
        if count >= min_count:
            vocab[word] = len(vocab)
    return vocab


def encode(caption, vocab):
    ids = [vocab.get(w, UNK) for w in caption.lower().split()][:MAX_LEN]
    mask = [1] * len(ids) + [0] * (MAX_LEN - len(ids))
    return torch.tensor(ids + [PAD] * (MAX_LEN - len(ids))), torch.tensor(mask)


class CachedFlickr8k(Dataset):
    """Images resized to 224x224 once and kept as uint8; one caption is sampled per call when training.

    Same preprocessing as the notebook (resize, then divide by 255). In evaluation mode caption_0 is used.
    """

    def __init__(self, images_uint8, captions_per_image, vocab, training, seed=0):
        self.images = images_uint8
        self.captions = captions_per_image
        self.vocab = vocab
        self.training = training
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        k = int(self.rng.randint(0, 5)) if self.training else 0
        ids, mask = encode(self.captions[index][k], self.vocab)
        return self.images[index].float() / 255.0, ids, mask


def load_split(split):
    """Resize every image of a Flickr8k split once. Returns (uint8 tensor [N,3,224,224], list of 5 captions)."""
    images, captions = [], []
    for sample in split:
        image = sample["image"].convert("RGB").resize((224, 224))
        images.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1))
        captions.append([sample[f"caption_{i}"] for i in range(5)])
    return torch.stack(images), captions


def load_flickr8k():
    from datasets import load_dataset

    return load_dataset("intro/flickr8k")
