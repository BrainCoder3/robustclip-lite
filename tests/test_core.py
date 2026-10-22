import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robustclip_lite.corruption import add_gaussian_noise, apply_noise_randomly  # noqa: E402
from robustclip_lite.data import build_vocab, encode  # noqa: E402
from robustclip_lite.metrics import recall_at_k  # noqa: E402
from robustclip_lite.model import RobustCLIP, contrastive_loss  # noqa: E402


class RecallTest(unittest.TestCase):
    def test_identity_is_perfect(self):
        self.assertEqual(recall_at_k(torch.eye(20), 1), 1.0)

    def test_known_rank(self):
        sim = torch.zeros(4, 4)
        sim[torch.arange(4), torch.arange(4)] = 1.0
        sim[0, 0], sim[0, 1], sim[0, 2] = 0.1, 0.9, 0.8   # true item of row 0 is third
        self.assertEqual(recall_at_k(sim, 1), 0.75)
        self.assertEqual(recall_at_k(sim, 2), 0.75)
        self.assertEqual(recall_at_k(sim, 3), 1.0)

    def test_transpose_direction(self):
        sim = torch.tensor([[0.9, 0.1], [0.8, 0.2]])      # row 1 prefers column 0, but column 1 prefers row 1
        self.assertEqual(recall_at_k(sim, 1), 0.5)         # image -> text
        self.assertEqual(recall_at_k(sim.T, 1), 1.0)       # text -> image


class NoiseTest(unittest.TestCase):
    def test_range_and_zero_sigma(self):
        x = torch.rand(8, 3, 16, 16)
        self.assertTrue(torch.equal(add_gaussian_noise(x, 0.0), x))
        y = add_gaussian_noise(x, 0.5)
        self.assertTrue(y.min() >= 0 and y.max() <= 1)

    def test_probability_extremes_and_rate(self):
        x = torch.rand(2000, 3, 4, 4)
        self.assertTrue(torch.equal(apply_noise_randomly(x, 0.2, 0.0), x))
        changed = (apply_noise_randomly(x, 0.2, 1.0) != x).flatten(1).any(dim=1).float().mean()
        self.assertGreater(changed, 0.99)
        torch.manual_seed(0)
        half = (apply_noise_randomly(x, 0.2, 0.5) != x).flatten(1).any(dim=1).float().mean().item()
        self.assertAlmostEqual(half, 0.5, delta=0.05)


class DataTest(unittest.TestCase):
    def test_vocab_threshold_and_unknown(self):
        vocab = build_vocab([["a dog runs", "a dog"], ["a cat"]])
        self.assertIn("a", vocab)
        self.assertIn("dog", vocab)
        self.assertNotIn("cat", vocab)            # appears once, below min_count=2
        ids, mask = encode("A cat dog", vocab)
        self.assertEqual(ids[1].item(), 1)         # unknown word
        self.assertEqual(mask.sum().item(), 3)
        self.assertEqual(len(ids), 32)

    def test_truncation(self):
        vocab = {"<PAD>": 0, "<UNK>": 1, "a": 2}
        ids, mask = encode(" ".join(["a"] * 50), vocab)
        self.assertEqual(mask.sum().item(), 32)


class ModelTest(unittest.TestCase):
    def test_shapes_and_loss_decreases(self):
        torch.manual_seed(0)
        model = RobustCLIP(vocab_size=50)
        images = torch.rand(8, 3, 224, 224)
        ids = torch.randint(2, 50, (8, 32))
        mask = torch.ones(8, 32, dtype=torch.long)
        logits = model(images, ids, mask)
        self.assertEqual(logits.shape, (8, 8))
        norms = model.embed_images(images).norm(dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones(8), atol=1e-5))
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        first = contrastive_loss(logits).item()
        for _ in range(15):
            opt.zero_grad()
            loss = contrastive_loss(model(images, ids, mask))
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), first)


if __name__ == "__main__":
    unittest.main()
