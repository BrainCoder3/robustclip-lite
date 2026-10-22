"""Clean vs Gaussian-noise-aware training over several seeds, same fixed budget.

Both models are trained for exactly --epochs epochs and the LAST state is evaluated (no early stopping, no
checkpoint picked on the test set). The test set is touched only once per trained model.

    python experiments/multi_seed_noise.py --seeds 42 43 44
    python experiments/multi_seed_noise.py --fake --epochs 2     # quick check of the code path, no download
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robustclip_lite.data import CachedFlickr8k, build_vocab, load_flickr8k, load_split  # noqa: E402
from robustclip_lite.metrics import evaluate_retrieval  # noqa: E402
from robustclip_lite.train import train_fixed_epochs  # noqa: E402

SIGMAS = [0.0, 0.05, 0.10, 0.20]
CONDITIONS = {
    "clean": dict(noise_probability=0.0),
    "noise_aware": dict(noise_probability=0.5),
}


def fake_split(n, rng):
    images = torch.randint(0, 256, (n, 3, 224, 224), dtype=torch.uint8)
    words = ["a", "dog", "runs", "on", "grass", "man", "woman", "sits", "red", "blue"]
    caps = [[" ".join(rng.choice(words, size=int(rng.integers(3, 8)))) for _ in range(5)] for _ in range(n)]
    return images, caps


def load_data(fake):
    if fake:
        import numpy as np
        rng = np.random.default_rng(0)
        return {"train": fake_split(96, rng), "dev": fake_split(64, rng), "test": fake_split(64, rng)}
    raw = load_flickr8k()
    out = {}
    for name in ("train", "dev", "test"):
        t0 = time.time()
        out[name] = load_split(raw[name])
        print(f"loaded {name}: {len(out[name][0])} images in {time.time() - t0:.0f}s", flush=True)
    return out


def summarize(runs):
    """mean and sample std (n-1) over seeds for every condition, sigma and direction at R@10."""
    summary = {}
    for cond in CONDITIONS:
        summary[cond] = {}
        for sigma in SIGMAS:
            row = {}
            for direction in ("image_to_text", "text_to_image"):
                vals = [r["test"][cond][str(sigma)][direction]["R@10"] for r in runs]
                row[direction] = {"mean": statistics.mean(vals),
                                  "std": statistics.stdev(vals) if len(vals) > 1 else None,
                                  "values": vals}
            summary[cond][str(sigma)] = row
    return summary


def markdown_table(summary, seeds):
    def fmt(c):
        return f"{100 * c['mean']:.1f} +- {100 * c['std']:.1f}" if c["std"] is not None else f"{100 * c['mean']:.1f}"

    lines = [f"R@10 (%), mean +- std over seeds {seeds}", "",
             "| Test sigma | I->T clean | I->T noise-aware | T->I clean | T->I noise-aware |",
             "|---:|---:|---:|---:|---:|"]
    for sigma in SIGMAS:
        s = str(sigma)
        lines.append(f"| {sigma:.2f} | {fmt(summary['clean'][s]['image_to_text'])} | {fmt(summary['noise_aware'][s]['image_to_text'])} "
                     f"| {fmt(summary['clean'][s]['text_to_image'])} | {fmt(summary['noise_aware'][s]['text_to_image'])} |")
    return "\n".join(lines)


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (direction, title) in zip(axes, [("image_to_text", "Image to text"), ("text_to_image", "Text to image")]):
        for cond, label in (("clean", "Clean training"), ("noise_aware", "Noise-aware training")):
            means = [100 * summary[cond][str(s)][direction]["mean"] for s in SIGMAS]
            stds = [100 * (summary[cond][str(s)][direction]["std"] or 0.0) for s in SIGMAS]
            ax.errorbar(SIGMAS, means, yerr=stds, marker="o", capsize=3, label=label)
        ax.set_title(title)
        ax.set_xlabel("Test Gaussian noise sigma")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("R@10 (%)")
    axes[0].legend()
    fig.suptitle("Mean +- std over seeds, same fixed training budget")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=19)
    ap.add_argument("--sigma-train", type=float, default=0.10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="results/multi_seed_noise.json")
    ap.add_argument("--figure", default="results/figures/multi_seed_noise.png")
    ap.add_argument("--fake", action="store_true", help="random data, only to check the code path")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    data = load_data(args.fake)
    vocab = build_vocab(data["train"][1])
    print("vocab size:", len(vocab), flush=True)

    dev_loader = DataLoader(CachedFlickr8k(*data["dev"], vocab, training=False), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(CachedFlickr8k(*data["test"], vocab, training=False), batch_size=args.batch_size, shuffle=False)

    runs = []
    for seed in args.seeds:
        run = {"seed": seed, "history": {}, "test": {}}
        for cond, cfg in CONDITIONS.items():
            t0 = time.time()
            gen = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(CachedFlickr8k(*data["train"], vocab, training=True, seed=seed),
                                      batch_size=args.batch_size, shuffle=True, generator=gen)
            print(f"== seed {seed} / {cond}", flush=True)
            model, history = train_fixed_epochs(len(vocab), train_loader, dev_loader, device, args.epochs, seed,
                                                noise_sigma=args.sigma_train, noise_probability=cfg["noise_probability"],
                                                log=lambda m: print(m, flush=True))
            run["history"][cond] = history
            run["test"][cond] = {str(s): evaluate_retrieval(model, test_loader, device, sigma=s) for s in SIGMAS}
            print(f"   done in {time.time() - t0:.0f}s", flush=True)
            del model
        runs.append(run)

        summary = summarize(runs)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"config": vars(args), "runs": runs, "summary": summary}, indent=2))

    table = markdown_table(summary, args.seeds)
    print("\n" + table)
    Path(args.out).with_suffix(".md").write_text(table + "\n")
    Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    plot(summary, args.figure)
    print("saved", args.out, "and", args.figure)


if __name__ == "__main__":
    main()
