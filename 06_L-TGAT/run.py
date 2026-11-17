"""Train L-TGAT end to end and evaluate the best validation checkpoint."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np

from protocol import Topic, metrics


DEFAULT_SEEDS = (42, 1024, 2024, 3407, 4096)
DEFAULT_FANOUT = 15


def sample_neighbors(follows, fanout, layers, rng):
    """Sample incoming neighbors without replacement, independently for each layer."""
    import torch

    groups = {}
    for src, dst in follows.tolist():
        groups.setdefault(dst, []).append(src)
    return [
        torch.tensor(
            [(src, dst) for dst, candidates in groups.items()
             for src in rng.sample(candidates, min(fanout, len(candidates)))],
            dtype=torch.long,
        ).reshape(-1, 2)
        for _ in range(layers)
    ]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_predictions(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_topic(topic, name, options, out):
    import torch
    from torch.nn.functional import mse_loss

    from model import LTGAT, fit_loyalty, prepare

    windows = {split: [w for w in topic.windows if w.split == split]
               for split in ("train", "validation", "test")}
    if not options.allow_incomplete_input and (not topic.content_dim or not topic.follows):
        raise ValueError(f"{name}: paper-mode run requires timestamped content and follower inputs")
    graphs = {split: [topic.graph(w) for w in records] for split, records in windows.items()}
    train_targets = np.array([w.target for w in windows["train"]])
    target_scale = max(float(train_targets.std()), 1e-6)
    target_mean = float(train_targets.mean())
    threshold = float(np.median(train_targets))
    rows = []
    summaries = []
    for seed in options.seeds:
        torch.manual_seed(seed)
        rng = random.Random(seed)
        model = LTGAT(topic.content_dim, options.hidden, options.heads, options.layers, options.variant)
        optimizer = torch.optim.Adam(model.parameters(), lr=options.lr)
        transform = fit_loyalty(graphs["train"], seed)
        prepared = {split: [prepare(graph, transform) for graph in gs] for split, gs in graphs.items()}
        labels = {
            split: torch.tensor([(w.target - target_mean) / target_scale for w in ws], dtype=torch.float32)
            for split, ws in windows.items()
        }
        # Fix validation/test sampling so checkpoint selection is comparable across epochs.
        heldout = {}
        for split in ("validation", "test"):
            heldout_rng = random.Random(seed)
            heldout[split] = [
                sample_neighbors(g[2], options.fanout, options.layers, heldout_rng) for g in prepared[split]
            ]

        def predict(split):
            with torch.no_grad():
                return torch.stack([model(g, neighbors)
                                    for g, neighbors in zip(prepared[split], heldout[split])])

        history, best, best_state, best_epoch = [], float("inf"), None, 0
        stop_reason = "epoch_limit"
        train_indices = list(range(len(windows["train"])))
        for epoch in range(1, options.epochs + 1):
            model.train()
            rng.shuffle(train_indices)
            total = 0.0
            for start in range(0, len(train_indices), options.batch_size):
                indices = train_indices[start:start + options.batch_size]
                optimizer.zero_grad()
                for i in indices:
                    graph = prepared["train"][i]
                    neighbors = sample_neighbors(graph[2], options.fanout, options.layers, rng)
                    loss = mse_loss(model(graph, neighbors), labels["train"][i])
                    (loss / len(indices)).backward()
                    total += float(loss.detach())
                optimizer.step()
            model.eval()
            validation = float(mse_loss(predict("validation"), labels["validation"]))
            if not np.isfinite(validation):
                raise RuntimeError(f"Non-finite validation loss for {name}, seed {seed}")
            history.append({"epoch": epoch, "train_mse_scaled": total / len(train_indices),
                            "validation_mse_scaled": validation})
            if validation < best:
                best, best_epoch = validation, epoch
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            if options.patience and epoch - best_epoch >= options.patience:
                stop_reason = "early_stopping"
                break

        model.load_state_dict(best_state)
        prediction = predict("test").numpy() * target_scale + target_mean
        test = windows["test"]
        score = metrics([w.target for w in test], prediction, threshold)
        summaries.append({"topic": name, "variant": options.variant, "seed": seed,
                          "threshold": threshold, "best_epoch": best_epoch,
                          "epochs_run": len(history), "stop_reason": stop_reason,
                          "best_validation_mse_scaled": best, **score})
        rows.extend({"topic": name, "variant": options.variant, "seed": seed,
                     "prediction_time_unix": w.time, "actual": w.target, "predicted": float(p)}
                    for w, p in zip(test, prediction))
        write_json(out / f"seed_{seed}_training.json", history)
        torch.save(best_state, out / f"seed_{seed}_model.pt")
    write_predictions(out / "predictions.csv", rows)
    write_json(out / "runs.json", summaries)
    aggregate = {}
    for key in ("Accuracy", "F1", "MSE", "RMSE", "R2"):
        values = [row[key] for row in summaries if row[key] is not None]
        aggregate[key] = {"mean": float(np.mean(values)) if values else None,
                          "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}
    write_json(out / "summary.json", aggregate)
    write_json(out / "windows.json", [asdict(w) for w in topic.windows])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Each topic produces test predictions, per-seed metrics and training history, "
               "best model weights, and a mean and sample standard deviation summary.",
    )
    parser.add_argument("--data", type=Path, required=True, help="Parent directory containing topic folders")
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--topics", nargs="+", choices=("A", "B", "C", "D"), default=["A", "B", "C", "D"])
    parser.add_argument("--variant", choices=("full", "raw-only", "no-behavior", "no-graphsage", "no-tgat"),
                        default="full")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="Training seeds (default: 42 1024 2024 3407 4096)")
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--epochs", type=int, default=256,
                        help="Maximum training epochs (reconstruction default: 256)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Stop after this many epochs without lower validation MSE (0 disables)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--fanout", type=int, default=DEFAULT_FANOUT,
                        help="GraphSAGE neighbor cap per node per layer (default: 15). "
                             "Keep all available neighbors at or below the cap.")
    parser.add_argument("--allow-incomplete-input", action="store_true")
    options = parser.parse_args()
    if (min(options.epochs, options.batch_size, options.hidden, options.heads,
            options.layers, options.fanout) < 1 or options.patience < 0
            or not np.isfinite(options.lr) or options.lr <= 0
            or options.hidden % options.heads or len(set(options.seeds)) != len(options.seeds)
            or any(not 0 <= seed < 2**32 for seed in options.seeds)):
        parser.error("Invalid model size, training settings or seed list")
    topics = {name: Topic(options.data / name) for name in dict.fromkeys(options.topics)}
    options.out.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(options).items()}
    config["optimizer"] = "Adam"
    config["checkpoint_monitor"] = "validation_mse_scaled"
    write_json(options.out / "config.json", config)
    for name, topic in topics.items():
        out = options.out / name
        out.mkdir()
        run_topic(topic, name, options, out)
        print(f"{name}: {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
