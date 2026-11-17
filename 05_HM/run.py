"""Evaluate the paper's deterministic historical-mean baseline."""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "06_L-TGAT"))

import numpy as np
from protocol import Topic, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--topics", nargs="+", choices=("A", "B", "C", "D"), default=["A", "B", "C", "D"])
    options = parser.parse_args()
    topics = {name: Topic(options.data / name) for name in dict.fromkeys(options.topics)}
    options.out.mkdir(parents=True, exist_ok=False)
    summaries = []
    with (options.out / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("topic", "prediction_time_unix", "actual", "predicted"))
        for name, topic in topics.items():
            threshold = float(np.median([w.target for w in topic.windows if w.split == "train"]))
            test = [w for w in topic.windows if w.split == "test"]
            writer.writerows((name, w.time, w.target, w.historical_mean) for w in test)
            summaries.append({
                "topic": name, "method": "HM", "threshold": threshold, "deterministic": True,
                **metrics([w.target for w in test], [w.historical_mean for w in test], threshold),
            })
    (options.out / "metrics.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
