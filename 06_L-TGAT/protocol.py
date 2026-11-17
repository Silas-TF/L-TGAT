"""Construct chronological topic samples from observed, timestamped records."""

from bisect import bisect_left
from collections import defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime
import json

import numpy as np

HOUR = 3600
HISTORY = 72 * HOUR
HORIZON = 24 * HOUR
ACTIONS = {"like", "comment", "repost"}


def timestamp(value):
    date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if date.utcoffset() is None:
        raise ValueError(f"Timestamp must include a time zone: {value}")
    return date.timestamp()


def rows(path, required):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: expected columns {', '.join(required)}")
        yield from reader


@dataclass(frozen=True)
class Window:
    time: float
    split: str
    target: float
    historical_mean: float


class Topic:
    def __init__(self, folder):
        cfg = json.loads((folder / "period.json").read_text(encoding="utf-8-sig"))
        start, validation, test, end = (
            timestamp(cfg[key]) for key in ("start", "validation_start", "test_start", "end")
        )
        if not start + HISTORY < validation < test < end:
            raise ValueError("Invalid coverage or partition boundaries")
        self.start = start
        self.events = []
        seen = {}
        for row in rows(folder / "events.csv",
                        ("event_id", "timestamp", "actor_id", "kind", "target_user_id", "endorsement")):
            time = timestamp(row["timestamp"])
            kind, actor, target = row["kind"], row["actor_id"], row["target_user_id"]
            endorsement = row["endorsement"]
            if kind not in ACTIONS | {"post"} or not row["event_id"] or not actor:
                raise ValueError(f"Invalid event: {row['event_id']}")
            if endorsement not in {"0", "1"} or (endorsement == "1" and kind not in {"comment", "repost"}):
                raise ValueError(f"Invalid endorsement flag: {row['event_id']}")
            event = (time, actor, kind, target, int(endorsement))
            event_id = row["event_id"]
            if event_id in seen and seen[event_id] != event:
                raise ValueError(f"Conflicting records for event {event_id}")
            if event_id not in seen and start <= time < end:
                self.events.append(event)
            seen[event_id] = event
        self.events.sort()
        self.times = [event[0] for event in self.events]
        action_times = [event[0] for event in self.events if event[2] in ACTIONS]

        def popularity(left, right):
            return (bisect_left(action_times, right) - bisect_left(action_times, left)) / 3

        self.windows = []
        cutoff = start + HISTORY
        while cutoff + HORIZON <= end:
            split = "train" if cutoff < validation else "validation" if cutoff < test else "test"
            boundary = {"train": validation, "validation": test, "test": end}[split]
            if cutoff + HORIZON <= boundary:
                history = [
                    popularity(cutoff - i * HORIZON, cutoff - (i - 1) * HORIZON)
                    for i in (1, 2, 3)
                ]
                self.windows.append(Window(
                    cutoff, split, popularity(cutoff, cutoff + HORIZON), sum(history) / 3
                ))
            cutoff += HOUR
        for split in ("train", "validation", "test"):
            if not any(w.split == split for w in self.windows):
                raise ValueError(f"No complete windows in {split}: {folder}")

        follow_file = folder / "follows.csv"
        self.follows = defaultdict(dict)
        if follow_file.exists():
            for row in rows(follow_file, ("follower_id", "followee_id", "known_at")):
                actor, target = row["follower_id"], row["followee_id"]
                if not actor or not target:
                    raise ValueError("Follow edge has an empty endpoint")
                time = timestamp(row["known_at"])
                self.follows[actor][target] = min(time, self.follows[actor].get(target, time))

        content_file = folder / "content.csv"
        self.content = []
        if content_file.exists():
            for row in rows(content_file, ("known_at", "features")):
                vector = np.asarray(json.loads(row["features"]), dtype=np.float32)
                if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
                    raise ValueError("Expected a finite one-dimensional content vector")
                self.content.append((timestamp(row["known_at"]), vector))
            self.content.sort(key=lambda item: item[0])
            if not self.content or len({v.size for _, v in self.content}) != 1:
                raise ValueError("Content feature dimensions must agree")
            self.content_times = [t for t, _ in self.content]
        self.content_dim = self.content[0][1].size if self.content else 0

    def events_between(self, left, right):
        return self.events[bisect_left(self.times, left):bisect_left(self.times, right)]

    def graph(self, window, *, include_context=True):
        cutoff = window.time
        history = self.events_between(cutoff - HISTORY, cutoff)
        by_user = defaultdict(list)
        for event in history:
            _, actor, kind, target, _ = event
            by_user[actor].append(event)
            if target and kind in ACTIONS:
                by_user.setdefault(target, [])
        users = sorted(by_user)
        node = {user: i for i, user in enumerate(users)}
        features = []
        for user in users:
            events = by_user[user]
            features.append([
                sum(event[4] for event in events),
                sum(event[2] in ACTIONS for event in events),
                (events[-1][0] - events[0][0]) / HOUR if events else 0,
            ])
        interactions = [
            (node[actor], node[target], (cutoff - time) / HOUR)
            for time, actor, kind, target, _ in history
            if kind in ACTIONS and target and actor != target
        ]
        follows = []
        if include_context:
            # Followers and followees are separate from timestamped interaction edges.
            follows = [
                (node[user], node[other])
                for user in users
                for other, known in self.follows.get(user, {}).items()
                if other in node and known < cutoff and other != user
            ]
        if include_context and self.content:
            index = bisect_left(self.content_times, cutoff) - 1
            if index < 0:
                raise ValueError("Content vector must be observable before each prediction")
            content = self.content[index][1]
        else:
            content = np.empty(0, dtype=np.float32)
        return np.asarray(features, dtype=np.float32).reshape(-1, 3), follows, interactions, content


def metrics(actual, predicted, threshold):
    y, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or not y.size or not np.isfinite([y, p]).all():
        raise ValueError("Metrics require equally sized, finite vectors")
    error = np.square(y - p)
    sst = np.square(y - y.mean()).sum()
    a, b = y >= threshold, p >= threshold
    tp = np.count_nonzero(a & b)
    positive_count = np.count_nonzero(a) + np.count_nonzero(b)
    mse = float(error.mean())
    return {
        "Accuracy": float((a == b).mean()),
        "F1": float(2 * tp / positive_count) if positive_count else 0.0,
        "MSE": mse,
        "RMSE": mse ** 0.5,
        "R2": float(1 - error.sum() / sst) if sst else None,
    }
