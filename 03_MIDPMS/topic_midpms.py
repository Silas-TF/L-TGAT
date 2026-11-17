"""Use the MIDPMS source layers with observed topic windows and item histories."""

from bisect import bisect_left
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from cascade_inputs import hourly_actions
from protocol import ACTIONS, HISTORY, HORIZON, rows, timestamp


source_path = Path(__file__).resolve().parent / "adapted" / "newcode" / "model.py"
spec = importlib.util.spec_from_file_location("midpms_source", source_path)
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def prepare(topic, windows, options, folder):
    if not topic.content_dim:
        raise ValueError(f"{folder}: MIDPMS needs dated focal-topic features in content.csv")
    path = folder / "item_history.csv"
    if not path.is_file():
        raise ValueError(f"{path}: supply observed user/item histories and content vectors")
    history = defaultdict(list)
    seen = {}
    for row in rows(path, ("user_id", "item_id", "known_at", "features")):
        user, item = row["user_id"], row["item_id"]
        time = timestamp(row["known_at"])
        vector = np.asarray(json.loads(row["features"]), dtype=np.float32)
        if (not user or not item or vector.shape != (topic.content_dim,)
                or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0):
            raise ValueError(f"{path}: invalid user, item, or content vector")
        key = user, item, time
        if key in seen:
            if not np.array_equal(seen[key], vector):
                raise ValueError(f"{path}: conflicting record {key}")
            continue
        seen[key] = vector
        history[user].append((time, item, vector / np.linalg.norm(vector)))
    for entries in history.values():
        entries.sort(key=lambda row: row[:2])
    history_times = {user: [row[0] for row in entries] for user, entries in history.items()}
    audit = {split: {"windows": 0, "windows_with_matched_history": 0, "matched_candidates": 0}
             for split in ("train", "validation", "test")}
    auxiliary_labels = 0
    result = []
    for window in windows:
        cutoff = window.time
        index = bisect_left(topic.content_times, cutoff) - 1
        if index < 0:
            raise ValueError(f"{folder}: no focal-topic content before {cutoff}")
        focal = topic.content[index][1]
        focal_norm = np.linalg.norm(focal)
        if focal_norm == 0:
            raise ValueError(f"{folder}: zero focal-topic vector at {cutoff}")
        focal = focal / focal_norm
        observed = topic.events_between(cutoff - HISTORY, cutoff)
        candidates = list(dict.fromkeys(
            event[1] for event in reversed(observed) if event[2] in ACTIONS
        ))[:options.candidates]
        scores = np.zeros(options.candidates, dtype=np.float32)
        mask = np.zeros(options.candidates, dtype=bool)
        for i, user in enumerate(candidates):
            end = bisect_left(history_times.get(user, ()), cutoff)
            entries = history.get(user, ())
            latest_items = set()
            for j in range(end - 1, -1, -1):
                _, item, vector = entries[j]
                if item not in latest_items:
                    scores[i] += vector @ focal
                    latest_items.add(item)
                if len(latest_items) == options.items_per_user:
                    break
            mask[i] = bool(latest_items)
        label = -1
        matched = int(mask.sum())
        audit[window.split]["windows"] += 1
        audit[window.split]["windows_with_matched_history"] += int(matched > 0)
        audit[window.split]["matched_candidates"] += matched
        if window.split == "train":
            next_actor = next(
                (event[1] for event in topic.events_between(cutoff, cutoff + HORIZON)
                 if event[2] in ACTIONS), None,
            )
            if next_actor in candidates and mask[candidates.index(next_actor)]:
                label = candidates.index(next_actor)
                auxiliary_labels += 1
        counts = hourly_actions(topic, cutoff)
        sequence = counts.sum(1) / 3
        result.append((torch.from_numpy(np.log1p(sequence)),
                       torch.from_numpy(np.log1p(counts.sum(0))),
                       torch.from_numpy(scores), torch.from_numpy(mask), label))
    if not audit["train"]["matched_candidates"]:
        raise ValueError(f"{path}: no dated histories match any training candidate")
    if options.midpms_aux_weight and not auxiliary_labels:
        raise ValueError(f"{path}: no observed next actor matches a training candidate with history")
    return result, {"partitions": audit, "training_next_actor_labels": auxiliary_labels}


class TopicMIDPMS(source.MSMfusion):
    def __init__(self, options):
        super().__init__(user_num=options.candidates, sequence_len=72, hidden_size=options.hidden,
                         rnn_num_layers=2, pre_train=False,
                         opt=SimpleNamespace(content_num=1, use_att=False))
        del self.em

    def forward(self, batch):
        sequences, influence, scores, mask, labels = zip(*batch)
        gru_out, hidden = self.popModel(torch.stack(sequences))
        temporal = F.softplus(self.timefactor(hidden))
        mask = torch.stack(mask)
        scores = torch.stack(scores).masked_fill(~mask, torch.finfo(gru_out.dtype).min)
        similarity = scores.softmax(1) * mask
        topology = self.Anticipation(torch.stack(influence))
        branches = torch.cat((temporal, similarity, topology), dim=1)
        substitution = F.softplus(self.linearmsm2(self.norm2(self.linearmsm1(branches))))
        prediction = (self.grulinear(self.norm1(gru_out)).relu() + substitution) / 2
        user_scores = (prediction / temporal.clamp_min(1e-8)) * similarity
        labels = torch.tensor(labels, dtype=torch.long)
        has_label = labels >= 0
        auxiliary = prediction.new_zeros(())
        if self.training and has_label.any():
            logits = user_scores.masked_fill(~mask, torch.finfo(user_scores.dtype).min)
            auxiliary = F.cross_entropy(logits[has_label], labels[has_label])
        return prediction[:, 0], auxiliary
