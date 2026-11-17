"""L-TGAT architecture reconstructed from Sections IV-A through IV-C."""

import math

import numpy as np
import torch
from torch import nn
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

VARIANTS = ("full", "raw-only", "no-behavior", "no-graphsage", "no-tgat")


def fit_loyalty(graphs, seed):
    vectors = np.concatenate([graph[0] for graph in graphs])
    if len(vectors) < 3 or len(np.unique(vectors, axis=0)) < 3:
        raise ValueError("Training history needs at least three distinct behavioral vectors")
    scaler = StandardScaler().fit(vectors)
    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=seed)
    gmm.fit(scaler.transform(vectors))
    order = np.argsort(-gmm.means_.sum(axis=1), kind="stable")
    scores = np.empty(3)
    scores[order] = [1.0, 0.5, 0.1]
    return scaler, gmm, scores


def prepare(graph, transforms):
    x, follows, interactions, content = graph
    if len(x):
        scaler, gmm, scores = transforms
        x = scaler.transform(x)
        loyalty = gmm.predict_proba(x) @ scores
    else:
        loyalty = np.empty(0)
    follow_edges = torch.tensor(follows, dtype=torch.long).reshape(-1, 2)
    edges = torch.tensor([event[:2] for event in interactions], dtype=torch.long).reshape(-1, 2)
    ages = torch.tensor([event[2] for event in interactions], dtype=torch.float32)
    return (torch.tensor(x, dtype=torch.float32), torch.tensor(loyalty[:, None], dtype=torch.float32),
            follow_edges, edges, ages, torch.as_tensor(content))


class MeanSAGE(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.update = nn.Linear(2 * hidden, hidden)

    def forward(self, hidden, edges):
        src, dst = edges.T
        totals = torch.zeros_like(hidden).index_add_(0, dst, hidden[src])
        counts = hidden.new_zeros((len(hidden), 1))
        counts.index_add_(0, dst, hidden.new_ones((len(dst), 1)))
        return self.update(torch.cat((hidden, totals / counts.clamp_min(1)), dim=1)).relu()


class TimeAttention(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        if heads < 1 or hidden % heads:
            raise ValueError("hidden size must be divisible by the number of heads")
        self.heads = heads
        self.width = hidden // heads
        self.time = nn.Linear(1, hidden, bias=False)
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, hidden)

    def forward(self, hidden, edges, ages, temporal):
        node_ids = torch.arange(len(hidden), device=hidden.device)
        src = torch.cat((edges[:, 0], node_ids))
        dst = torch.cat((edges[:, 1], node_ids))
        ages = torch.cat((ages, ages.new_zeros(len(hidden))))
        neighbors = hidden[src]
        if temporal:
            neighbors = neighbors + self.time(ages[:, None]).sin()
        query = self.query(hidden)[dst].reshape(-1, self.heads, self.width)
        key = self.key(neighbors).reshape_as(query)
        value = self.value(neighbors).reshape_as(query)
        logits = (query * key).sum(dim=-1) / math.sqrt(self.width)
        index = dst[:, None].expand_as(logits)
        maxima = logits.new_full((len(hidden), self.heads), -torch.inf)
        maxima.scatter_reduce_(0, index, logits.detach(), reduce="amax", include_self=True)
        exp = (logits - maxima[dst]).exp()
        denominator = logits.new_zeros(maxima.shape).scatter_add_(0, index, exp)
        weighted = value * (exp / denominator[dst]).unsqueeze(-1)
        aggregate = value.new_zeros((len(hidden), self.heads, self.width))
        aggregate.index_add_(0, dst, weighted)
        return (hidden + self.output(aggregate.flatten(1))).relu()


class LTGAT(nn.Module):
    def __init__(self, content_dim, hidden, heads, layers, variant):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        self.variant = variant
        self.node_input = nn.Linear(4, hidden)
        self.sage = nn.ModuleList(MeanSAGE(hidden) for _ in range(layers))
        self.influence = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.loyalty_feature = nn.Linear(1, hidden, bias=False)
        self.influence_feature = nn.Linear(1, hidden, bias=False)
        self.base_feature = nn.Linear(3, hidden, bias=False)
        self.fusion = nn.Linear(hidden, 1, bias=False)
        self.attention = TimeAttention(hidden, heads)
        self.readout = nn.Sequential(nn.Linear(hidden + content_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, graph, neighbor_sets):
        x, loyalty, _, temporal_edges, ages, content = graph
        if self.variant in {"raw-only", "no-behavior"}:
            loyalty = torch.zeros_like(loyalty)
        if self.variant == "no-behavior":
            x = torch.zeros_like(x)
        hidden = self.node_input(torch.cat((x, loyalty), dim=1)).relu()
        if len(x):
            if self.variant != "no-graphsage":
                for layer, edges in zip(self.sage, neighbor_sets):
                    hidden = layer(hidden, edges)
            influence = self.influence(hidden)
            combined = .6 * loyalty + .4 * influence
            features = torch.stack((self.loyalty_feature(loyalty),
                                    self.influence_feature(combined), self.base_feature(x)), dim=1)
            logits = self.fusion(features).squeeze(-1)
            if self.variant in {"raw-only", "no-behavior"}:
                logits[:, 0] = -torch.inf
            fused = (features * logits.softmax(dim=1).unsqueeze(-1)).sum(dim=1)
            embedding = self.attention(fused, temporal_edges, ages, self.variant != "no-tgat").mean(0)
        else:
            embedding = hidden.new_zeros(hidden.shape[1])
        return self.readout(torch.cat((embedding, content))).squeeze(-1)
