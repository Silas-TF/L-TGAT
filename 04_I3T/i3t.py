"""Topic adaptation of Tai et al., DOI 10.1016/j.ins.2023.119705."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence

from cascade_inputs import cascade, walk_context


def prepare(topic, windows, options):
    samples = []
    for window in windows:
        sample = cascade(topic, window, options.walks, options.walk_length)
        context = walk_context(sample.paths, len(sample.nodes))
        samples.append((sample, context))
    return samples


class I3T(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.graph = nn.Linear(4, hidden)
        self.path = nn.LSTM(hidden + 8, hidden, bidirectional=True, batch_first=True)
        self.temporal = nn.GRU(3, hidden, batch_first=True)
        self.key = nn.Linear(2 * hidden, hidden)
        self.output = nn.Linear(3 * hidden, 1)

    def forward(self, prepared):
        sample, walk_embeddings = prepared
        ids = torch.arange(len(sample.nodes))
        source = torch.cat((sample.edges[:, 0], ids))
        target = torch.cat((sample.edges[:, 1], ids))
        degree = torch.bincount(target, minlength=len(ids)).float()
        weights = (degree[source] * degree[target]).rsqrt()
        graph = torch.zeros_like(sample.nodes).index_add_(
            0, target, weights[:, None] * sample.nodes[source]
        )
        nodes = torch.cat((self.graph(graph).relu(), walk_embeddings), dim=1)
        paths = pack_sequence([nodes[path] for path in sample.paths], enforce_sorted=False)
        _, (structure, _) = self.path(paths)
        structure = torch.cat((structure[0], structure[1]), dim=1)
        _, time = self.temporal(sample.hourly.unsqueeze(0))
        time = time[0, 0]
        weights = (self.key(structure) @ time / time.numel() ** 0.5).softmax(0)
        return self.output(torch.cat((weights @ structure, time))).squeeze()
