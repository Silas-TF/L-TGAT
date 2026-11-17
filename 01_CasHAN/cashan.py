"""Topic adaptation of Zhong et al., DOI 10.1016/j.ins.2022.11.163."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

from cascade_inputs import cascade


def prepare(topic, windows, options):
    return [cascade(topic, w, options.walks, options.walk_length) for w in windows]


class CasHAN(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.encoder = nn.GRU(4, hidden, bidirectional=True, batch_first=True)
        self.node_attention = nn.Linear(2 * hidden + 2, 1)
        self.path_attention = nn.Linear(2 * hidden + 1, 1)
        self.output = nn.Linear(2 * hidden, 1)

    def forward(self, sample):
        paths = [sample.nodes[path] for path in sample.paths]
        encoded, _ = self.encoder(pack_sequence(paths, enforce_sorted=False))
        encoded, lengths = pad_packed_sequence(encoded, batch_first=True)
        influence = nn.utils.rnn.pad_sequence([path[:, 2:] for path in paths], batch_first=True)
        valid = torch.arange(encoded.shape[1]) < lengths[:, None]
        attention = self.node_attention(torch.cat((encoded, influence), dim=-1)).squeeze(-1)
        sequence = (encoded * attention.masked_fill(~valid, -torch.inf).softmax(1)[..., None]).sum(1)
        prior = sample.redundancy[:, None]
        weights = self.path_attention(torch.cat((sequence, prior), dim=-1)).squeeze(-1) - prior[:, 0]
        return self.output((sequence * weights.softmax(0)[:, None]).sum(0)).squeeze()
