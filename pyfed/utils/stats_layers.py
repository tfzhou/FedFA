import torch
import torch.nn as nn

import numpy as np
import random

from pyfed.utils.surprise import surprise_bins

SEED = 404
EPS = 1e-8


class StatsLayer(nn.Module):
    """
    Base module for all Stats layers.
    """

    def __init__(self, n_features, eps=1e-5, momentum=0.9, rng=None, track_stats=False, calc_surprise=False,
                 surprise_score="KL_Q_P", cpu=True):
        super(StatsLayer, self).__init__()
        self.n_features = n_features
        self.eps = eps
        self.momentum = momentum
        self.rng = np.random.RandomState(SEED) if rng is None else rng
        self.track_stats = track_stats
        self.calc_surprise = calc_surprise
        self.cpu = cpu
        self.register_buffer('surprise', torch.zeros(self.n_features))
        self.parent_scores = None
        self.surprise_score = surprise_score

    def get_stats(self):
        raise NotImplementedError

    def reset_stats(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def reset_rng(self):
        self.rng.seed(SEED)
        torch.manual_seed(SEED)
        random.seed(SEED)

    def extra_repr(self):
        return '{n_features}'.format(**self.__dict__)


class BinStats(StatsLayer):
    """
        Module to track the unit activation distributions of a layer with bins.
    """

    def __init__(self, n_features, eps=1e-5, momentum=0.9, rng=None, track_stats=False,
                 calc_surprise=False, track_range=False, n_bins=8, surprise_score="PSI", cpu=True, norm_range=True):
        super(BinStats, self).__init__(n_features, eps, momentum, rng, track_stats, calc_surprise, surprise_score, cpu)
        #  What to track : buffers (savable params) -- used for bins later
        self.register_buffer('mins', 1e6 * torch.ones(self.n_features))
        self.register_buffer('maxs', -1e6 * torch.ones(self.n_features))
        self.track_range = track_range
        self.n_bins = n_bins
        self.norm_range = norm_range
        self.n_bin_edges = n_bins + 1
        self.register_buffer('bin_edges', torch.zeros([self.n_features, self.n_bin_edges + 2]))  # 2 end bins
        self.register_buffer('bin_counts', torch.zeros([self.n_features, self.n_bins + 2]))  # 2 end bins
        self.register_buffer('feature_ranges', torch.ones(self.n_features, 1))

    def reset_stats(self):
        self.bin_counts.zero_()

    def reset_range(self):
        self.min.fill_(1e6)
        self.max.fill_(-1e6)
        self.bin_edges.zero_()

    def get_stats(self):
        return {"edges": self.bin_edges.detach().cpu().numpy(), "counts": self.bin_counts.detach().cpu().numpy(),
                "mins": self.mins.detach().cpu().numpy(), "maxs": self.maxs.detach().cpu().numpy()}

    def update_range(self, x):
        """
        Update activation range, if inputs fall outside current range. Flatten conv "spatial samples" by default.
        """
        # Conv: [B, C, H, W] --> [C, BxHxW], FC: [B, C] --> [C, B]
        x = x.transpose(0, 1).flatten(1).detach()

        # Get min over dim=1, i.e. over samples, one per feature/channel
        mns, mxs = x.min(1)[0], x.max(1)[0]
        self.mins = torch.where(self.mins < mns, self.mins, mns).detach()
        self.maxs = torch.where(self.maxs > mxs, self.maxs, mxs).detach()

    def init_bins(self):
        """
        Initialise bins with running range [min, max].
        """
        self.maxs += EPS  # add small value to edge to incl act in rightmost bin

        #  Set bin edges
        mns, mxs = self.mins, self.maxs
        if self.norm_range:
            feature_ranges = mxs - mns
            mxs = mxs / feature_ranges
            mns = mns / feature_ranges
            self.feature_ranges = feature_ranges.unsqueeze(1)

        b_edges = [torch.linspace(st, sp, self.n_bin_edges, device=self.maxs.device) for st, sp in list(zip(mns, mxs))]
        b_edges = torch.stack(b_edges, 0)
        # b_edges = torch.stack(b_edges, 0).to(torch.device('cuda'))   # [num_features, n_bin_edges]
        is_relu = torch.allclose(mns, torch.zeros_like(mns))

        #  Set width of additional end bins (i.e. range extensions)
        exts = 0.25 * (mxs - mns)  # end bin widths = 0.25 * range
        r_exts = (mxs + exts).reshape(-1, 1)  # right
        if is_relu:  # relu unit, larger left end bin width required
            l_exts = (mns - 2 * exts).reshape(-1, 1)
        else:
            l_exts = (mns - exts).reshape(-1, 1)

        self.bin_edges = torch.cat([l_exts, b_edges, r_exts], 1).detach()  # [num_features, n_bin_edges + 2]

    def _get_batch_counts(self, x):
        # Conv: [B, C, H, W] --> [C, BxHxW], FC: [B, C] --> [C, B]
        with torch.no_grad():
            x = x.transpose(0, 1).flatten(1)
            x = self._norm_inputs(x)
            bin_indices = torch.searchsorted(self.bin_edges[:, 1:-1], x)
            batch_counts = bincount2D_vectorized(bin_indices, minlength=(self.n_bins + 2))
        return batch_counts.detach()

    def _norm_inputs(self, x):
        """ x.shape = [C, -1]"""
        if self.norm_range and not self.track_range:
            return x / self.feature_ranges
        return x

    def update_bins(self, x):
        """
        Update bin counts for new inputs.
        :param x: array of inputs of dim (b,c,h,w) or (b,c).
        """
        if (self.bin_counts == 0.).all():  # counts all zero
            self.init_bins()

        batch_counts = self._get_batch_counts(x)
        self.bin_counts += batch_counts
        # self.bin_counts = self.bin_counts * self.momentum + batch_counts * (1. - self.momentum)

    def update_surprise(self, x):
        p_counts = self.bin_counts.detach().clone().cpu().numpy()
        batch_counts = self._get_batch_counts(x).detach().float().cpu().numpy()  # q_counts
        scores = surprise_bins(p_counts, batch_counts, score_type=self.surprise_score, fast=True)
        self.surprise = torch.from_numpy(scores).float().to(x.device)

    def forward(self, x):
        self._check_input_dim_if_implem(x)
        tracked_x = x.clone()  # branch off the in comp. graph

        if self.track_range:
            # Update mins and maxs
            self.update_range(tracked_x)

        elif self.track_stats:
            # Update bin counts
            self.update_bins(tracked_x)

        if self.calc_surprise:
            # Update surprise scores
            self.update_surprise(tracked_x)
        return x

    def _check_input_dim_if_implem(self, x):
        _check_input_dim = getattr(self, "_check_input_dim", None)
        if callable(_check_input_dim):  # i.e. object instance has a method called _check_input_dim
            _check_input_dim(x)

    def _load_from_state_dict(self, *args):
        super(BinStats, self)._load_from_state_dict(*args)
        if self.maxs[0] >= self.mins[0]:  # valid range has been loaded
            self.track_range = False


def bincount2D_vectorized(a, minlength=None):
    if minlength is None:
        minlength = a.max() + 1
    a_offs = a + torch.arange(a.shape[0], device=a.device)[:, None] * minlength
    return torch.bincount(a_offs.flatten(), minlength=a.shape[0] * minlength).reshape(-1, minlength)
