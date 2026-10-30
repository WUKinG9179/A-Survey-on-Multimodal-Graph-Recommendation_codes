"""DualGNN with train-only co-occurrence user graphs and shared benchmarking."""
from collections import Counter, defaultdict
import heapq
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utility.data import RecommendationData
from utility.experiment import build_parser, run_experiment, seed_everything


def build_user_neighbors(train, num_users, topk):
    """Top-k other users by shared train-item count; ties use ascending user ID.

    An inverted item index avoids storing a dense user-by-user matrix.
    Returns immutable lists; per-epoch padding never modifies these candidates.
    """
    item_users = defaultdict(list)
    for user, items in train.items():
        for item in sorted(items):
            item_users[item].append(user)
    neighbors = []
    for user in range(num_users):
        counts = Counter()
        for item in sorted(train.get(user, ())):
            counts.update(item_users[item])
        counts.pop(user, None)
        selected = heapq.nsmallest(topk, counts, key=lambda other: (-counts[other], other))
        neighbors.append((tuple(selected), tuple(counts[other] for other in selected)))
    return neighbors


def build_ui_graph(data, aggregation):
    graph = data.adjacency("add" if aggregation == "add" else "mean").coalesce()
    if aggregation == "mean":
        return graph
    rows, columns = graph.indices()
    degree = torch.bincount(rows, minlength=graph.shape[0]).float().clamp_min(1)
    values = graph.values() * degree[rows].rsqrt() * degree[columns].rsqrt()
    return torch.sparse_coo_tensor(graph.indices(), values, graph.shape).coalesce()


class GCN(nn.Module):
    def __init__(self, num_users, feature_dim, embedding_dim):
        super().__init__()
        self.preference = nn.Parameter(torch.empty(num_users, embedding_dim))
        nn.init.xavier_normal_(self.preference)
        self.projection = nn.Sequential(nn.Linear(feature_dim, 4 * embedding_dim),
                                        nn.LeakyReLU(), nn.Linear(4 * embedding_dim, embedding_dim))

    def forward(self, features, adjacency):
        x = F.normalize(torch.cat((self.preference, self.projection(features)), dim=0), dim=1)
        first = torch.sparse.mm(adjacency, x)
        second = torch.sparse.mm(adjacency, first)
        return x + first + second


class UserGraphSample(nn.Module):
    def forward(self, features, neighbors, weights):
        # Sum along neighbor axis only: preserve shape for a single user or dim=1.
        return (features[neighbors] * weights.unsqueeze(-1)).sum(dim=1)


class DualGNN(nn.Module):
    def __init__(self, data, embedding_dim=64, reg_weight=1e-4, uu_topk=40,
                 aggr_mode="add", construction="weighted_sum", user_aggr_mode="softmax", seed=2026):
        super().__init__()
        if embedding_dim < 1 or uu_topk < 1:
            raise ValueError("embedding_dim and uu_topk must be positive")
        if aggr_mode not in ("add", "mean") or user_aggr_mode not in ("softmax", "mean"):
            raise ValueError("Invalid graph aggregation mode")
        if construction not in ("weighted_sum", "mean", "max", "cat_mlp"):
            raise ValueError("Invalid modality construction mode")
        self.num_users, self.num_items = data.num_users, data.num_items
        self.output_dim, self.k = embedding_dim, uu_topk
        self.reg_weight, self.construction = reg_weight, construction
        self.user_aggr_mode = user_aggr_mode
        self.register_buffer("ui_graph", build_ui_graph(data, aggr_mode), persistent=False)
        self.branches = nn.ModuleDict()
        for name, features in data.features.items():
            self.register_buffer(name + "_features", features, persistent=False)
            self.branches[name] = GCN(self.num_users, features.shape[1], embedding_dim)
        if construction != "mean":
            self.weight_u = nn.Parameter(torch.empty(self.num_users, len(self.branches), 1))
            nn.init.xavier_normal_(self.weight_u)
            with torch.no_grad():
                self.weight_u.copy_(torch.softmax(self.weight_u, dim=1))
        else:
            self.register_parameter("weight_u", None)
        self.user_projection = (nn.Linear(embedding_dim * len(self.branches), embedding_dim)
                                if construction == "cat_mlp" else None)
        self.user_graph = UserGraphSample()
        start = time.perf_counter()
        self.user_neighbors = build_user_neighbors(data.train, self.num_users, uu_topk)
        self.user_graph_build_seconds = time.perf_counter() - start
        self._sampling_rng = np.random.default_rng(seed)
        # Sampled graph is part of the checkpoint, for exact restored evaluation.
        self.register_buffer("epoch_user_graph", torch.zeros(self.num_users, uu_topk, dtype=torch.long))
        self.register_buffer("user_weight_matrix", torch.zeros(self.num_users, uu_topk))
        self.pre_epoch_processing()

    def pre_epoch_processing(self):
        indices = np.zeros((self.num_users, self.k), dtype=np.int64)
        counts = np.zeros((self.num_users, self.k), dtype=np.float32)
        active = np.zeros(self.num_users, dtype=bool)
        for user, (candidates, shared) in enumerate(self.user_neighbors):
            if not candidates:
                continue
            active[user] = True
            selection = list(range(len(candidates)))
            # Retain original behavior: pad short neighbor lists by random repeats.
            if len(selection) < self.k:
                selection.extend(self._sampling_rng.integers(len(candidates), size=self.k - len(selection)).tolist())
            indices[user] = np.asarray(candidates)[selection]
            counts[user] = np.asarray(shared)[selection]
        if self.user_aggr_mode == "softmax":
            counts -= counts.max(axis=1, keepdims=True)
            weights = np.exp(counts)
            weights /= weights.sum(axis=1, keepdims=True)
        else:
            weights = np.full_like(counts, 1 / self.k)
        weights[~active] = 0
        self.epoch_user_graph.copy_(torch.from_numpy(indices).to(self.epoch_user_graph.device))
        self.user_weight_matrix.copy_(torch.from_numpy(weights).to(self.user_weight_matrix.device))

    def encode(self):
        representations = torch.stack([
            branch(getattr(self, name + "_features"), self.ui_graph)
            for name, branch in self.branches.items()], dim=1)
        users = representations[:self.num_users]
        if self.construction == "mean":
            users = users.mean(dim=1)
        elif self.construction == "weighted_sum":
            users = (users * self.weight_u).sum(dim=1)
        elif self.construction == "max":
            users = (users * self.weight_u).max(dim=1).values
        else:
            users = self.user_projection((users * self.weight_u).flatten(start_dim=1))
        # Preserve original item fusion: sum, not user-specific weighted fusion.
        items = representations[self.num_users:].sum(dim=1)
        users = users + self.user_graph(users, self.epoch_user_graph, self.user_weight_matrix)
        return users, items

    def forward(self):
        return torch.cat(self.encode(), dim=0)

    def compute_loss(self, users, positive, negative):
        user_emb, item_emb = self.encode()
        margin = (user_emb[users] * (item_emb[positive] - item_emb[negative])).sum(dim=1)
        loss = F.softplus(-margin).mean()
        reg = sum(branch.preference[users].square().mean() for branch in self.branches.values())
        if self.construction == "weighted_sum":
            reg = reg + self.weight_u.square().mean()
        elif self.construction == "cat_mlp":
            reg = reg + self.user_projection.weight.square().mean()
        return loss + self.reg_weight * reg


def main():
    parser = build_parser("DualGNN")
    parser.add_argument("--uu-topk", type=int, default=40)
    parser.add_argument("--aggr-mode", choices=("add", "mean"), default="add")
    parser.add_argument("--construction", choices=("weighted_sum", "mean", "max", "cat_mlp"), default="weighted_sum")
    parser.add_argument("--user-aggr-mode", choices=("softmax", "mean"), default="softmax")
    args = parser.parse_args()
    if args.embedding_dim < 1 or args.uu_topk < 1:
        parser.error("embedding-dim and uu-topk must be positive")
    seed_everything(args.seed)
    data = RecommendationData(args.data_root, args.dataset, args.core, args.modalities)
    print("Building user graph from shared training items...", flush=True)
    model = DualGNN(data, args.embedding_dim, args.reg_weight, args.uu_topk,
                    args.aggr_mode, args.construction, args.user_aggr_mode, args.seed)
    args.output_embedding_dim = model.output_dim
    args.user_graph_build_seconds = model.user_graph_build_seconds
    args.user_graph_source = "train_shared_item_count"
    print(f"User graph built in {model.user_graph_build_seconds:.3f}s", flush=True)
    run_experiment(model, data, args)


if __name__ == "__main__":
    main()
