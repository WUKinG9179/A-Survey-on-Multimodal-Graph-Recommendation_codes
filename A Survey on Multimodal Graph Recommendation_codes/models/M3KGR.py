"""M3KGR-inspired NO-KG adaptation, explicitly authorized for local datasets.

Not the paper's full M3KGR: no entities/relations, TransR loss or CLIP extraction.
Uses train-only UI attention, supplied modality features, EMA keys, queues, and
similarity-based negative downweighting. See utility/README.md for equations.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from utility.data import RecommendationData
from utility.experiment import build_parser, run_experiment, seed_everything


class MomentumQueue(nn.Module):
    def __init__(self, capacity, dimension):
        super().__init__()
        self.register_buffer('keys', torch.zeros(capacity, dimension))
        self.register_buffer('ids', torch.full((capacity,), -1, dtype=torch.long))
        self.register_buffer('pointer', torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, keys, ids):
        count, capacity = len(ids), len(self.ids)
        if not count:
            return
        # Preserve circular FIFO semantics even for batches larger than capacity.
        start = max(0, count - capacity)
        positions = (torch.arange(start, count, device=ids.device) + self.pointer) % capacity
        self.keys[positions] = keys[start:]
        self.ids[positions] = ids[start:]
        self.pointer.copy_((self.pointer + count) % capacity)

    def valid(self):
        mask = self.ids >= 0
        return self.keys[mask], self.ids[mask]


class UIEncoder(nn.Module):
    def __init__(self, num_users, num_items, dimensions, width, layers, dropout):
        super().__init__()
        self.num_users, self.layers, self.dropout = num_users, layers, dropout
        self.users = nn.Embedding(num_users, width)
        self.items = nn.Embedding(num_items, width)
        nn.init.xavier_uniform_(self.users.weight)
        nn.init.xavier_uniform_(self.items.weight)
        self.modality_projection = nn.ModuleDict({name: nn.Linear(dim, width) for name, dim in dimensions.items()})
        self.modal_logits = nn.Parameter(torch.zeros(len(dimensions)))
        self.fusion = nn.Linear(2 * width, width)
        self.transforms = nn.ModuleList([nn.Linear(width, width) for _ in range(layers)])
        self.projection_head = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width))

    def forward(self, features, edges, augment=False):
        weights = self.modal_logits.softmax(dim=0)
        modalities = [F.normalize(self.modality_projection[name](value), dim=1) for name, value in features.items()]
        content = sum(weight * value for weight, value in zip(weights, modalities))
        items = self.fusion(torch.cat((self.items.weight, content), dim=1))
        x = F.normalize(torch.cat((self.users.weight, items)), dim=1)
        x = F.dropout(x, p=self.dropout, training=augment)
        layers = [x]
        source, target = edges
        for transform in self.transforms:
            logits = (x[source] * x[target]).sum(dim=1) / (x.shape[1] ** 0.5)
            maximum = logits.new_full((len(x),), -torch.inf)
            maximum.scatter_reduce_(0, target, logits.detach(), reduce='amax', include_self=True)
            numerator = (logits - maximum[target]).exp()
            denominator = logits.new_zeros(len(x)).index_add_(0, target, numerator)
            attention = numerator / denominator[target].clamp_min(torch.finfo(x.dtype).tiny)
            messages = torch.zeros_like(x).index_add_(0, target, attention[:, None] * x[source])
            x = F.normalize(F.leaky_relu(transform(x + messages)), dim=1)
            layers.append(x)
        return torch.stack(layers).mean(dim=0)

    def project(self, x):
        return F.normalize(self.projection_head(x), dim=1)


class M3KGR(nn.Module):
    model_variant = 'M3KGR-inspired-NoKG'

    def __init__(self, data, embedding_dim=64, reg_weight=1e-4, n_layers=2,
                 queue_size=4096, momentum=0.999, temperature=0.2, ssl_weight=0.01,
                 momentum_weight=0.01, edge_dropout=0.1, feature_dropout=0.1,
                 negative_threshold=0.8, negative_strength=5.0):
        super().__init__()
        if min(embedding_dim, queue_size) < 1 or n_layers < 0 or temperature <= 0:
            raise ValueError('Invalid widths, queue size, layers or temperature')
        if not 0 <= momentum < 1 or not 0 <= edge_dropout < 1 or not 0 <= feature_dropout < 1:
            raise ValueError('Momentum and dropout rates must be in [0,1)')
        if min(reg_weight, ssl_weight, momentum_weight, negative_strength) < 0 or not -1 <= negative_threshold <= 1:
            raise ValueError('Invalid objective weights or negative threshold')
        self.num_users, self.num_items = data.num_users, data.num_items
        self.output_dim, self.reg_weight = embedding_dim, reg_weight
        self.momentum, self.temperature = momentum, temperature
        self.ssl_weight, self.momentum_weight = ssl_weight, momentum_weight
        self.edge_dropout = edge_dropout
        self.negative_threshold, self.negative_strength = negative_threshold, negative_strength
        pairs = torch.from_numpy(data.train_pairs.copy()).T
        pairs[1] += self.num_users
        self.register_buffer('train_edges', pairs, persistent=False)
        for name, feature in data.features.items():
            self.register_buffer(name + '_features', feature, persistent=False)
        self.modalities = list(data.features)
        self.query_encoder = UIEncoder(self.num_users, self.num_items,
            {name: feature.shape[1] for name, feature in data.features.items()}, embedding_dim, n_layers, feature_dropout)
        self.key_encoder = copy.deepcopy(self.query_encoder)
        self.key_encoder.requires_grad_(False)
        self.user_queue = MomentumQueue(queue_size, embedding_dim)
        self.item_queue = MomentumQueue(queue_size, embedding_dim)
        self._pending = None

    def features(self):
        return {name: getattr(self, name + '_features') for name in self.modalities}

    def graph(self, augmented=False):
        edges = self.train_edges
        if augmented:
            edges = edges[:, torch.rand(edges.shape[1], device=edges.device) >= self.edge_dropout]
        return torch.cat((edges, edges.flip(0)), dim=1)

    def encode(self):
        return self.query_encoder(self.features(), self.graph()).split((self.num_users, self.num_items))

    def forward(self):
        return self.encode()

    def queue_loss(self, query, positive_keys, ids, queue):
        stored, stored_ids = queue.valid()
        candidates = torch.cat((positive_keys.detach(), stored.detach()))
        candidate_ids = torch.cat((ids, stored_ids))
        positive = (query * positive_keys.detach()).sum(dim=1, keepdim=True) / self.temperature
        negative = query @ candidates.T / self.temperature
        # A configurable NO-KG heuristic, not a claim to implement paper equations.
        with torch.no_grad():
            similarity = positive_keys @ candidates.T
            log_weight = -self.negative_strength * F.relu(similarity - self.negative_threshold)
            same_id = ids[:, None] == candidate_ids[None, :]
        negative = (negative + log_weight).masked_fill(same_id, -torch.inf)
        logits = torch.cat((positive, negative), dim=1)
        return F.cross_entropy(logits, torch.zeros(len(ids), dtype=torch.long, device=ids.device))

    def compute_loss(self, users, positive, negative):
        self._pending = None
        user_emb, item_emb = self.encode()
        margin = (user_emb[users] * (item_emb[positive] - item_emb[negative])).sum(dim=1)
        loss = F.softplus(-margin).mean()
        reg = self.query_encoder.users(users).square().mean()
        reg = reg + self.query_encoder.items(positive).square().mean() + self.query_encoder.items(negative).square().mean()
        loss = loss + self.reg_weight * reg
        if not (self.ssl_weight or self.momentum_weight):
            return loss
        features = self.features()
        first_graph, second_graph = self.graph(True), self.graph(True)
        first = self.query_encoder(features, first_graph, augment=True)
        unique_users, unique_items = users.unique(), positive.unique()
        selections = (unique_users, unique_items + self.num_users)
        queries = [self.query_encoder.project(first[index]) for index in selections]
        if self.ssl_weight:
            second = self.query_encoder(features, second_graph, augment=True)
            positives = [self.query_encoder.project(second[index]) for index in selections]
            ssl = sum(F.cross_entropy(q @ p.T / self.temperature,
                      torch.arange(len(q), device=q.device)) for q, p in zip(queries, positives))
            loss = loss + self.ssl_weight * ssl
        if self.momentum_weight:
            with torch.no_grad():
                keys = self.key_encoder(features, second_graph, augment=True)
                keys = [self.key_encoder.project(keys[index]) for index in selections]
            contrast = self.queue_loss(queries[0], keys[0], unique_users, self.user_queue)
            contrast = contrast + self.queue_loss(queries[1], keys[1], unique_items, self.item_queue)
            loss = loss + self.momentum_weight * contrast
            self._pending = (keys[0].detach(), unique_users.detach(), keys[1].detach(), unique_items.detach())
        return loss

    @torch.no_grad()
    def post_optimizer_step(self):
        # Called only after backward and a successful optimizer step.
        for query, key in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            key.lerp_(query, 1 - self.momentum)
        if self._pending is not None:
            users, user_ids, items, item_ids = self._pending
            self.user_queue.enqueue(users, user_ids)
            self.item_queue.enqueue(items, item_ids)
            self._pending = None


def main():
    parser = build_parser('M3KGR (NO-KG adaptation)')
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--queue-size', type=int, default=4096)
    parser.add_argument('--momentum', type=float, default=0.999)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--ssl-weight', type=float, default=0.01)
    parser.add_argument('--momentum-weight', type=float, default=0.01)
    parser.add_argument('--edge-dropout', type=float, default=0.1)
    parser.add_argument('--feature-dropout', type=float, default=0.1)
    parser.add_argument('--negative-threshold', type=float, default=0.8)
    parser.add_argument('--negative-strength', type=float, default=5.0)
    args = parser.parse_args()
    seed_everything(args.seed)
    data = RecommendationData(args.data_root, args.dataset, args.core, args.modalities)
    model = M3KGR(data, args.embedding_dim, args.reg_weight, args.n_layers, args.queue_size,
                   args.momentum, args.temperature, args.ssl_weight, args.momentum_weight,
                   args.edge_dropout, args.feature_dropout, args.negative_threshold, args.negative_strength)
    args.model_variant = model.model_variant
    args.knowledge_graph_used = False
    args.clip_features_verified = False
    args.output_embedding_dim = model.output_dim
    args.negative_weighting = 'exp(-strength*relu(key_cosine-threshold)); same-ID masked'
    print('M3KGR-inspired-NoKG: authorized adaptation, NOT full paper M3KGR.', flush=True)
    run_experiment(model, data, args)


if __name__ == '__main__':
    main()
