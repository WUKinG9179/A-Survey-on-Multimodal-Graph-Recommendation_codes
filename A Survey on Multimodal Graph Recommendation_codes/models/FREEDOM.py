"""FREEDOM: frozen modality item graphs and degree-sensitive UI pruning.

Shared data/metrics/runner adapter. Graphs use only training interactions and
input modality features; evaluation always uses the complete training UI graph.
"""
import torch
from torch import nn
from torch.nn import functional as F

from utility.data import RecommendationData
from utility.experiment import build_parser, run_experiment, seed_everything


def interaction_weights(pairs, num_users, num_items):
    users, items = pairs.T
    user_degree = torch.bincount(users, minlength=num_users).float() + 1e-7
    item_degree = torch.bincount(items, minlength=num_items).float() + 1e-7
    return user_degree[users].rsqrt() * item_degree[items].rsqrt()


def interaction_graph(pairs, num_users, num_items):
    weights = interaction_weights(pairs, num_users, num_items)
    users, items = pairs.T
    items = items + num_users
    indices = torch.stack((torch.cat((users, items)), torch.cat((items, users))))
    size = num_users + num_items
    return torch.sparse_coo_tensor(indices, weights.repeat(2), (size, size)).coalesce()


@torch.no_grad()
def frozen_knn_graph(features, topk=10, block_size=256):
    """Cosine selects binary neighbors; edge weights are NOT cosine scores.

    Includes self among candidates, matching the supplied model. Row-degree
    symmetric normalization of the directed binary kNN graph is preserved.
    """
    normalized = F.normalize(features, dim=1)
    size, k = len(features), min(topk, len(features))
    columns = []
    for start in range(0, size, block_size):
        columns.append((normalized[start:start + block_size] @ normalized.T).topk(k, dim=1).indices)
    rows = torch.arange(size, device=features.device).repeat_interleave(k)
    cols = torch.cat(columns).reshape(-1)
    degree = torch.bincount(rows, minlength=size).float() + 1e-7
    values = degree[rows].rsqrt() * degree[cols].rsqrt()
    return torch.sparse_coo_tensor(torch.stack((rows, cols)), values, (size, size)).coalesce()


class FREEDOM(nn.Module):
    def __init__(self, data, embedding_dim=64, reg_weight=1e-4, dropout=0.9,
                 n_layers=2, mm_layers=1, ii_topk=10, mm_image_weight=None,
                 graph_block_size=256, freeze_features=False):
        super().__init__()
        if min(embedding_dim, mm_layers, ii_topk, graph_block_size) < 1 or n_layers < 0:
            raise ValueError("Widths, mm-layers, topk, block size must be positive; n-layers >= 0")
        if not 0 <= dropout < 1 or reg_weight < 0:
            raise ValueError("dropout must be in [0,1); reg_weight must be nonnegative")
        if mm_image_weight is not None and not 0 <= mm_image_weight <= 1:
            raise ValueError("mm_image_weight must be in [0,1]")
        self.num_users, self.num_items = data.num_users, data.num_items
        self.output_dim, self.reg_weight = embedding_dim, reg_weight
        self.dropout, self.n_layers, self.mm_layers = dropout, n_layers, mm_layers
        self.ii_topk, self.graph_block_size = ii_topk, graph_block_size
        pairs = torch.from_numpy(data.train_pairs.copy())
        self.register_buffer("train_pairs", pairs, persistent=False)
        self.register_buffer("edge_weights", interaction_weights(pairs, self.num_users, self.num_items), persistent=False)
        self.register_buffer("norm_adj", interaction_graph(pairs, self.num_users, self.num_items), persistent=False)
        self.register_buffer("masked_adj", None, persistent=False)
        self.register_buffer("mm_adj", None, persistent=False)
        self.user_embedding = nn.Embedding(self.num_users, embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.feature_embeddings = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        names = list(data.features)
        weights = [1 / len(names)] * len(names)
        if mm_image_weight is not None:
            weights = [mm_image_weight if name == "image" else (1 - mm_image_weight) / (len(names) - 1) for name in names]
        self.register_buffer("modality_weights", torch.tensor(weights))
        for name, features in data.features.items():
            self.register_buffer(name + "_original_features", features, persistent=False)
            self.feature_embeddings[name] = nn.Embedding.from_pretrained(features.clone(), freeze=freeze_features)
            # Auxiliary BPR dot products require the same width as user embeddings.
            self.projections[name] = nn.Linear(features.shape[1], embedding_dim)

    def prepare_item_graph(self):
        if self.mm_adj is None:
            # Cache normal tensors even if the first call occurs under inference_mode.
            with torch.inference_mode(False), torch.no_grad():
                graphs = [frozen_knn_graph(getattr(self, name + "_original_features"), self.ii_topk,
                                          self.graph_block_size) for name in self.feature_embeddings]
                indices = torch.cat([graph.indices() for graph in graphs], dim=1)
                values = torch.cat([graph.values() * weight for graph, weight in zip(graphs, self.modality_weights)])
                self.mm_adj = torch.sparse_coo_tensor(indices, values, (self.num_items, self.num_items)).coalesce()

    @torch.no_grad()
    def pre_epoch_processing(self):
        self.prepare_item_graph()
        if self.dropout == 0:
            self.masked_adj = self.norm_adj
            return
        count = int(len(self.train_pairs) * (1 - self.dropout))
        selected = (torch.multinomial(self.edge_weights, count, replacement=False)
                    if count else torch.empty(0, dtype=torch.long, device=self.train_pairs.device))
        self.masked_adj = interaction_graph(self.train_pairs[selected], self.num_users, self.num_items)

    def propagate(self, adjacency):
        self.prepare_item_graph()
        content = self.item_embedding.weight
        for _ in range(self.mm_layers):
            content = torch.sparse.mm(self.mm_adj, content)
        x = torch.cat((self.user_embedding.weight, self.item_embedding.weight))
        outputs = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(adjacency, x)
            outputs.append(x)
        users, items = torch.stack(outputs).mean(dim=0).split((self.num_users, self.num_items))
        return users, items + content

    def encode(self):
        return self.propagate(self.norm_adj)

    def forward(self):
        return self.encode()

    @staticmethod
    def bpr_loss(users, positives, negatives):
        return F.softplus(-(users * (positives - negatives)).sum(dim=1)).mean()

    def compute_loss(self, users, positive, negative):
        if self.masked_adj is None:
            self.pre_epoch_processing()
        user_emb, item_emb = self.propagate(self.masked_adj)
        u = user_emb[users]
        loss = self.bpr_loss(u, item_emb[positive], item_emb[negative])
        if self.reg_weight:
            # These are auxiliary modality BPR objectives, not L2 regularization.
            for name, embedding in self.feature_embeddings.items():
                projection = self.projections[name]
                positive_features = projection(embedding(positive))
                negative_features = projection(embedding(negative))
                loss = loss + self.reg_weight * self.bpr_loss(u, positive_features, negative_features)
        return loss


def main():
    parser = build_parser("FREEDOM")
    parser.add_argument("--dropout", type=float, default=0.9)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--mm-layers", type=int, default=1)
    parser.add_argument("--ii-topk", type=int, default=10)
    parser.add_argument("--mm-image-weight", type=float, default=None, help="default equal modalities; residual shared by non-image modalities")
    parser.add_argument("--graph-block-size", type=int, default=256)
    parser.add_argument("--freeze-features", action="store_true")
    args = parser.parse_args()
    if min(args.embedding_dim, args.mm_layers, args.ii_topk, args.graph_block_size) < 1 or args.n_layers < 0:
        parser.error("Widths, mm-layers, topk, block size must be positive; n-layers >= 0")
    if not 0 <= args.dropout < 1 or (args.mm_image_weight is not None and not 0 <= args.mm_image_weight <= 1):
        parser.error("dropout must be in [0,1); mm-image-weight in [0,1]")
    seed_everything(args.seed)
    data = RecommendationData(args.data_root, args.dataset, args.core, args.modalities)
    model = FREEDOM(data, args.embedding_dim, args.reg_weight, args.dropout, args.n_layers,
                    args.mm_layers, args.ii_topk, args.mm_image_weight, args.graph_block_size, args.freeze_features)
    args.output_embedding_dim = model.output_dim
    args.modality_graph_weights = dict(zip(model.feature_embeddings, model.modality_weights.tolist()))
    args.regularization_type = "auxiliary_modality_bpr"
    run_experiment(model, data, args)


if __name__ == "__main__":
    main()
