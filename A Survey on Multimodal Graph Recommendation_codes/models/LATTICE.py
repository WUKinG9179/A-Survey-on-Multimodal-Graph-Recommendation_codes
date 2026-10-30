"""LATTICE adapter with blockwise kNN construction and sparse propagation.

Original modality graphs are fixed; learned graphs are rebuilt on every forward
so each optimizer step updates features, projections and modality weights.
"""
import torch
from torch import nn
from torch.nn import functional as F

from utility.data import RecommendationData
from utility.experiment import build_parser, run_experiment, seed_everything


def build_knn_graph(features, topk, block_size):
    """Exact cosine top-k (including self), nonnegative sparse edge weights.

    Discrete neighbor selection needs no gradient. Recomputing selected cosine
    values preserves the piecewise top-k gradient without an N-by-N autograd graph.
    """
    count = len(features)
    k = min(topk, count)
    normalized = F.normalize(features, p=2, dim=1)
    indices = []
    with torch.no_grad():
        detached = normalized.detach()
        for start in range(0, count, block_size):
            similarities = detached[start:start + block_size] @ detached.T
            indices.append(similarities.topk(k, dim=1).indices)
    columns = torch.cat(indices).reshape(-1)
    rows = torch.arange(count, device=features.device).repeat_interleave(k)
    # Gather in blocks as well to bound temporary edge-feature tensors.
    values = []
    for start in range(0, len(rows), block_size * k):
        row, col = rows[start:start + block_size * k], columns[start:start + block_size * k]
        values.append((normalized[row] * normalized[col]).sum(dim=1).clamp_min(0))
    return torch.sparse_coo_tensor(torch.stack((rows, columns)), torch.cat(values),
                                   (count, count)).coalesce()


def normalize_graph(graph):
    """D^(-1/2) A D^(-1/2), using row sums as in the supplied LATTICE."""
    graph = graph.coalesce()
    rows, cols = graph.indices()
    values = graph.values()
    degree = values.new_zeros(graph.shape[0]).index_add_(0, rows, values)
    inv = degree.clamp_min(torch.finfo(values.dtype).eps).rsqrt()
    inv = torch.where(degree > 0, inv, torch.zeros_like(inv))
    return torch.sparse_coo_tensor(graph.indices(), values * inv[rows] * inv[cols],
                                   graph.shape).coalesce()


def combine_graphs(graphs, weights):
    indices = torch.cat([g.coalesce().indices() for g in graphs], dim=1)
    values = torch.cat([g.coalesce().values() * weight for g, weight in zip(graphs, weights)])
    return torch.sparse_coo_tensor(indices, values, graphs[0].shape).coalesce()


class LATTICE(nn.Module):
    def __init__(self, data, embedding_dim=64, reg_weight=1e-4, feat_embed_dim=64,
                 n_layers=2, mm_layers=1, ii_topk=10, lambda_coeff=0.5,
                 graph_block_size=256, freeze_features=False):
        super().__init__()
        if min(embedding_dim, feat_embed_dim, ii_topk, graph_block_size, mm_layers) < 1 or n_layers < 0:
            raise ValueError("Widths, topk, block size and mm-layers must be positive; n-layers >= 0")
        if not 0 <= lambda_coeff <= 1:
            raise ValueError("lambda-coeff must be between zero and one")
        self.num_users, self.num_items = data.num_users, data.num_items
        self.output_dim = embedding_dim
        self.reg_weight, self.n_layers, self.mm_layers = reg_weight, n_layers, mm_layers
        self.ii_topk, self.lambda_coeff = ii_topk, lambda_coeff
        self.graph_block_size = graph_block_size
        self.user_embedding = nn.Embedding(self.num_users, embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.register_buffer("ui_graph", normalize_graph(data.adjacency("add")), persistent=False)
        self.feature_embeddings = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        for name, features in data.features.items():
            self.feature_embeddings[name] = nn.Embedding.from_pretrained(features.clone(), freeze=freeze_features)
            self.projections[name] = nn.Linear(features.shape[1], feat_embed_dim)
            # Fixed original graph is based on input data, not trained feature weights.
            self.register_buffer(name + "_original_features", features, persistent=False)
            self.register_buffer(name + "_original_graph", None, persistent=False)
        self.modal_weight = nn.Parameter(torch.full((len(self.feature_embeddings),), 0.5))

    def _original_graphs(self):
        graphs = []
        for name in self.feature_embeddings:
            graph = getattr(self, name + "_original_graph")
            if graph is None:
                # Normal tensors even if the first call is under inference_mode;
                # fixed graph values are needed by subsequent training autograd.
                with torch.inference_mode(False), torch.no_grad():
                    features = getattr(self, name + "_original_features")
                    graph = normalize_graph(build_knn_graph(features, self.ii_topk, self.graph_block_size))
                setattr(self, name + "_original_graph", graph)
            graphs.append(graph)
        return graphs

    def build_item_graph(self):
        originals = self._original_graphs()
        weights = torch.softmax(self.modal_weight, dim=0)
        learned = [build_knn_graph(self.projections[name](embedding.weight), self.ii_topk,
                                   self.graph_block_size)
                   for name, embedding in self.feature_embeddings.items()]
        learned = normalize_graph(combine_graphs(learned, weights))
        original = combine_graphs(originals, weights)
        return combine_graphs([learned, original], [1 - self.lambda_coeff, self.lambda_coeff])

    def encode(self):
        graph = self.build_item_graph()
        h = self.item_embedding.weight
        for _ in range(self.mm_layers):
            h = torch.sparse.mm(graph, h)
        x = torch.cat((self.user_embedding.weight, self.item_embedding.weight))
        layers = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.ui_graph, x)
            layers.append(x)
        users, items = torch.stack(layers).mean(dim=0).split((self.num_users, self.num_items))
        return users, items + F.normalize(h, p=2, dim=1)

    def forward(self):
        return torch.cat(self.encode(), dim=0)

    def compute_loss(self, users, positive, negative):
        user_emb, item_emb = self.encode()
        u, p, n = user_emb[users], item_emb[positive], item_emb[negative]
        bpr = F.softplus(-(u * (p - n)).sum(dim=1)).mean()
        reg = u.square().mean() + p.square().mean() + n.square().mean()
        return bpr + self.reg_weight * reg


def main():
    parser = build_parser("LATTICE")
    parser.add_argument("--feat-embed-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--mm-layers", type=int, default=1)
    parser.add_argument("--ii-topk", type=int, default=10)
    parser.add_argument("--lambda-coeff", type=float, default=0.5)
    parser.add_argument("--graph-block-size", type=int, default=256)
    parser.add_argument("--freeze-features", action="store_true")
    args = parser.parse_args()
    if min(args.embedding_dim, args.feat_embed_dim, args.mm_layers, args.ii_topk, args.graph_block_size) < 1 or args.n_layers < 0:
        parser.error("Widths, mm-layers, topk and graph-block-size must be positive; n-layers >= 0")
    if not 0 <= args.lambda_coeff <= 1:
        parser.error("lambda-coeff must be in [0, 1]")
    seed_everything(args.seed)
    data = RecommendationData(args.data_root, args.dataset, args.core, args.modalities)
    model = LATTICE(data, args.embedding_dim, args.reg_weight, args.feat_embed_dim,
                    args.n_layers, args.mm_layers, args.ii_topk, args.lambda_coeff,
                    args.graph_block_size, args.freeze_features)
    args.output_embedding_dim = model.output_dim
    args.graph_rebuild = "every_forward"
    run_experiment(model, data, args)


if __name__ == "__main__":
    main()
