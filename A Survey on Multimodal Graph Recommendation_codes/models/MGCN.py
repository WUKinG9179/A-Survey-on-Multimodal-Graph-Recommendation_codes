"""MGCN adapter: behavior-guided purification, fusion, and cross-view InfoNCE."""
import torch
from torch import nn
from torch.nn import functional as F

from utility.data import RecommendationData
from utility.experiment import build_parser, run_experiment, seed_everything


@torch.no_grad()
def fixed_cosine_graph(features, topk, block_size):
    x = F.normalize(features, dim=1)
    size, k = len(x), min(topk, len(x))
    columns, values = [], []
    for start in range(0, size, block_size):
        weights, indices = (x[start:start + block_size] @ x.T).topk(k, dim=1)
        columns.append(indices.reshape(-1))
        values.append(weights.reshape(-1).clamp_min(0))
    rows = torch.arange(size, device=x.device).repeat_interleave(k)
    cols, weights = torch.cat(columns), torch.cat(values)
    degree = weights.new_zeros(size).index_add_(0, rows, weights)
    inverse = degree.clamp_min(torch.finfo(weights.dtype).eps).rsqrt()
    inverse = torch.where(degree > 0, inverse, torch.zeros_like(inverse))
    return torch.sparse_coo_tensor(torch.stack((rows, cols)),
                                   weights * inverse[rows] * inverse[cols], (size, size)).coalesce()


def training_graphs(data):
    pairs = torch.from_numpy(data.train_pairs.copy()).T
    users, items = pairs
    user_degree = torch.bincount(users, minlength=data.num_users).float().clamp_min(1)
    item_degree = torch.bincount(items, minlength=data.num_items).float().clamp_min(1)
    values = user_degree[users].rsqrt() * item_degree[items].rsqrt()
    interaction = torch.sparse_coo_tensor(pairs, values, (data.num_users, data.num_items)).coalesce()
    item_nodes = items + data.num_users
    indices = torch.stack((torch.cat((users, item_nodes)), torch.cat((item_nodes, users))))
    size = data.num_users + data.num_items
    graph = torch.sparse_coo_tensor(indices, values.repeat(2), (size, size)).coalesce()
    return graph, interaction


class MGCN(nn.Module):
    def __init__(self, data, embedding_dim=64, reg_weight=1e-4, n_layers=1,
                 n_ui_layers=2, ssl_temp=0.2, ssl_alpha=0.01, ii_topk=10,
                 graph_block_size=256, freeze_features=False):
        super().__init__()
        if min(embedding_dim, ii_topk, graph_block_size) < 1 or min(n_layers, n_ui_layers) < 0:
            raise ValueError("Widths, topk and block size must be positive; layer counts nonnegative")
        if ssl_temp <= 0 or min(ssl_alpha, reg_weight) < 0:
            raise ValueError("ssl_temp must be positive; loss weights nonnegative")
        self.num_users, self.num_items = data.num_users, data.num_items
        self.output_dim, self.reg_weight = embedding_dim, reg_weight
        self.n_layers, self.n_ui_layers = n_layers, n_ui_layers
        self.ssl_temp, self.ssl_alpha = ssl_temp, ssl_alpha
        self.ii_topk, self.graph_block_size = ii_topk, graph_block_size
        graph, interaction = training_graphs(data)
        self.register_buffer("norm_adj", graph, persistent=False)
        self.register_buffer("interaction", interaction, persistent=False)
        self.user_embedding = nn.Embedding(self.num_users, embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.feature_embeddings = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        self.purifiers = nn.ModuleDict()
        self.preference_gates = nn.ModuleDict()
        for name, features in data.features.items():
            self.register_buffer(name + "_original_features", features, persistent=False)
            self.register_buffer(name + "_graph", None, persistent=False)
            self.feature_embeddings[name] = nn.Embedding.from_pretrained(features.clone(), freeze=freeze_features)
            self.projections[name] = nn.Linear(features.shape[1], embedding_dim)
            self.purifiers[name] = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.Sigmoid())
            self.preference_gates[name] = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.Sigmoid())
        self.query_common = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.Tanh(),
                                          nn.Linear(embedding_dim, 1, bias=False))

    def prepare_graphs(self):
        for name in self.feature_embeddings:
            if getattr(self, name + "_graph") is None:
                with torch.inference_mode(False), torch.no_grad():
                    graph = fixed_cosine_graph(getattr(self, name + "_original_features"),
                                               self.ii_topk, self.graph_block_size)
                setattr(self, name + "_graph", graph)

    def fuse_views(self, content, modalities):
        logits = torch.cat([self.query_common(view) for view in modalities], dim=1)
        weights = torch.softmax(logits, dim=1)
        common = sum(weights[:, index:index + 1] * view for index, view in enumerate(modalities))
        separate = [self.preference_gates[name](content) * (view - common)
                    for name, view in zip(self.feature_embeddings, modalities)]
        # Original two-modality denominator 3 becomes M+1 for M modalities.
        return (common + sum(separate)) / (len(modalities) + 1)

    def forward(self):
        self.prepare_graphs()
        x = torch.cat((self.user_embedding.weight, self.item_embedding.weight))
        layers = [x]
        for _ in range(self.n_ui_layers):
            x = torch.sparse.mm(self.norm_adj, x)
            layers.append(x)
        content = torch.stack(layers).mean(dim=0)
        modalities = []
        for name, embedding in self.feature_embeddings.items():
            features = self.projections[name](embedding.weight)
            items = self.item_embedding.weight * self.purifiers[name](features)
            for _ in range(self.n_layers):
                items = torch.sparse.mm(getattr(self, name + "_graph"), items)
            users = torch.sparse.mm(self.interaction, items)
            modalities.append(torch.cat((users, items)))
        side = self.fuse_views(content, modalities)
        users, items = (content + side).split((self.num_users, self.num_items))
        return users, items, side, content

    def encode(self):
        users, items, _, _ = self.forward()
        return users, items

    def info_nce(self, first, second):
        logits = F.normalize(first, dim=1) @ F.normalize(second, dim=1).T / self.ssl_temp
        labels = torch.arange(len(first), device=first.device)
        return F.cross_entropy(logits, labels)

    def compute_loss(self, users, positive, negative):
        user_emb, item_emb, side, content = self.forward()
        u, p, n = user_emb[users], item_emb[positive], item_emb[negative]
        bpr = F.softplus(-(u * (p - n)).sum(dim=1)).mean()
        reg = u.square().mean() + p.square().mean() + n.square().mean()
        loss = bpr + self.reg_weight * reg
        if self.ssl_alpha:
            item_nodes = positive + self.num_users
            ssl = self.info_nce(side[users], content[users]) + self.info_nce(side[item_nodes], content[item_nodes])
            loss = loss + self.ssl_alpha * ssl
        return loss


def main():
    parser = build_parser("MGCN")
    parser.add_argument("--n-layers", type=int, default=1, help="item graph layers")
    parser.add_argument("--n-ui-layers", type=int, default=2)
    parser.add_argument("--ssl-temp", type=float, default=0.2)
    parser.add_argument("--ssl-alpha", type=float, default=0.01)
    parser.add_argument("--ii-topk", type=int, default=10)
    parser.add_argument("--graph-block-size", type=int, default=256)
    parser.add_argument("--freeze-features", action="store_true")
    args = parser.parse_args()
    if min(args.embedding_dim, args.ii_topk, args.graph_block_size) < 1 or min(args.n_layers, args.n_ui_layers) < 0:
        parser.error("Widths, topk, block size must be positive; layer counts nonnegative")
    if args.ssl_temp <= 0 or args.ssl_alpha < 0:
        parser.error("ssl-temp must be positive; ssl-alpha nonnegative")
    seed_everything(args.seed)
    data = RecommendationData(args.data_root, args.dataset, args.core, args.modalities)
    model = MGCN(data, args.embedding_dim, args.reg_weight, args.n_layers, args.n_ui_layers,
                 args.ssl_temp, args.ssl_alpha, args.ii_topk, args.graph_block_size, args.freeze_features)
    args.output_embedding_dim = model.output_dim
    args.item_graph_type = "fixed_nonnegative_cosine_knn"
    run_experiment(model, data, args)


if __name__ == "__main__":
    main()
