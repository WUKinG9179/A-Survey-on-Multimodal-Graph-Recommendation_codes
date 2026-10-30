"""GRCN adapter using the shared data, evaluation and experiment runner.

Native PyTorch content attention, routing, confidence pruning and ID propagation.
Default mean fusion fixes final width at embedding_dim; concat remains available.
"""
import torch
from torch import nn
from torch.nn import functional as F

from utility.data import RecommendationData
from utility.experiment import build_parser, run_experiment, seed_everything


def edge_softmax(logits, target, num_nodes):
    maxima = logits.new_full((num_nodes,), -torch.inf)
    maxima.scatter_reduce_(0, target, logits.detach(), reduce="amax", include_self=True)
    numerator = (logits - maxima[target]).exp()
    denominator = logits.new_zeros(num_nodes).index_add_(0, target, numerator)
    return numerator / denominator[target].clamp_min(torch.finfo(logits.dtype).tiny)


class GATConv(nn.Module):
    """Parameter-free dot-product attention, normalized over incoming edges."""
    def forward(self, x, edges):
        source, target = edges
        alpha = edge_softmax((x[source] * x[target]).sum(dim=1), target, len(x))
        out = torch.zeros_like(x).index_add_(0, target, x[source] * alpha[:, None])
        return out, alpha


class SAGEConv(nn.Module):
    def __init__(self, aggregation="mean"):
        super().__init__()
        self.aggregation = aggregation

    def forward(self, x, edges, weights):
        source, target = edges
        out = torch.zeros_like(x).index_add_(0, target, x[source] * weights[:, None])
        if self.aggregation == "mean":
            degree = torch.bincount(target, minlength=len(x)).to(x.dtype).clamp_min(1)
            out = out / degree[:, None]
        return out


class EGCN(nn.Module):
    def __init__(self, num_nodes, embedding_dim, aggregation):
        super().__init__()
        self.id_embedding = nn.Parameter(torch.empty(num_nodes, embedding_dim))
        nn.init.xavier_normal_(self.id_embedding)
        self.conv = SAGEConv(aggregation)

    def forward(self, edges, weights):
        x = F.normalize(self.id_embedding, dim=1)
        first = F.leaky_relu(self.conv(x, edges, weights))
        second = F.leaky_relu(self.conv(first, edges, weights))
        return x + first + second


class CGCN(nn.Module):
    def __init__(self, num_users, feature_dim, content_dim, num_routing):
        super().__init__()
        self.num_users, self.num_routing = num_users, num_routing
        self.preference = nn.Parameter(torch.empty(num_users, content_dim))
        nn.init.xavier_normal_(self.preference)
        self.projection = nn.Linear(feature_dim, content_dim)
        self.attention = GATConv()

    def forward(self, features, routing_edges, bidirectional_edges):
        items = F.normalize(F.leaky_relu(self.projection(features)), dim=1)
        preference = F.normalize(self.preference, dim=1)
        for _ in range(self.num_routing):
            x = torch.cat((preference, items), dim=0)
            messages, _ = self.attention(x, routing_edges)
            preference = F.normalize(preference + messages[:self.num_users], dim=1)
        x = torch.cat((preference, items), dim=0)
        messages, alpha = self.attention(x, bidirectional_edges)
        return x + F.leaky_relu(messages), alpha


class GRCN(nn.Module):
    def __init__(self, data, embedding_dim=64, reg_weight=1e-4, content_dim=None,
                 dropout=0.1, num_routing=2, aggr_mode="mean", weight_mode="confid",
                 fusion_mode="mean", pruning=True):
        super().__init__()
        content_dim = embedding_dim if content_dim is None else content_dim
        if min(embedding_dim, content_dim) < 1 or num_routing < 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid dimensions, routing count or dropout")
        if aggr_mode not in ("mean", "add") or weight_mode not in ("mean", "max", "confid"):
            raise ValueError("Invalid aggregation or weight mode")
        if fusion_mode not in ("mean", "concat", "id"):
            raise ValueError("Invalid fusion mode")
        if fusion_mode == "mean" and content_dim != embedding_dim:
            raise ValueError("Mean fusion requires content-dim == embedding-dim")
        self.num_users, self.num_items = data.num_users, data.num_items
        self.reg_weight, self.dropout = reg_weight, dropout
        self.weight_mode, self.fusion_mode, self.pruning = weight_mode, fusion_mode, pruning
        # Source-to-target convention: routing must send item content to users.
        pairs = torch.from_numpy(data.train_pairs.copy()).T
        pairs[1] += self.num_users
        self.register_buffer("routing_edges", pairs.flip(0), persistent=False)
        self.id_gcn = EGCN(self.num_users + self.num_items, embedding_dim, aggr_mode)
        self.branches = nn.ModuleDict()
        for name, feature in data.features.items():
            self.register_buffer(name + "_features", feature, persistent=False)
            self.branches[name] = CGCN(self.num_users, feature.shape[1], content_dim, num_routing)
        if weight_mode == "confid":
            self.model_specific_conf = nn.Parameter(torch.empty(self.num_users + self.num_items, len(self.branches)))
            nn.init.xavier_normal_(self.model_specific_conf)
        else:
            self.register_parameter("model_specific_conf", None)
        self.output_dim = embedding_dim + len(self.branches) * content_dim if fusion_mode == "concat" else embedding_dim

    def forward(self):
        edges = self.routing_edges
        if self.training and self.dropout:
            edges = edges[:, torch.rand(edges.shape[1], device=edges.device) >= self.dropout]
        bidirectional = torch.cat((edges, edges.flip(0)), dim=1)
        content, attention = [], []
        for name, branch in self.branches.items():
            representation, alpha = branch(getattr(self, name + "_features"), edges, bidirectional)
            content.append(representation)
            attention.append(alpha)
        weights = torch.stack(attention, dim=1)
        if self.weight_mode == "mean":
            weights = weights.mean(dim=1)
        elif self.weight_mode == "max":
            weights = weights.max(dim=1).values
        else:
            # Preserve source-node confidence for each directed message.
            weights = (weights * self.model_specific_conf[bidirectional[0]]).max(dim=1).values
        if self.pruning:
            weights = F.relu(weights)
        identity = self.id_gcn(bidirectional, weights)
        if self.fusion_mode == "id":
            return identity
        if self.fusion_mode == "concat":
            return torch.cat([identity] + content, dim=1)
        return torch.stack([identity] + content).mean(dim=0)

    def encode(self):
        return self.forward().split((self.num_users, self.num_items))

    def compute_loss(self, users, positive, negative):
        user_emb, item_emb = self.encode()
        margin = (user_emb[users] * (item_emb[positive] - item_emb[negative])).sum(dim=1)
        bpr = F.softplus(-margin).mean()
        ids = self.id_gcn.id_embedding
        reg = ids[users].square().mean() + 0.5 * (
            ids[positive + self.num_users].square().mean() + ids[negative + self.num_users].square().mean())
        reg = reg + sum(branch.preference[users].square().mean() for branch in self.branches.values())
        return bpr + self.reg_weight * reg


def main():
    parser = build_parser("GRCN")
    parser.add_argument("--content-dim", type=int, default=None, help="defaults to embedding-dim")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-routing", type=int, default=2)
    parser.add_argument("--aggr-mode", choices=("mean", "add"), default="mean")
    parser.add_argument("--weight-mode", choices=("confid", "mean", "max"), default="confid")
    parser.add_argument("--fusion-mode", choices=("mean", "concat", "id"), default="mean")
    parser.add_argument("--no-pruning", action="store_true")
    args = parser.parse_args()
    args.content_dim = args.embedding_dim if args.content_dim is None else args.content_dim
    if min(args.embedding_dim, args.content_dim) < 1 or args.num_routing < 0 or not 0 <= args.dropout < 1:
        parser.error("Dimensions must be positive; num-routing >= 0; 0 <= dropout < 1")
    if args.fusion_mode == "mean" and args.content_dim != args.embedding_dim:
        parser.error("Mean fusion requires content-dim == embedding-dim")
    seed_everything(args.seed)
    data = RecommendationData(args.data_root, args.dataset, args.core, args.modalities)
    model = GRCN(data, args.embedding_dim, args.reg_weight, args.content_dim, args.dropout,
                 args.num_routing, args.aggr_mode, args.weight_mode, args.fusion_mode, not args.no_pruning)
    args.output_embedding_dim = model.output_dim
    print(f"GRCN fusion: {args.fusion_mode}; scoring embedding width: {model.output_dim}", flush=True)
    run_experiment(model, data, args)


if __name__ == "__main__":
    main()
