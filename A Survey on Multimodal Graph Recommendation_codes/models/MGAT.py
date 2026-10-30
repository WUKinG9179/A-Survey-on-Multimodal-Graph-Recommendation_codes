"""MGAT adapter for the shared recommendation benchmark.

Retains the supplied gated dot-product attention and three-layer ID fusion.
Default layer mean keeps final representations at embedding_dim; --layer-fusion
concat restores the supplied layer concatenation (layers * embedding_dim).
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from utility.data import RecommendationData
from utility.experiment import build_parser, run_experiment, seed_everything


class GraphGAT(nn.Module):
    """Native torch equivalent of the supplied source-to-target MessagePassing.

    Gate uses source degree; softmax is over edges arriving at each target.
    No self loops or dropout, matching the original active code.
    """
    def __init__(self, in_channels, out_channels, normalize=True, bias=True):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.normalize = normalize
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        bound = 1 / math.sqrt(in_channels)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, edge_index):
        source, target = edge_index
        keep = source != target
        source, target = source[keep], target[keep]
        x = x @ self.weight
        degree = torch.bincount(source, minlength=x.shape[0]).to(x.dtype)
        inner = (x[target] * F.leaky_relu(x[source])).sum(dim=-1)
        gate = torch.sigmoid(inner * degree[source].clamp_min(1).rsqrt())
        logits = inner * gate
        # Detaching the stabilizing maximum preserves softmax gradients.
        maxima = x.new_full((x.shape[0],), -torch.inf)
        maxima.scatter_reduce_(0, target, logits.detach(), reduce="amax", include_self=True)
        numerator = (logits - maxima[target]).exp()
        denominator = x.new_zeros(x.shape[0]).index_add_(0, target, numerator)
        attention = numerator / denominator[target].clamp_min(torch.finfo(x.dtype).tiny)
        output = x.new_zeros((x.shape[0], self.out_channels))
        output.index_add_(0, target, x[source] * attention.unsqueeze(1))
        if self.bias is not None:
            output = output + self.bias
        return F.normalize(output, p=2, dim=-1) if self.normalize else output


class GNN(nn.Module):
    def __init__(self, num_users, feature_dim, embedding_dim, latent_dim,
                 layers=3, layer_fusion="mean"):
        super().__init__()
        self.layer_fusion = layer_fusion
        self.preference = nn.Embedding(num_users, latent_dim)
        nn.init.xavier_normal_(self.preference.weight)
        self.projection = nn.Linear(feature_dim, latent_dim)
        self.convs = nn.ModuleList()
        self.self_linear = nn.ModuleList()
        self.gates = nn.ModuleList()
        for width in [latent_dim] + [embedding_dim] * (layers - 1):
            conv = GraphGAT(width, width)
            own, gate = nn.Linear(width, embedding_dim), nn.Linear(width, embedding_dim)
            for weight in (conv.weight, own.weight, gate.weight):
                nn.init.xavier_normal_(weight)
            self.convs.append(conv)
            self.self_linear.append(own)
            self.gates.append(gate)

    def forward(self, features, id_embedding, edge_index):
        x = F.normalize(torch.cat((self.preference.weight, torch.tanh(self.projection(features))), dim=0), dim=1)
        outputs = []
        for conv, own, gate in zip(self.convs, self.self_linear, self.gates):
            h = F.leaky_relu(conv(x, edge_index))
            x_hat = F.leaky_relu(own(x)) + id_embedding
            x = F.leaky_relu(gate(h) + x_hat)
            outputs.append(x)
        if self.layer_fusion == "concat":
            return torch.cat(outputs, dim=1)
        return torch.stack(outputs).mean(dim=0)


class MGAT(nn.Module):
    """Shared adapter: local item IDs, encode(), and scalar compute_loss()."""
    def __init__(self, data, embedding_dim=64, reg_weight=1e-4, layers=3,
                 visual_dim=256, text_dim=100, audio_dim=128, layer_fusion="mean"):
        super().__init__()
        if min(embedding_dim, layers, visual_dim, text_dim, audio_dim) < 1:
            raise ValueError("Embedding widths and layer count must be positive")
        if layer_fusion not in ("mean", "concat"):
            raise ValueError("layer_fusion must be mean or concat")
        self.num_users, self.num_items = data.num_users, data.num_items
        self.reg_weight = reg_weight
        self.output_dim = embedding_dim * layers if layer_fusion == "concat" else embedding_dim
        pairs = torch.from_numpy(data.train_pairs.copy()).T
        pairs[1] += self.num_users
        self.register_buffer("edge_index", torch.cat((pairs, pairs.flip(0)), dim=1), persistent=False)
        self.id_embedding = nn.Embedding(self.num_users + self.num_items, embedding_dim)
        nn.init.xavier_normal_(self.id_embedding.weight)
        self.branches = nn.ModuleDict()
        latent_dims = {"image": visual_dim, "text": text_dim, "audio": audio_dim}
        for name, feature in data.features.items():
            self.register_buffer(name + "_features", feature, persistent=False)
            self.branches[name] = GNN(self.num_users, feature.shape[1], embedding_dim,
                                      latent_dims[name], layers, layer_fusion)

    def forward(self):
        outputs = [branch(getattr(self, name + "_features"), self.id_embedding.weight, self.edge_index)
                   for name, branch in self.branches.items()]
        return torch.stack(outputs).mean(dim=0)

    def encode(self):
        return self.forward().split((self.num_users, self.num_items))

    def compute_loss(self, users, positive, negative):
        user_emb, item_emb = self.encode()
        u, p, n = user_emb[users], item_emb[positive], item_emb[negative]
        margin = (u * (p - n)).sum(dim=-1)
        bpr = F.softplus(-margin).mean()
        # Preserve regularization on the sampled final representations.
        reg = u.square().mean() + p.square().mean() + n.square().mean()
        return bpr + self.reg_weight * reg


def main():
    parser = build_parser("MGAT")
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--visual-dim", type=int, default=256)
    parser.add_argument("--text-dim", type=int, default=100)
    parser.add_argument("--audio-dim", type=int, default=128)
    parser.add_argument("--layer-fusion", choices=("mean", "concat"), default="mean",
                        help="mean keeps output width fixed; concat restores original layer fusion")
    args = parser.parse_args()
    if min(args.embedding_dim, args.layers, args.visual_dim, args.text_dim, args.audio_dim) < 1:
        parser.error("Embedding widths and layer count must be positive")
    seed_everything(args.seed)
    data = RecommendationData(args.data_root, args.dataset, args.core, args.modalities)
    model = MGAT(data, args.embedding_dim, args.reg_weight, args.layers,
                 args.visual_dim, args.text_dim, args.audio_dim, args.layer_fusion)
    # Persist actual scoring width so concat runs cannot be mistaken for 64-dim runs.
    args.output_embedding_dim = model.output_dim
    print(f"MGAT layer fusion: {args.layer_fusion}; scoring embedding width: {model.output_dim}", flush=True)
    run_experiment(model, data, args)


if __name__ == "__main__":
    main()
