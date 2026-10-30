"""MMGCN using shared benchmark tools and native PyTorch sparse aggregation.

Neighbor aggregation follows the official MMGCN BaseModel:
https://github.com/iLearn-Lab/MM19-MMGCN/blob/master/BaseModel.py
The supplied four-layer architecture is retained by default.
"""
import torch
from torch import nn
from torch.nn import functional as F

from utility.data import RecommendationData
from utility.experiment import build_parser, run_experiment, seed_everything


class GCN(nn.Module):
    def __init__(self, num_users, feature_dim, embedding_dim, layers=4,
                 latent_dim=None, concatenate=False, has_id=True):
        super().__init__()
        width = latent_dim or feature_dim
        self.projection = nn.Linear(feature_dim, width) if latent_dim else nn.Identity()
        self.preference = nn.Parameter(torch.empty(num_users, width))
        nn.init.xavier_normal_(self.preference)
        self.concatenate, self.has_id = concatenate, has_id
        self.neighbor = nn.ModuleList()
        self.self_linear = nn.ModuleList()
        self.fusion = nn.ModuleList()
        for _ in range(layers):
            self.neighbor.append(nn.Linear(width, width, bias=False))
            self.self_linear.append(nn.Linear(width, embedding_dim))
            self.fusion.append(nn.Linear(width + embedding_dim if concatenate else width,
                                         embedding_dim))
            width = embedding_dim
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features, id_embedding, adjacency):
        x = F.normalize(torch.cat((self.preference, self.projection(features)), dim=0), dim=1)
        for neighbor, self_linear, fusion in zip(self.neighbor, self.self_linear, self.fusion):
            h = F.leaky_relu(torch.sparse.mm(adjacency, neighbor(x)))
            own = F.leaky_relu(self_linear(x))
            if self.has_id:
                own = own + id_embedding
            x = F.leaky_relu(fusion(torch.cat((h, own), dim=1)) if self.concatenate
                             else fusion(h) + own)
        return x


class MMGCN(nn.Module):
    """Public item IDs are local zero-based IDs; only graph nodes use offsets."""
    def __init__(self, data, embedding_dim=64, reg_weight=1e-4, layers=4,
                 aggr_mode="mean", concatenate=False, has_id=True, visual_dim=256):
        super().__init__()
        self.num_users, self.num_items = data.num_users, data.num_items
        self.reg_weight = reg_weight
        self.register_buffer("adjacency", data.adjacency(aggr_mode), persistent=False)
        self.id_embedding = nn.Parameter(torch.empty(data.num_users + data.num_items, embedding_dim))
        nn.init.xavier_normal_(self.id_embedding)
        self.branches = nn.ModuleDict()
        for name, feature in data.features.items():
            self.register_buffer(name + "_features", feature, persistent=False)
            self.branches[name] = GCN(data.num_users, feature.shape[1], embedding_dim,
                                      layers, visual_dim if name == "image" else None,
                                      concatenate, has_id)

    def forward(self):
        outputs = [branch(getattr(self, name + "_features"), self.id_embedding, self.adjacency)
                   for name, branch in self.branches.items()]
        return torch.stack(outputs).mean(dim=0)

    def encode(self):
        return self.forward().split((self.num_users, self.num_items))

    def compute_loss(self, users, positive, negative):
        user_emb, item_emb = self.encode()
        margin = (user_emb[users] * (item_emb[positive] - item_emb[negative])).sum(dim=1)
        bpr = F.softplus(-margin).mean()
        # Preserve the supplied paired-ID and visual-preference regularizer.
        reg = self.id_embedding[users].square().mean()
        reg = reg + 0.5 * (self.id_embedding[positive + self.num_users].square().mean()
                           + self.id_embedding[negative + self.num_users].square().mean())
        reg = reg + self.branches["image"].preference.square().mean()
        return bpr + self.reg_weight * reg


def main():
    parser = build_parser("MMGCN")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--visual-dim", type=int, default=256)
    parser.add_argument("--aggr-mode", choices=("mean", "add"), default="mean")
    parser.add_argument("--concatenate", action="store_true")
    parser.add_argument("--no-id", action="store_true")
    args = parser.parse_args()
    if args.layers < 1 or args.visual_dim < 1:
        parser.error("layers and visual-dim must be positive")
    seed_everything(args.seed)
    data = RecommendationData(args.data_root, args.dataset, args.core, args.modalities)
    model = MMGCN(data, args.embedding_dim, args.reg_weight, args.layers,
                  args.aggr_mode, args.concatenate, not args.no_id, args.visual_dim)
    run_experiment(model, data, args)


if __name__ == "__main__":
    main()
