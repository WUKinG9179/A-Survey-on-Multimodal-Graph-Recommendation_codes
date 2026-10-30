"""MMGCL adapter preserving the supplied file's effective default objective.

ED+MM+CN is a legacy label: the supplied code never adds its computed CN term.
This adapter records that limitation and does not invent a replacement objective.
"""
import torch
from torch import nn
from torch.nn import functional as F

from utility.data import RecommendationData
from utility.experiment import build_parser, run_experiment, seed_everything


def normalized_ui_graph(pairs, num_users, num_items):
    users, items = pairs.T
    items = items + num_users
    row, col = torch.cat((users, items)), torch.cat((items, users))
    size = num_users + num_items
    degree = torch.bincount(row, minlength=size).float().clamp_min(1)
    values = degree[row].rsqrt() * degree[col].rsqrt()
    return torch.sparse_coo_tensor(torch.stack((row, col)), values, (size, size)).coalesce()


class MMGCL(nn.Module):
    def __init__(self, data, embedding_dim=64, reg_weight=1e-4, n_layers=2,
                 ssl_alpha=0.1, ssl_temp=0.2, dropout=0.05,
                 ssl_task="ED+MM+CN", use_regularization=False):
        super().__init__()
        if embedding_dim < 1 or n_layers < 0 or ssl_alpha < 0 or ssl_temp <= 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid dimensions, layers, SSL parameters or dropout")
        if ssl_task not in ("ED+MM+CN", "ED+MM", "ED", "MM"):
            raise ValueError("Invalid ssl_task")
        self.num_users, self.num_items = data.num_users, data.num_items
        self.output_dim, self.n_layers = embedding_dim, n_layers
        self.reg_weight, self.use_regularization = reg_weight, use_regularization
        self.ssl_alpha, self.ssl_temp = ssl_alpha, ssl_temp
        self.dropout_rate, self.ssl_task = dropout, ssl_task
        pairs = torch.from_numpy(data.train_pairs.copy())
        self.register_buffer("train_pairs", pairs, persistent=False)
        self.register_buffer("norm_adj", normalized_ui_graph(pairs, self.num_users, self.num_items), persistent=False)
        self.user_embeddings = nn.Embedding(self.num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(self.num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.item_embeddings.weight)
        self.projections = nn.ModuleDict()
        for name, features in data.features.items():
            self.register_buffer(name + "_features", F.normalize(features, dim=1), persistent=False)
            self.projections[name] = nn.Linear(features.shape[1], embedding_dim)
            nn.init.xavier_uniform_(self.projections[name].weight)
        width = embedding_dim * (1 + len(self.projections))
        self.read_user, self.read_item = nn.Linear(width, embedding_dim), nn.Linear(width, embedding_dim)
        nn.init.xavier_uniform_(self.read_user.weight)
        nn.init.xavier_uniform_(self.read_item.weight)

    def graph_augment(self, kind):
        pairs = self.train_pairs
        device = pairs.device
        if kind == "ED":
            count = int(len(pairs) * (1 - self.dropout_rate))
            pairs = pairs[torch.randperm(len(pairs), device=device)[:count]]
        elif kind == "MM":
            keep_users = torch.ones(self.num_users, dtype=torch.bool, device=device)
            keep_items = torch.ones(self.num_items, dtype=torch.bool, device=device)
            keep_users[torch.randperm(self.num_users, device=device)[:int(self.num_users * self.dropout_rate)]] = False
            keep_items[torch.randperm(self.num_items, device=device)[:int(self.num_items * self.dropout_rate)]] = False
            pairs = pairs[keep_users[pairs[:, 0]] & keep_items[pairs[:, 1]]]
        else:
            raise ValueError("Augmentation must be ED or MM")
        return normalized_ui_graph(pairs, self.num_users, self.num_items)

    def sgl_encoder(self, items, graph):
        x = torch.cat((self.user_embeddings.weight, items))
        layers = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(graph, x)
            layers.append(x)
        return torch.stack(layers).mean(dim=0).split((self.num_users, self.num_items))

    def projected_features(self):
        return {name: projection(getattr(self, name + "_features")) for name, projection in self.projections.items()}

    def view(self, projected, kind=None, masked_modality=None):
        graph = self.norm_adj if kind is None else self.graph_augment(kind)
        if kind == "MM" and masked_modality is None:
            names = list(projected)
            masked_modality = names[int(torch.randint(len(names), (), device=self.train_pairs.device))]
        identity_graph = graph if kind == "ED" else self.norm_adj
        parts = [self.sgl_encoder(self.item_embeddings.weight, identity_graph)]
        for name, features in projected.items():
            modality_graph = graph if kind == "ED" or (kind == "MM" and name == masked_modality) else self.norm_adj
            parts.append(self.sgl_encoder(features, modality_graph))
        users = self.read_user(torch.cat([part[0] for part in parts], dim=1))
        items = self.read_item(torch.cat([part[1] for part in parts], dim=1))
        return users, items

    def encode(self):
        # No graph augmentation or RNG consumption during full-ranking inference.
        return self.view(self.projected_features())

    def forward(self):
        return self.encode()

    def contrastive_loss(self, projected, users, positive):
        task = self.ssl_task
        first = self.view(projected, "MM" if task == "MM" else "ED")
        second = self.view(projected, "ED" if task == "ED" else "MM")
        u1, i1 = F.normalize(first[0][users], dim=1), F.normalize(first[1][positive], dim=1)
        u2, i2 = F.normalize(second[0][users], dim=1), F.normalize(second[1][positive], dim=1)
        labels = torch.arange(len(users), device=users.device)
        first_loss = F.cross_entropy(u1 @ i1.T / self.ssl_temp, labels)
        if task == "ED+MM":
            # Preserve the supplied nondefault branch, including item self-contrast.
            second_logits = i1 @ i1.T
        elif task in ("ED", "MM"):
            second_logits = u2 @ i2.T
        else:
            # Effective local ED+MM+CN behavior: the separate CN term was unused.
            second_logits = u1 @ i2.T
        return first_loss + F.cross_entropy(second_logits / self.ssl_temp, labels)

    def compute_loss(self, users, positive, negative):
        projected = self.projected_features()
        user_emb, item_emb = self.view(projected)
        u, p, n = user_emb[users], item_emb[positive], item_emb[negative]
        loss = F.softplus(-(u * (p - n)).sum(dim=1)).mean()
        if self.use_regularization:
            loss = loss + self.reg_weight * (u.square().mean() + p.square().mean() + n.square().mean())
        if self.ssl_alpha:
            loss = loss + self.ssl_alpha * self.contrastive_loss(projected, users, positive)
        return loss


def main():
    parser = build_parser("MMGCL")
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--ssl-alpha", type=float, default=0.1)
    parser.add_argument("--ssl-temp", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--ssl-task", choices=("ED+MM+CN", "ED+MM", "ED", "MM"), default="ED+MM+CN")
    parser.add_argument("--use-regularization", action="store_true")
    args = parser.parse_args()
    if args.embedding_dim < 1 or args.n_layers < 0 or args.ssl_alpha < 0 or args.ssl_temp <= 0 or not 0 <= args.dropout < 1:
        parser.error("Invalid dimensions, n-layers, ssl-alpha, ssl-temp or dropout")
    seed_everything(args.seed)
    data = RecommendationData(args.data_root, args.dataset, args.core, args.modalities)
    model = MMGCL(data, args.embedding_dim, args.reg_weight, args.n_layers, args.ssl_alpha,
                  args.ssl_temp, args.dropout, args.ssl_task, args.use_regularization)
    args.output_embedding_dim = model.output_dim
    args.cn_term_enabled = False
    args.objective_implementation = "local_effective_loss"
    if args.ssl_task == "ED+MM+CN":
        print("Using local ED+MM+CN effective loss: two contrastive terms; CN term is not included.", flush=True)
    run_experiment(model, data, args)


if __name__ == "__main__":
    main()
