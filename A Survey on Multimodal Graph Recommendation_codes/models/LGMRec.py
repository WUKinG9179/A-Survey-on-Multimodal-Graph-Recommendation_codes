"""LGMRec with shared benchmarking and deterministic hypergraph inference."""
from itertools import combinations

import torch
from torch import nn
from torch.nn import functional as F

from utility.data import RecommendationData
from utility.experiment import build_parser, run_experiment, seed_everything


class HGNNLayer(nn.Module):
    def __init__(self, layers=1):
        super().__init__()
        self.layers = layers

    def forward(self, item_hyper, user_hyper, items):
        for _ in range(self.layers):
            latent = item_hyper.T @ items
            items = item_hyper @ latent
            users = user_hyper @ latent
        return users, items


class LGMRec(nn.Module):
    def __init__(self, data, embedding_dim=64, reg_weight=1e-4, n_layers=2,
                 ssl_alpha=1e-4, n_mm_layers=2, n_hyper_layers=1, hyper_num=4,
                 keep_rate=0.2, tau=0.2, alpha=0.2, ssl_chunk_size=128):
        super().__init__()
        if min(embedding_dim, n_hyper_layers, hyper_num, ssl_chunk_size) < 1 or min(n_layers, n_mm_layers) < 0:
            raise ValueError("Invalid dimensions, layer counts or chunk size")
        if not 0 < keep_rate <= 1 or tau <= 0 or min(reg_weight, ssl_alpha, alpha) < 0:
            raise ValueError("Invalid keep rate, temperature or loss weights")
        self.num_users, self.num_items = data.num_users, data.num_items
        self.output_dim, self.reg_weight = embedding_dim, reg_weight
        self.n_layers, self.n_mm_layers = n_layers, n_mm_layers
        self.ssl_alpha, self.tau, self.alpha = ssl_alpha, tau, alpha
        self.ssl_chunk_size = ssl_chunk_size
        pairs = torch.from_numpy(data.train_pairs.copy()).T
        users, items = pairs
        self.register_buffer("interaction", torch.sparse_coo_tensor(pairs, torch.ones(len(users)),
                             (self.num_users, self.num_items)).coalesce(), persistent=False)
        u_degree = torch.bincount(users, minlength=self.num_users).float() + 1e-7
        i_degree = torch.bincount(items, minlength=self.num_items).float() + 1e-7
        self.register_buffer("inverse_user_degree", u_degree.reciprocal().unsqueeze(1), persistent=False)
        item_nodes = items + self.num_users
        indices = torch.stack((torch.cat((users, item_nodes)), torch.cat((item_nodes, users))))
        values = (u_degree[users].rsqrt() * i_degree[items].rsqrt()).repeat(2)
        size = self.num_users + self.num_items
        self.register_buffer("norm_adj", torch.sparse_coo_tensor(indices, values, (size, size)).coalesce(), persistent=False)
        self.user_embedding = nn.Embedding(self.num_users, embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.projections = nn.ParameterDict()
        self.hyper_projections = nn.ParameterDict()
        for name, feature in data.features.items():
            self.register_buffer(name + "_features", feature, persistent=False)
            self.projections[name] = nn.Parameter(torch.empty(feature.shape[1], embedding_dim))
            self.hyper_projections[name] = nn.Parameter(torch.empty(feature.shape[1], hyper_num))
            nn.init.xavier_uniform_(self.projections[name])
            nn.init.xavier_uniform_(self.hyper_projections[name])
        self.drop = nn.Dropout(1 - keep_rate)
        self.hgnn = HGNNLayer(n_hyper_layers)

    def cge(self):
        x = torch.cat((self.user_embedding.weight, self.item_embedding.weight))
        layers = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.norm_adj, x)
            layers.append(x)
        return torch.stack(layers).mean(dim=0)

    def mge(self, name):
        items = getattr(self, name + "_features") @ self.projections[name]
        users = torch.sparse.mm(self.interaction, items) * self.inverse_user_degree
        x = torch.cat((users, items))
        for _ in range(self.n_mm_layers):
            x = torch.sparse.mm(self.norm_adj, x)
        return x

    def membership(self, logits):
        if self.training:
            return F.gumbel_softmax(logits, tau=self.tau, hard=False, dim=1)
        return torch.softmax(logits / self.tau, dim=1)

    def forward(self):
        collaborative = self.cge()
        local = collaborative + sum(F.normalize(self.mge(name), dim=1) for name in self.projections)
        hyper = {}
        for name in self.projections:
            item_logits = getattr(self, name + "_features") @ self.hyper_projections[name]
            user_logits = torch.sparse.mm(self.interaction, item_logits)
            item_membership = self.drop(self.membership(item_logits))
            user_membership = self.drop(self.membership(user_logits))
            hyper[name] = self.hgnn(item_membership, user_membership, collaborative[self.num_users:])
        global_emb = sum(torch.cat(pair) for pair in hyper.values())
        users, items = (local + self.alpha * F.normalize(global_emb, dim=1)).split((self.num_users, self.num_items))
        return users, items, hyper

    def encode(self):
        users, items, _ = self.forward()
        return users, items

    def ssl_triple_loss(self, first, positive, candidates):
        """Original SUM reduction and full-catalog negatives, stable logsumexp."""
        first, positive, candidates = (F.normalize(x, dim=1) for x in (first, positive, candidates))
        total = first.new_zeros(())
        for start in range(0, len(first), self.ssl_chunk_size):
            queries = first[start:start + self.ssl_chunk_size]
            positives = positive[start:start + self.ssl_chunk_size]
            # Checkpoint avoids retaining every chunk's catalog logits for backward.
            def block_loss(q, p, c):
                return (torch.logsumexp(q @ c.T / self.tau, dim=1)
                        - (q * p).sum(dim=1) / self.tau).sum()
            if torch.is_grad_enabled() and (queries.requires_grad or candidates.requires_grad):
                from torch.utils.checkpoint import checkpoint
                total = total + checkpoint(block_loss, queries, positives, candidates, use_reentrant=False)
            else:
                total = total + block_loss(queries, positives, candidates)
        return total

    def compute_loss(self, users, positive, negative):
        user_emb, item_emb, hyper = self.forward()
        u, p, n = user_emb[users], item_emb[positive], item_emb[negative]
        loss = F.softplus(-(u * (p - n)).sum(dim=1)).mean()
        loss = loss + self.reg_weight * (u.square().mean() + p.square().mean() + n.square().mean())
        if self.ssl_alpha:
            losses = []
            # Two modalities preserve the original directed pair; M modalities
            # average all unordered pairs with their configured modality order.
            for first, second in combinations(hyper.values(), 2):
                losses.append(self.ssl_triple_loss(first[0][users], second[0][users], second[0])
                              + self.ssl_triple_loss(first[1][positive], second[1][positive], second[1]))
            if losses:
                loss = loss + self.ssl_alpha * torch.stack(losses).mean()
        return loss


def main():
    parser = build_parser("LGMRec")
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-mm-layers", type=int, default=2)
    parser.add_argument("--n-hyper-layers", type=int, default=1)
    parser.add_argument("--hyper-num", type=int, default=4)
    parser.add_argument("--keep-rate", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--ssl-alpha", type=float, default=1e-4)
    parser.add_argument("--ssl-chunk-size", type=int, default=128)
    args = parser.parse_args()
    if min(args.embedding_dim, args.n_hyper_layers, args.hyper_num, args.ssl_chunk_size) < 1 or min(args.n_layers, args.n_mm_layers) < 0:
        parser.error("Invalid dimensions, layer counts or ssl-chunk-size")
    if not 0 < args.keep_rate <= 1 or args.tau <= 0 or min(args.alpha, args.ssl_alpha) < 0:
        parser.error("Invalid keep-rate, tau or loss weights")
    seed_everything(args.seed)
    data = RecommendationData(args.data_root, args.dataset, args.core, args.modalities)
    model = LGMRec(data, args.embedding_dim, args.reg_weight, args.n_layers, args.ssl_alpha,
                   args.n_mm_layers, args.n_hyper_layers, args.hyper_num, args.keep_rate,
                   args.tau, args.alpha, args.ssl_chunk_size)
    args.output_embedding_dim = model.output_dim
    args.eval_membership = "deterministic_softmax"
    args.ssl_reduction = "sum_batch_mean_modality_pairs_full_catalog"
    run_experiment(model, data, args)


if __name__ == "__main__":
    main()
