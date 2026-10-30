"""GUME (CIKM 2024) adapted from the author's implementation.

Source: https://github.com/NanGongNingYi/GUME
Commit: 44a4dcec975fa054a689ff072ec390f9992ad7f8
Reuses the local MGCN modality encoding/fusion components; GUME's enhanced
behavior graph, extended user interests and complete objectives are added here.
"""
from itertools import combinations

import torch
from torch import nn
from torch.nn import functional as F

from MGCN import MGCN
from utility.data import RecommendationData
from utility.experiment import build_parser, run_experiment, seed_everything

SOURCE_COMMIT = "44a4dcec975fa054a689ff072ec390f9992ad7f8"


def semantic_intersection(graphs):
    """Directed neighbors present in every modality's top-k, excluding self.

    Works for arbitrary k; does not assume the reference's hard-coded k=10.
    """
    size = graphs[0].shape[0]
    keys = []
    for graph in graphs:
        row, col = graph.coalesce().indices()
        valid = row != col
        keys.append(torch.unique(row[valid] * size + col[valid]))
    common, counts = torch.cat(keys).unique(return_counts=True)
    common = common[counts == len(graphs)]
    return torch.stack((common // size, common % size))


def enhance_graph(base, item_edges, num_users, num_items):
    """Binary train UI edges plus semantic II edges, then degree normalization.

    II edges retain their directed kNN intersection orientation, as in source.
    R is extracted AFTER this enhancement and degree normalization.
    """
    indices = torch.cat((base.coalesce().indices(), item_edges + num_users), dim=1)
    size = num_users + num_items
    raw = torch.sparse_coo_tensor(indices, torch.ones(indices.shape[1], device=indices.device),
                                  (size, size)).coalesce()
    row, col = raw.indices()
    degree = raw.values().new_zeros(size).index_add_(0, row, raw.values())
    inverse = torch.where(degree > 0, degree.clamp_min(1).rsqrt(), torch.zeros_like(degree))
    values = raw.values() * inverse[row] * inverse[col]
    graph = torch.sparse_coo_tensor(raw.indices(), values, raw.shape).coalesce()
    mask = (row < num_users) & (col >= num_users)
    interaction = torch.sparse_coo_tensor(torch.stack((row[mask], col[mask] - num_users)),
                                          values[mask], (num_users, num_items)).coalesce()
    return graph, interaction


class GUME(MGCN):
    def __init__(self, data, embedding_dim=64, reg_weight=1e-4, n_layers=2, n_ui_layers=3,
                 ii_topk=10, bm_weight=0.01, um_weight=0.01, alignment_weight=0.1,
                 extended_reg_weight=0.1, bm_temp=0.4, um_temp=0.1, noise_scale=0.1,
                 graph_block_size=256, freeze_features=False):
        super().__init__(data, embedding_dim=embedding_dim, reg_weight=reg_weight,
                         n_layers=n_layers, n_ui_layers=n_ui_layers, ssl_alpha=0,
                         ii_topk=ii_topk, graph_block_size=graph_block_size,
                         freeze_features=freeze_features)
        if min(bm_temp, um_temp) <= 0 or min(bm_weight, um_weight, alignment_weight,
                                           extended_reg_weight, noise_scale) < 0:
            raise ValueError("Temperatures must be positive; objective weights/noise nonnegative")
        self.bm_weight, self.um_weight, self.alignment_weight = bm_weight, um_weight, alignment_weight
        self.extended_reg_weight = extended_reg_weight
        self.bm_temp, self.um_temp, self.noise_scale = bm_temp, um_temp, noise_scale
        self.extended_users = nn.ModuleDict()
        for name in self.feature_embeddings:
            self.extended_users[name] = nn.Embedding(self.num_users, embedding_dim)
            nn.init.xavier_uniform_(self.extended_users[name].weight)
        self.register_buffer("base_ui_graph", self.norm_adj, persistent=False)
        self._enhanced_ready = False

    def prepare_graphs(self):
        super().prepare_graphs()
        if not self._enhanced_ready:
            with torch.inference_mode(False), torch.no_grad():
                common = semantic_intersection([getattr(self, name + "_graph") for name in self.feature_embeddings])
                self.norm_adj, self.interaction = enhance_graph(self.base_ui_graph, common, self.num_users, self.num_items)
            self.semantic_edge_count = common.shape[1]
            self._enhanced_ready = True

    def conv_ui(self, users, items):
        x = torch.cat((users, items))
        layers = [x]
        for _ in range(self.n_ui_layers):
            x = torch.sparse.mm(self.norm_adj, x)
            layers.append(x)
        return torch.stack(layers).mean(dim=0)

    def forward(self, with_auxiliary=False):
        self.prepare_graphs()
        behavior = self.conv_ui(self.user_embedding.weight, self.item_embedding.weight)
        explicit, extended = [], []
        for name, embedding in self.feature_embeddings.items():
            items = self.item_embedding.weight * self.purifiers[name](self.projections[name](embedding.weight))
            for _ in range(self.n_layers):
                items = torch.sparse.mm(getattr(self, name + "_graph"), items)
            users = torch.sparse.mm(self.interaction, items)
            explicit.append(torch.cat((users, items)))
            if with_auxiliary:
                extended.append(self.conv_ui(self.extended_users[name].weight, items))
        integration = self.fuse_views(behavior, explicit)
        result = behavior + integration
        if with_auxiliary:
            return result, integration, behavior, torch.stack(extended).mean(dim=0), explicit
        return result

    def encode(self):
        return self.forward().split((self.num_users, self.num_items))

    @staticmethod
    def info_nce(first, second, temperature):
        logits = F.normalize(first, dim=1) @ F.normalize(second, dim=1).T / temperature
        return F.cross_entropy(logits, torch.arange(len(first), device=first.device))

    @staticmethod
    def align_modalities(first, second):
        # Scalar mean/variance over all users/items and dimensions (source behavior).
        return (first.var() - second.var()).abs() + (first.mean() - second.mean()).abs()

    def noise_loss(self, user_ids, all_users):
        # Generate one noise vector per UNIQUE selected user so duplicate batch IDs
        # share noise, equivalent to perturbing the full user matrix then indexing.
        unique, inverse = torch.unique(user_ids, return_inverse=True)
        selected = all_users[unique]
        def perturb():
            noise = F.normalize(torch.rand_like(selected), dim=1)
            return (selected + selected.sign() * noise * self.noise_scale)[inverse]
        return self.info_nce(perturb(), perturb(), self.um_temp)

    def compute_loss(self, users, positive, negative):
        result, integration, behavior, extended, explicit = self.forward(with_auxiliary=True)
        p_nodes, n_nodes = positive + self.num_users, negative + self.num_users
        u, p, n = result[users], result[p_nodes], result[n_nodes]
        loss = F.softplus(-(u * (p - n)).sum(dim=1)).mean()
        # Keep half-sum regularization, scaled by actual batch length (incl. tail).
        loss = loss + self.reg_weight * 0.5 * (u.square().sum() + p.square().sum() + n.square().sum()) / len(users)
        loss = loss + self.extended_reg_weight * 0.5 * extended[p_nodes].square().sum() / len(users)
        if self.alignment_weight:
            alignment = [self.align_modalities(a, b) for a, b in combinations(explicit, 2)]
            if alignment:
                loss = loss + self.alignment_weight * torch.stack(alignment).mean()
        if self.bm_weight:
            loss = loss + self.bm_weight * (self.info_nce(integration[users], behavior[users], self.bm_temp)
                                            + self.info_nce(integration[p_nodes], behavior[p_nodes], self.bm_temp))
        if self.um_weight:
            user_loss = self.info_nce(extended[users], integration[users], self.um_temp)
            user_loss = user_loss + self.noise_loss(users, integration[:self.num_users])
            user_loss = user_loss + self.noise_loss(users, extended[:self.num_users])
            loss = loss + self.um_weight * user_loss
        return loss


def main():
    parser = build_parser("GUME")
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-ui-layers", type=int, default=3)
    parser.add_argument("--ii-topk", type=int, default=10)
    parser.add_argument("--bm-weight", type=float, default=0.01)
    parser.add_argument("--um-weight", type=float, default=0.01)
    parser.add_argument("--alignment-weight", type=float, default=0.1)
    parser.add_argument("--extended-reg-weight", type=float, default=0.1)
    parser.add_argument("--bm-temp", type=float, default=0.4)
    parser.add_argument("--um-temp", type=float, default=0.1)
    parser.add_argument("--noise-scale", type=float, default=0.1)
    parser.add_argument("--graph-block-size", type=int, default=256)
    parser.add_argument("--freeze-features", action="store_true")
    args = parser.parse_args()
    if min(args.embedding_dim, args.ii_topk, args.graph_block_size) < 1 or min(args.n_layers, args.n_ui_layers) < 0:
        parser.error("Invalid embedding width, layer count, topk or block size")
    if min(args.bm_temp, args.um_temp) <= 0 or min(args.bm_weight, args.um_weight, args.alignment_weight, args.extended_reg_weight, args.noise_scale) < 0:
        parser.error("Temperatures must be positive; loss weights/noise nonnegative")
    seed_everything(args.seed)
    data = RecommendationData(args.data_root, args.dataset, args.core, args.modalities)
    model = GUME(data, args.embedding_dim, args.reg_weight, args.n_layers, args.n_ui_layers,
                 args.ii_topk, args.bm_weight, args.um_weight, args.alignment_weight,
                 args.extended_reg_weight, args.bm_temp, args.um_temp, args.noise_scale,
                 args.graph_block_size, args.freeze_features)
    args.output_embedding_dim = model.output_dim
    args.reference_repository = "https://github.com/NanGongNingYi/GUME"
    args.reference_commit = SOURCE_COMMIT
    args.graph_enhancement = "intersection_all_modalities_directed_item_edges"
    args.default_model_profile = "author_active_baby_config_with_shared_reg_weight"
    run_experiment(model, data, args)


if __name__ == "__main__":
    main()
