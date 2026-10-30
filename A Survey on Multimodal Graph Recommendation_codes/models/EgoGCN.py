"""EgoGCN: Breaking Isolation (ACM Multimedia 2022), shared benchmark adapter.

Reference: https://github.com/feiyuchen7/EgoGCN
Commit: 81bbfef1fd4c50d29f5ee8449a38c7e30f3e8653
Core: Model_routing_21sp.py, GATConv_pre.py, GATConv_ori.py, GCNConv.py.
See utility/README.md for the unified configuration and explicit deviations.
"""
import torch
from torch import nn
from torch.nn import functional as F

from utility.data import RecommendationData
from utility.experiment import build_parser, run_experiment, seed_everything

SOURCE_COMMIT = "81bbfef1fd4c50d29f5ee8449a38c7e30f3e8653"


def incoming_softmax(logits, receivers, num_nodes):
    maximum = logits.new_full((num_nodes,), -torch.inf)
    maximum.scatter_reduce_(0, receivers, logits.detach(), reduce="amax", include_self=True)
    numerator = (logits - maximum[receivers]).exp()
    denominator = logits.new_zeros(num_nodes).index_add_(0, receivers, numerator)
    return numerator / denominator[receivers].clamp_min(torch.finfo(logits.dtype).tiny)


def attention_propagate(x, edges):
    source, target = edges
    alpha = incoming_softmax((x[target] * x[source]).sum(dim=1), target, len(x))
    return torch.zeros_like(x).index_add_(0, target, alpha[:, None] * x[source])


class EdgeModulation(nn.Module):
    """Source-message EGO modulation followed by incoming-neighbor reattention.

    Implements the active central_item branch of the author's GATConv_pre.
    Other modalities supervise each focal modality on the same directed edge.
    """
    def __init__(self, embedding_dim, other_modalities, modulation="hard", threshold=0.6,
                 reattention=True):
        super().__init__()
        if modulation not in ("hard", "soft") or not 0 <= threshold <= 1:
            raise ValueError("Invalid modulation or threshold")
        self.modulation, self.threshold, self.reattention = modulation, threshold, reattention
        self.gates = nn.ModuleDict({name: nn.Linear(embedding_dim, embedding_dim)
                                   for name in other_modalities} if modulation == "soft" else {})

    def forward(self, focal, others, edges):
        source, target = edges
        source_message = focal[source]
        alpha = incoming_softmax((focal[target] * source_message).sum(dim=1), target, len(focal))
        count = 1 + len(others)
        if self.modulation == "hard":
            mixed = source_message
            for values in others.values():
                mixed = mixed + values[source]
            message = torch.where((alpha > self.threshold)[:, None], mixed / count, source_message)
        else:
            # Gate can be computed per node then indexed: identical to per-edge Linear.
            message = source_message
            for name, values in others.items():
                gated = torch.sigmoid(self.gates[name](values)) * values
                message = message + gated[source]
            message = message / count
        if self.reattention:
            alpha = incoming_softmax((focal[target] * message).sum(dim=1), target, len(focal))
        return torch.zeros_like(focal).index_add_(0, target, message * alpha[:, None])


class EgoGCN(nn.Module):
    def __init__(self, data, embedding_dim=64, reg_weight=1e-4, num_routing=1,
                 modulation="hard", alpha_threshold=0.6, reattention=True, fusion_mode="mean"):
        super().__init__()
        if embedding_dim < 1 or num_routing < 0 or reg_weight < 0:
            raise ValueError("Invalid embedding dimension, routing count or regularization")
        if fusion_mode not in ("mean", "concat"):
            raise ValueError("fusion_mode must be mean or concat")
        self.num_users, self.num_items = data.num_users, data.num_items
        self.num_routing, self.reg_weight, self.fusion_mode = num_routing, reg_weight, fusion_mode
        self.modalities = list(data.features)
        self.output_dim = embedding_dim * (1 + len(self.modalities)) if fusion_mode == "concat" else embedding_dim
        pairs = torch.from_numpy(data.train_pairs.copy()).T
        pairs[1] += self.num_users
        edges = torch.cat((pairs, pairs.flip(0)), dim=1)
        self.register_buffer("edges", edges, persistent=False)
        size = self.num_users + self.num_items
        row, col = edges
        degree = torch.bincount(row, minlength=size).float().clamp_min(1)
        graph = torch.sparse_coo_tensor(edges, degree[row].rsqrt() * degree[col].rsqrt(), (size, size)).coalesce()
        self.register_buffer("id_graph", graph, persistent=False)
        self.id_embedding = nn.Parameter(torch.empty(size, embedding_dim))
        nn.init.xavier_normal_(self.id_embedding)
        self.preferences = nn.ParameterDict()
        self.projections = nn.ModuleDict()
        self.modulations = nn.ModuleDict()
        for name, feature in data.features.items():
            self.register_buffer(name + "_features", feature, persistent=False)
            self.preferences[name] = nn.Parameter(torch.empty(self.num_users, embedding_dim))
            nn.init.xavier_normal_(self.preferences[name])
            self.projections[name] = nn.Linear(feature.shape[1], embedding_dim)
            self.modulations[name] = EdgeModulation(embedding_dim,
                [other for other in self.modalities if other != name], modulation, alpha_threshold, reattention)

    def initial_content(self):
        content = {}
        for name in self.modalities:
            items = F.normalize(F.leaky_relu(self.projections[name](getattr(self, name + "_features"))), dim=1)
            # Differentiable normalization instead of mutating Parameter.data in forward.
            users = F.normalize(self.preferences[name], dim=1)
            content[name] = torch.cat((users, items), dim=0)
        return content

    def content_encode(self):
        initial = self.initial_content()
        outputs = []
        for name in self.modalities:
            other = {key: value for key, value in initial.items() if key != name}
            x = F.normalize(self.modulations[name](initial[name], other, self.edges), dim=1)
            for _ in range(self.num_routing):
                x = F.normalize(x + attention_propagate(x, self.edges), dim=1)
            outputs.append(x)
        return outputs

    def identity_encode(self):
        x = F.normalize(self.id_embedding, dim=1)
        first = torch.sparse.mm(self.id_graph, x)
        second = torch.sparse.mm(self.id_graph, first)
        # Author RealGCN feeds first-hop output to both conv2 and conv3.
        # Both operators are parameter-free: x + A*x + 2*A^2*x is equivalent.
        return x + first + 2 * second

    def encode(self):
        branches = [self.identity_encode()] + self.content_encode()
        result = torch.cat(branches, dim=1) if self.fusion_mode == "concat" else torch.stack(branches).mean(dim=0)
        return result.split((self.num_users, self.num_items))

    def forward(self):
        return torch.cat(self.encode(), dim=0)

    def compute_loss(self, users, positive, negative):
        user_emb, item_emb = self.encode()
        margin = (user_emb[users] * (item_emb[positive] - item_emb[negative])).sum(dim=1)
        loss = F.softplus(-margin).mean()
        reg = self.id_embedding[users].square().mean() + 0.5 * (
            self.id_embedding[positive + self.num_users].square().mean()
            + self.id_embedding[negative + self.num_users].square().mean())
        reg = reg + sum(preference[users].square().mean() for preference in self.preferences.values())
        return loss + self.reg_weight * reg


def main():
    parser = build_parser("EgoGCN")
    parser.add_argument("--modulation", choices=("hard", "soft"), default="hard")
    parser.add_argument("--alpha-threshold", type=float, default=0.6)
    parser.add_argument("--num-routing", type=int, default=1)
    parser.add_argument("--no-reattention", action="store_true")
    parser.add_argument("--fusion-mode", choices=("mean", "concat"), default="mean")
    args = parser.parse_args()
    if args.embedding_dim < 1 or args.num_routing < 0 or not 0 <= args.alpha_threshold <= 1:
        parser.error("embedding-dim must be positive; num-routing >= 0; alpha-threshold in [0,1]")
    seed_everything(args.seed)
    data = RecommendationData(args.data_root, args.dataset, args.core, args.modalities)
    model = EgoGCN(data, args.embedding_dim, args.reg_weight, args.num_routing, args.modulation,
                   args.alpha_threshold, not args.no_reattention, args.fusion_mode)
    args.output_embedding_dim = model.output_dim
    args.reference_repository = "https://github.com/feiyuchen7/EgoGCN"
    args.reference_commit = SOURCE_COMMIT
    args.id_propagation = "author_code_x_plus_Ax_plus_2A2x"
    args.preference_normalization = "differentiable_no_parameter_mutation"
    print(f"EgoGCN-{args.modulation}; fusion={args.fusion_mode}; final width={model.output_dim}", flush=True)
    run_experiment(model, data, args)


if __name__ == "__main__":
    main()
