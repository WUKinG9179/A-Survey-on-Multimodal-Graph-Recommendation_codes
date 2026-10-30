"""MGCE adapter for the shared recommendation benchmark.

Reference: Li et al., TKDE 2024, DOI 10.1109/TKDE.2024.3424268.
Algorithm/defaults are transcribed from the authors' Beauty implementation:
https://github.com/shuaiyangli/MGCE/blob/main/Model-beauty/model_beauty.py
https://github.com/shuaiyangli/MGCE/blob/main/Model-beauty/load_data.py
The paper's full text was not available for independent parameter verification.
See reports/MGCE_implementation.md for source differences and benchmark adaptations.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from utility.data import RecommendationData
from utility.experiment import build_parser, run_experiment, seed_everything


class _InverseSquareRoot(torch.autograd.Function):
    """Exact spectral inverse square root with a repeated-eigenvalue-safe gradient.

    For f(a)=a**(-1/2), the divided difference is
    -1 / (sqrt(a)*sqrt(b)*(sqrt(a)+sqrt(b))), including a == b.
    This avoids the eigenvector-gradient singularity of naive eigh/SVD autograd.
    """
    @staticmethod
    def forward(ctx, covariance, epsilon):
        ctx.input_dtype = covariance.dtype
        eigenvalues, vectors = torch.linalg.eigh(covariance.double())
        roots = (eigenvalues.clamp_min(0) + epsilon).sqrt()
        ctx.save_for_backward(roots, vectors)
        result = (vectors / roots.unsqueeze(0)) @ vectors.T
        return result.to(covariance.dtype)

    @staticmethod
    def backward(ctx, gradient):
        roots, vectors = ctx.saved_tensors
        gradient = (gradient.double() + gradient.double().T) * 0.5
        spectral = vectors.T @ gradient @ vectors
        a, b = roots[:, None], roots[None, :]
        result = vectors @ (-spectral / (a * b * (a + b))) @ vectors.T
        return result.to(ctx.input_dtype), None


def zca_whiten(features, epsilon=1e-5):
    """Uncentered ZCA on projected item features, as in the author code."""
    covariance = features.T @ features / features.shape[0]
    return features @ _InverseSquareRoot.apply(covariance, epsilon)


def causal_adjacency(data, sender_exponent):
    """D^-0.5 (A+I) D^sender_exponent; training edges only."""
    pairs = torch.as_tensor(data.train_pairs, dtype=torch.long)
    users, items = pairs[:, 0], pairs[:, 1] + data.num_users
    size = data.num_users + data.num_items
    diagonal = torch.arange(size)
    row = torch.cat((users, items, diagonal))
    col = torch.cat((items, users, diagonal))
    degree = torch.bincount(row, minlength=size).float()
    values = degree[row].pow(-0.5) * degree[col].pow(sender_exponent)
    return torch.sparse_coo_tensor(torch.stack((row, col)), values, (size, size)).coalesce()


def popularity_buckets(data, count=3):
    """Training interaction counts; clip the maximum into the last valid bin."""
    popularity = torch.bincount(torch.as_tensor(data.train_pairs[:, 1]),
                               minlength=data.num_items).float()
    span = popularity.max() - popularity.min()
    if span == 0:
        return torch.zeros(data.num_items, dtype=torch.long)
    return ((popularity - popularity.min()) / span * count).floor().long().clamp(0, count - 1)


class ModalityBranch(nn.Module):
    def __init__(self, num_users, feature_dim, embedding_dim):
        super().__init__()
        self.projection = nn.Linear(feature_dim, embedding_dim, bias=False)
        self.interest = nn.Parameter(torch.empty(num_users, embedding_dim))
        self.conformity = nn.Parameter(torch.empty(num_users, embedding_dim))
        for value in (self.projection.weight, self.interest, self.conformity):
            nn.init.xavier_uniform_(value)


class MGCE(nn.Module):
    def __init__(self, data, embedding_dim=64, reg_weight=1e-4,
                 base_layers=4, interest_layers=1, conformity_layers=2,
                 lambda_m=0.5, mju_mf=0.3, mju_emb=0.1, eit=0.3,
                 base_exponent=-0.4, interest_exponent=-0.3,
                 conformity_exponent=-0.5, pop_buckets=3, zca_epsilon=1e-5,
                 output_fusion="projection"):
        super().__init__()
        if output_fusion not in ("projection", "concat"):
            raise ValueError("output-fusion must be projection or concat")
        if min(embedding_dim, base_layers, interest_layers, conformity_layers, pop_buckets) < 1:
            raise ValueError("Dimensions, layer counts and pop-buckets must be positive")
        scalars = (reg_weight, lambda_m, mju_mf, mju_emb, eit)
        if any(not math.isfinite(v) or v < 0 for v in scalars):
            raise ValueError("Loss and modality weights must be finite and nonnegative")
        if not math.isfinite(zca_epsilon) or zca_epsilon <= 0:
            raise ValueError("zca-epsilon must be finite and positive")
        if any(not math.isfinite(v) or not -1 <= v <= 0
               for v in (base_exponent, interest_exponent, conformity_exponent)):
            raise ValueError("Degree exponents must lie in [-1, 0]")
        self.num_users, self.num_items = data.num_users, data.num_items
        self.reg_weight, self.lambda_m = reg_weight, lambda_m
        self.mju_mf, self.mju_emb, self.eit = mju_mf, mju_emb, eit
        self.base_layers, self.interest_layers = base_layers, interest_layers
        self.conformity_layers, self.zca_epsilon = conformity_layers, zca_epsilon
        for name, exponent in (("base", base_exponent), ("interest", interest_exponent),
                               ("conformity", conformity_exponent)):
            self.register_buffer(name + "_graph", causal_adjacency(data, exponent), persistent=False)
        self.register_buffer("popularity_ids", popularity_buckets(data, pop_buckets), persistent=False)
        self.user_embedding = nn.Parameter(torch.empty(data.num_users, embedding_dim))
        self.item_embedding = nn.Parameter(torch.empty(data.num_items, embedding_dim))
        nn.init.xavier_uniform_(self.user_embedding)
        nn.init.xavier_uniform_(self.item_embedding)
        self.popularity_embedding = nn.Embedding(pop_buckets, embedding_dim)
        nn.init.uniform_(self.popularity_embedding.weight, -0.05, 0.05)
        self.branches = nn.ModuleDict()
        for name, features in data.features.items():
            self.register_buffer(name + "_features", features, persistent=False)
            self.branches[name] = ModalityBranch(data.num_users, features.shape[1], embedding_dim)
        self.concatenated_dim = (1 + 2 * len(self.branches)) * embedding_dim
        self.output_fusion = output_fusion
        self.output_dim = embedding_dim if output_fusion == "projection" else self.concatenated_dim
        self.output_projection = (nn.Linear(self.concatenated_dim, embedding_dim, bias=False)
                                  if output_fusion == "projection" else nn.Identity())
        if output_fusion == "projection":
            nn.init.xavier_uniform_(self.output_projection.weight)
        self.model_variant = ("MGCE-OfficialCode-PyTorch" if set(self.branches) == {"image", "text"}
                              else "MGCE-OfficialCode-PyTorch-AudioExtension")
        if output_fusion == "projection":
            self.model_variant = f"MGCE-Projected{embedding_dim}" + ("-AudioExtension" if "audio" in self.branches else "")

    @staticmethod
    def propagate(graph, features, layers):
        # Author code uses the last layer only, not LightGCN's layer average.
        for _ in range(layers):
            features = torch.sparse.mm(graph, features)
        return features

    def representations(self):
        base = self.propagate(self.base_graph,
                              torch.cat((self.user_embedding, self.item_embedding)), self.base_layers)
        popularity = self.popularity_embedding(self.popularity_ids)
        branches = {}
        for name, branch in self.branches.items():
            item = zca_whiten(branch.projection(getattr(self, name + "_features")), self.zca_epsilon)
            interest = self.propagate(self.interest_graph, torch.cat((branch.interest, item)),
                                      self.interest_layers)
            conformity = self.propagate(self.conformity_graph,
                                        torch.cat((branch.conformity, item + popularity)),
                                        self.conformity_layers)
            branches[name] = (interest, conformity, item)
        return base, branches

    def scoring_representation(self, base, branches):
        pieces = [base]
        for name, (interest, conformity, _) in branches.items():
            # Concatenation with sqrt(weight) exactly reproduces the author's
            # sum of branch dot products without introducing cross terms.
            scale = math.sqrt(self.lambda_m) if name == "image" else 1.0
            pieces.extend((scale * interest, scale * conformity))
        # New adaptation: projected scores are not equivalent to historical scores.
        return self.output_projection(torch.cat(pieces, dim=1))

    def encode(self):
        return self.scoring_representation(*self.representations()).split((self.num_users, self.num_items))

    def bpr(self, representation, users, positive, negative):
        u = representation[users]
        p, n = representation[positive + self.num_users], representation[negative + self.num_users]
        return F.softplus(-(u * (p - n)).sum(dim=1)).mean()

    @staticmethod
    def embedding_penalty(user, positive, negative):
        # tf.nn.l2_loss is half the sum of squares, divided by batch size.
        return 0.5 * (user.square().sum() + positive.square().sum()
                      + negative.square().sum()) / len(user)

    @staticmethod
    def mean_cosine(x, y):
        x = F.normalize(x.mean(dim=0), dim=0, eps=1e-6)
        y = F.normalize(y.mean(dim=0), dim=0, eps=1e-6)
        return (x * y).sum() / (x.norm() * y.norm() + 1e-8)

    def loss_components(self, users, positive, negative):
        base, branches = self.representations()
        base_loss = self.bpr(base, users, positive, negative)
        auxiliary = base_loss.new_zeros(())
        discrepancy = base_loss.new_zeros(())
        regularizer = self.embedding_penalty(self.user_embedding[users],
                                             self.item_embedding[positive], self.item_embedding[negative])
        for name, (interest, conformity, item) in branches.items():
            branch = self.branches[name]
            auxiliary = auxiliary + self.bpr(interest, users, positive, negative)
            auxiliary = auxiliary + self.bpr(conformity, users, positive, negative)
            discrepancy = discrepancy + self.mean_cosine(interest[users], conformity[users])
            discrepancy = discrepancy + self.mean_cosine(interest[positive + self.num_users],
                                                          conformity[positive + self.num_users])
            # The released code regularizes whitened content in both branches,
            # not the popularity-augmented conformity input. Preserve that choice.
            regularizer = regularizer + self.mju_emb * (
                self.embedding_penalty(branch.interest[users], item[positive], item[negative])
                + self.embedding_penalty(branch.conformity[users], item[positive], item[negative]))
        # Preserve the RELEASED negative cosine sign. This rewards similarity;
        # it must not be described as verified cosine-distance disentanglement.
        discrepancy = -discrepancy / len(users)
        components = {"base": base_loss, "auxiliary": self.mju_mf * auxiliary,
                      "discrepancy": self.eit * discrepancy, "regularization": self.reg_weight * regularizer}
        if self.output_fusion == "projection":
            # Train the same final representation used for recommendation scores.
            components["projected_recommendation"] = self.bpr(
                self.scoring_representation(base, branches), users, positive, negative)
            components["projection_regularization"] = (
                0.5 * self.reg_weight * self.output_projection.weight.square().sum())
        return components

    def compute_loss(self, users, positive, negative):
        return sum(self.loss_components(users, positive, negative).values())


def make_parser():
    parser = build_parser("MGCE")
    parser.add_argument("--output-fusion", choices=("projection", "concat"), default="projection",
                        help="projection: final embedding-dim output, requires retraining; concat: historical scoring")
    parser.add_argument("--base-layers", type=int, default=4)
    parser.add_argument("--interest-layers", type=int, default=1)
    parser.add_argument("--conformity-layers", type=int, default=2)
    parser.add_argument("--lambda-m", type=float, default=0.5, help="visual scoring weight; text/audio weight = 1")
    parser.add_argument("--mju-mf", type=float, default=0.3, help="auxiliary branch BPR coefficient")
    parser.add_argument("--mju-emb", type=float, default=0.1, help="auxiliary embedding regularizer coefficient")
    parser.add_argument("--eit", type=float, default=0.3, help="released mean-cosine term coefficient")
    parser.add_argument("--base-exponent", type=float, default=-0.4)
    parser.add_argument("--interest-exponent", type=float, default=-0.3)
    parser.add_argument("--conformity-exponent", type=float, default=-0.5)
    parser.add_argument("--pop-buckets", type=int, default=3)
    parser.add_argument("--zca-epsilon", type=float, default=1e-5)
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    data = RecommendationData(args.data_root, args.dataset, args.core, args.modalities)
    try:
        model = MGCE(data, **{key: getattr(args, key) for key in (
            "embedding_dim", "reg_weight", "base_layers", "interest_layers", "conformity_layers",
            "lambda_m", "mju_mf", "mju_emb", "eit", "base_exponent", "interest_exponent",
            "conformity_exponent", "pop_buckets", "zca_epsilon", "output_fusion")})
    except ValueError as error:
        parser.error(str(error))
    args.output_embedding_dim = model.output_dim
    args.concatenated_embedding_dim = model.concatenated_dim
    args.mgce_parameter_source = "author Beauty code; paper full-text parameters unverified"
    args.mgce_popularity_source = "training interaction counts only"
    args.mgce_discrepancy = "negative cosine of batch mean embeddings, matching released code"
    print(f"MGCE: {model.model_variant}; scoring embedding width: {model.output_dim}", flush=True)
    print("MGCE defaults: author Beauty code; common settings: shared benchmark. "
          "Paper-specific parameter verification pending.", flush=True)
    run_experiment(model, data, args)


if __name__ == "__main__":
    main()
