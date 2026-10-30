"""SMORE benchmark adapter, based on github.com/kennethorq/SMORE.

Two modalities follow the author's spectral and preference equations; automatic
three-modality input extends the spectral product and view mean. See README.
"""
import torch
from torch import nn
from torch.nn import functional as F

from MGCN import MGCN
from utility.data import RecommendationData
from utility.experiment import build_parser, run_experiment, seed_everything


@torch.no_grad()
def max_pool_graphs(graphs):
    """Maximum over existing sparse edges, without renormalizing the union."""
    graphs = [graph.coalesce() for graph in graphs]
    size = graphs[0].shape[0]
    indices = torch.cat([g.indices() for g in graphs], dim=1)
    values = torch.cat([g.values() for g in graphs])
    codes, inverse = torch.unique(indices[0] * size + indices[1], return_inverse=True)
    maximum = values.new_full((len(codes),), -torch.inf)
    maximum.scatter_reduce_(0, inverse, values, reduce='amax', include_self=True)
    return torch.sparse_coo_tensor(torch.stack((codes // size, codes % size)), maximum,
                                   (size, size)).coalesce()


class SMORE(MGCN):
    def __init__(self, data, embedding_dim=64, reg_weight=1e-4, n_layers=1,
                 n_ui_layers=3, ssl_temp=0.2, ssl_alpha=0.01, ii_topk=10,
                 graph_block_size=256, freeze_features=False, dropout=0.0):
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        if len(data.features) < 2:
            raise ValueError('SMORE requires at least two modalities')
        super().__init__(data, embedding_dim, reg_weight, n_layers, n_ui_layers,
                         ssl_temp, ssl_alpha, ii_topk, graph_block_size, freeze_features)
        del self.query_common
        self.model_variant = 'SMORE' if len(data.features) == 2 else 'SMORE-multimodal-extension'
        self.spectral_weights = nn.ParameterDict({name: nn.Parameter(torch.randn(1, embedding_dim//2+1, 2))
                                                  for name in data.features})
        self.fusion_weight = nn.Parameter(torch.randn(1, embedding_dim//2+1, 2))
        self.fusion_gate = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.Sigmoid())
        self.fusion_preference = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.Sigmoid())
        self.queries = nn.ModuleDict({name: nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.Tanh(),
                                    nn.Linear(embedding_dim, embedding_dim, bias=False)) for name in data.features})
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('fusion_graph', None, persistent=False)

    def prepare_graphs(self):
        super().prepare_graphs()
        if self.fusion_graph is None:
            with torch.inference_mode(False), torch.no_grad():
                self.fusion_graph = max_pool_graphs([getattr(self, name+'_graph') for name in self.feature_embeddings])

    def spectrum_convolution(self, features):
        spectra = {name: torch.fft.rfft(x, dim=1, norm='ortho') for name, x in features.items()}
        denoised = {name: torch.fft.irfft(x * torch.view_as_complex(self.spectral_weights[name]),
                    n=self.output_dim, dim=1, norm='ortho') for name, x in spectra.items()}
        product = torch.stack(list(spectra.values())).prod(dim=0)
        fused = torch.fft.irfft(product * torch.view_as_complex(self.fusion_weight),
                                n=self.output_dim, dim=1, norm='ortho')
        return denoised, fused

    def fuse_views(self, content, modalities, fusion):
        views = [self.dropout(self.preference_gates[name](content)) *
                 self.queries[name](fusion).softmax(dim=-1) * view
                 for name, view in zip(self.feature_embeddings, modalities)]
        views.append(self.dropout(self.fusion_preference(content)) * fusion)
        return torch.stack(views).mean(dim=0)

    def forward(self):
        self.prepare_graphs()
        features = {name: self.projections[name](embedding.weight)
                    for name, embedding in self.feature_embeddings.items()}
        denoised, fused = self.spectrum_convolution(features)
        x = torch.cat((self.user_embedding.weight, self.item_embedding.weight))
        layers = [x]
        for _ in range(self.n_ui_layers):
            x = torch.sparse.mm(self.norm_adj, x)
            layers.append(x)
        content = torch.stack(layers).mean(dim=0)
        modalities = []
        for name, features in denoised.items():
            items = self.item_embedding.weight * self.purifiers[name](features)
            modalities.append(self.propagate_items(items, getattr(self, name+'_graph')))
        fusion = self.propagate_items(self.item_embedding.weight * self.fusion_gate(fused), self.fusion_graph)
        side = self.fuse_views(content, modalities, fusion)
        users, items = (content + side).split((self.num_users, self.num_items))
        return users, items, side, content

    def propagate_items(self, items, graph):
        for _ in range(self.n_layers):
            items = torch.sparse.mm(graph, items)
        return torch.cat((torch.sparse.mm(self.interaction, items), items))

    def compute_loss(self, users, positive, negative):
        user_emb, item_emb, side, content = self.forward()
        u, p, n = user_emb[users], item_emb[positive], item_emb[negative]
        loss = F.softplus(-(u*(p-n)).sum(dim=1)).mean()
        loss = loss + self.reg_weight * 0.5 * (u.square().sum()+p.square().sum()+n.square().sum()) / len(users)
        if self.ssl_alpha:
            nodes = positive + self.num_users
            loss = loss + self.ssl_alpha * (self.info_nce(side[users], content[users]) + self.info_nce(side[nodes], content[nodes]))
        return loss


def main():
    parser = build_parser('SMORE')
    parser.add_argument('--n-layers', type=int, default=1)
    parser.add_argument('--n-ui-layers', type=int, default=3)
    parser.add_argument('--ssl-temp', type=float, default=0.2)
    parser.add_argument('--ssl-alpha', type=float, default=0.01)
    parser.add_argument('--ii-topk', type=int, default=10)
    parser.add_argument('--graph-block-size', type=int, default=256)
    parser.add_argument('--freeze-features', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.0)
    args = parser.parse_args()
    seed_everything(args.seed)
    data = RecommendationData(args.data_root, args.dataset, args.core, args.modalities)
    model = SMORE(data, args.embedding_dim, args.reg_weight, args.n_layers, args.n_ui_layers,
                  args.ssl_temp, args.ssl_alpha, args.ii_topk, args.graph_block_size, args.freeze_features, args.dropout)
    args.model_variant = model.model_variant
    args.output_embedding_dim = model.output_dim
    args.item_graph_type = 'fixed_nonnegative_cosine_knn_max_pool_union'
    args.fusion_modalities = list(data.features)
    run_experiment(model, data, args)


if __name__ == '__main__':
    main()
