import os
import numpy as np
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

from utility.data import RecommendationData
from utility.experiment import build_parser, run_experiment, seed_everything
from MGCN import training_graphs


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, args):
        super().__init__()
        self.args = args
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

    def forward(self, adj):
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.args.UI_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings


class MMHCLBase(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, args):
        super().__init__()
        self.args = args
        self.n_users = n_users
        self.n_items = n_items
        self.embeddings_dim = embedding_dim

        self.user_ui_embedding = nn.Embedding(n_users, self.embeddings_dim)
        self.item_ui_embedding = nn.Embedding(n_items, self.embeddings_dim)

        self.uu_embedding = nn.Embedding(n_users, self.embeddings_dim)
        self.ii_embedding = nn.Embedding(n_items, self.embeddings_dim)

        if self.args.cf_model == 'NGCF':
            self.GC_Linear_list = nn.ModuleList()
            self.Bi_Linear_list = nn.ModuleList()
            self.dropout_list = nn.ModuleList()
            for i in range(self.args.UI_layers):
                self.GC_Linear_list.append(nn.Linear(self.embeddings_dim, self.embeddings_dim))
                self.Bi_Linear_list.append(nn.Linear(self.embeddings_dim, self.embeddings_dim))
                self.dropout_list.append(nn.Dropout(0.1))

        nn.init.xavier_uniform_(self.user_ui_embedding.weight)
        nn.init.xavier_uniform_(self.item_ui_embedding.weight)
        nn.init.xavier_uniform_(self.uu_embedding.weight)
        nn.init.xavier_uniform_(self.ii_embedding.weight)

        self.tau = self.args.temperature

    def _propagate_hyper_embeddings(self, I2I_mat, U2U_mat, return_layers=False):
        ii_emb = self.ii_embedding.weight
        uu_emb = self.uu_embedding.weight
        ii_layers = [ii_emb] if return_layers else None
        uu_layers = [uu_emb] if return_layers else None

        if self.args.item_loss_ratio != 0:
            for _ in range(self.args.Item_layers):
                ii_emb = self.propagate_items(ii_emb) if I2I_mat is None else torch.sparse.mm(I2I_mat, ii_emb)
                if return_layers:
                    ii_layers.append(ii_emb)

        if self.args.user_loss_ratio != 0:
            for _ in range(self.args.User_layers):
                uu_emb = self.propagate_users(uu_emb) if U2U_mat is None else torch.sparse.mm(U2U_mat, uu_emb)
                if return_layers:
                    uu_layers.append(uu_emb)

        if return_layers:
            return ii_emb, uu_emb, ii_layers, uu_layers
        return ii_emb, uu_emb

    def _propagate_ui_embeddings(self, UI_mat):
        UI_mat = self.norm_adj if UI_mat is None else UI_mat
        if self.args.cf_model == 'LightGCN':
            ego_embeddings = torch.cat((self.user_ui_embedding.weight, self.item_ui_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for _ in range(self.args.UI_layers):
                side_embeddings = torch.sparse.mm(UI_mat, ego_embeddings)
                ego_embeddings = side_embeddings
                all_embeddings += [ego_embeddings]
            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_ui_emb, i_ui_emb = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        elif self.args.cf_model == 'NGCF':
            ego_embeddings = torch.cat((self.user_ui_embedding.weight, self.item_ui_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.args.UI_layers):
                side_embeddings = torch.sparse.mm(UI_mat, ego_embeddings)
                sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
                bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
                bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
                ego_embeddings = sum_embeddings + bi_embeddings
                ego_embeddings = self.dropout_list[i](ego_embeddings)

                norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
                all_embeddings += [norm_embeddings]

            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_ui_emb, i_ui_emb = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        elif self.args.cf_model == 'MF':
            u_ui_emb, i_ui_emb = self.user_ui_embedding.weight, self.item_ui_embedding.weight
        else:
            raise ValueError(f'Unsupported cf_model: {self.args.cf_model}')

        return u_ui_emb, i_ui_emb

    def get_embedding_components(self, UI_mat=None, I2I_mat=None, U2U_mat=None, return_layers=False):
        if return_layers:
            ii_emb, uu_emb, ii_layers, uu_layers = self._propagate_hyper_embeddings(
                I2I_mat,
                U2U_mat,
                return_layers=True,
            )
        else:
            ii_emb, uu_emb = self._propagate_hyper_embeddings(I2I_mat, U2U_mat)
            ii_layers, uu_layers = None, None
        u_ui_emb, i_ui_emb = self._propagate_ui_embeddings(UI_mat)

        if self.args.item_loss_ratio != 0:
            item_hyper_emb = F.normalize(ii_emb, p=2, dim=1)
        else:
            item_hyper_emb = torch.zeros_like(i_ui_emb)

        if self.args.user_loss_ratio != 0:
            user_hyper_emb = F.normalize(uu_emb, p=2, dim=1)
        else:
            user_hyper_emb = torch.zeros_like(u_ui_emb)

        u_final_emb = u_ui_emb + user_hyper_emb
        i_final_emb = i_ui_emb + item_hyper_emb

        result = {
            'u_ui_emb': u_ui_emb,
            'i_ui_emb': i_ui_emb,
            'u_hyper_emb': user_hyper_emb,
            'i_hyper_emb': item_hyper_emb,
            'u_final_emb': u_final_emb,
            'i_final_emb': i_final_emb,
            'uu_raw_emb': uu_emb,
            'ii_raw_emb': ii_emb,
        }
        if return_layers:
            result['uu_layers'] = uu_layers
            result['ii_layers'] = ii_layers
        return result

    def apply_counterfactual(self, components, mode='both'):
        mode = mode.lower()
        if mode == 'user':
            u_cf = components['u_ui_emb']
            i_cf = components['i_final_emb']
        elif mode == 'item':
            u_cf = components['u_final_emb']
            i_cf = components['i_ui_emb']
        elif mode in {'both', 'full', 'all'}:
            u_cf = components['u_ui_emb']
            i_cf = components['i_ui_emb']
        elif mode in {'none', 'off'}:
            u_cf = components['u_final_emb']
            i_cf = components['i_final_emb']
        else:
            raise ValueError(f'Unsupported counterfactual mode: {mode}')
        return u_cf, i_cf

    def counterfactual_forward(self, UI_mat=None, I2I_mat=None, U2U_mat=None, mode='both', return_components=False, return_layers=False):
        components = self.get_embedding_components(UI_mat=None, I2I_mat=None, U2U_mat=None, return_layers=return_layers)
        u_cf, i_cf = self.apply_counterfactual(components, mode=mode)
        if return_components:
            return u_cf, i_cf, components
        return u_cf, i_cf

    def forward(self, UI_mat=None, I2I_mat=None, U2U_mat=None):
        components = self.get_embedding_components(UI_mat, I2I_mat, U2U_mat)
        return (
            components['u_final_emb'],
            components['i_final_emb'],
            components['ii_raw_emb'],
            components['uu_raw_emb'],
        )

    def encode(self):
        users, items, _, _ = self.forward()
        return users, items

    def sim(self, first, second):
        return F.normalize(first, dim=1) @ F.normalize(second, dim=1).T

    def batched_contrastive_loss(self, z1, z2, batch_size=None):
        from torch.utils.checkpoint import checkpoint
        chunk = self.ssl_chunk_size if batch_size is None else batch_size
        if chunk < 1 or z1.shape != z2.shape or not len(z1):
            raise ValueError('Invalid contrastive inputs or chunk size')
        first, second = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
        total = first.new_zeros(())
        for start in range(0, len(first), chunk):
            ids = torch.arange(start, min(start+chunk, len(first)), device=first.device)
            def block(a, b, index):
                queries = a[index]
                within = queries @ a.T / self.tau
                cross = queries @ b.T / self.tau
                mask = index[:, None] == torch.arange(len(a), device=a.device)[None, :]
                within = within.masked_fill(mask, -torch.inf)
                return (torch.logsumexp(torch.cat((within, cross), dim=1), dim=1)
                        - (queries*b[index]).sum(1)/self.tau).sum()
            if torch.is_grad_enabled() and (first.requires_grad or second.requires_grad):
                total = total + checkpoint(block, first, second, ids, use_reentrant=False)
            else:
                total = total + block(first, second, ids)
        return total/len(first)

    def compute_loss(self, users, positive, negative):
        ue, ie, ii, uu = self.forward()
        u, p, n = ue[users], ie[positive], ie[negative]
        loss = F.softplus(-(u*(p-n)).sum(1)).mean()
        loss = loss + self.reg_weight*.5*(u.square().sum()+p.square().sum()+n.square().sum())/len(users)
        if self.args.item_loss_ratio:
            loss = loss + self.args.item_loss_ratio*self.batched_contrastive_loss(ie, ii)
        if self.args.user_loss_ratio:
            loss = loss + self.args.user_loss_ratio*self.batched_contrastive_loss(ue, uu)
        return loss


@torch.no_grad()
def modality_incidence(features, topk, block_size):
    """Positive cosine top-k binary supports concatenated along hyperedges."""
    size = len(next(iter(features.values())))
    rows, columns = [], []
    for modality, value in enumerate(features.values()):
        x = F.normalize(value, dim=1)
        for start in range(0, size, block_size):
            scores, indices = (x[start:start+block_size] @ x.T).topk(min(topk, size), dim=1)
            row = torch.arange(start, start+len(scores), device=x.device)[:, None].expand_as(indices)
            valid = scores > 0
            rows.append(row[valid])
            columns.append(indices[valid]+modality*size)
    indices = torch.stack((torch.cat(rows), torch.cat(columns)))
    return torch.sparse_coo_tensor(indices, torch.ones(indices.shape[1], device=indices.device),
                                   (size, size*len(features))).coalesce()


class MMHCL(MMHCLBase):
    """Author graph/loss semantics with factorized sparse propagation.

    Reference: https://github.com/Xu107/MMHCL (ACM TOMM 2025).
    """
    def __init__(self, data, embedding_dim=64, reg_weight=1e-4, ui_layers=2,
                 user_layers=3, item_layers=2, topk=5, temperature=0.6,
                 user_loss_ratio=0.03, item_loss_ratio=0.07, cf_model='LightGCN',
                 graph_block_size=256, ssl_chunk_size=128):
        from types import SimpleNamespace
        if min(embedding_dim, topk, graph_block_size, ssl_chunk_size) < 1 or min(ui_layers, user_layers, item_layers) < 0:
            raise ValueError('Invalid widths, topk, chunk sizes or layer counts')
        if temperature <= 0 or min(reg_weight, user_loss_ratio, item_loss_ratio) < 0:
            raise ValueError('Invalid temperature or loss weights')
        if cf_model not in ('LightGCN', 'NGCF', 'MF'):
            raise ValueError('Unsupported cf_model')
        options = SimpleNamespace(UI_layers=ui_layers, User_layers=user_layers, Item_layers=item_layers,
                                  item_loss_ratio=item_loss_ratio, user_loss_ratio=user_loss_ratio,
                                  temperature=temperature, cf_model=cf_model)
        super().__init__(data.num_users, data.num_items, embedding_dim, options)
        self.output_dim, self.reg_weight = embedding_dim, reg_weight
        self.topk, self.graph_block_size, self.ssl_chunk_size = topk, graph_block_size, ssl_chunk_size
        graph, _ = training_graphs(data)
        self.register_buffer('norm_adj', graph, persistent=False)
        pairs = torch.from_numpy(data.train_pairs.copy()).T
        interaction = torch.sparse_coo_tensor(pairs, torch.ones(pairs.shape[1]), (self.n_users, self.n_items)).coalesce()
        self.register_buffer('interaction', interaction, persistent=False)
        self.register_buffer('interaction_t', interaction.T.coalesce(), persistent=False)
        degree = torch.bincount(pairs[0], minlength=self.n_users).float()[:, None]
        item_degree = torch.bincount(pairs[1], minlength=self.n_items).float()[:, None]
        neighbor_degree = torch.sparse.mm(interaction, item_degree)-degree
        self.register_buffer('user_degree', degree, persistent=False)
        self.register_buffer('neighbor_degree', neighbor_degree.clamp_min(1), persistent=False)
        self.modalities = list(data.features)
        for name, feature in data.features.items():
            self.register_buffer(name+'_features', feature, persistent=False)
        for name in ('incidence', 'incidence_t', 'item_inverse_degree'):
            self.register_buffer(name, None, persistent=False)

    def prepare_graphs(self):
        if self.incidence is None and self.args.item_loss_ratio:
            with torch.inference_mode(False), torch.no_grad():
                h = modality_incidence({name: getattr(self, name+'_features') for name in self.modalities},
                                       self.topk, self.graph_block_size)
                ht = h.T.coalesce()
                degree = torch.sparse.mm(h, torch.sparse.mm(ht, torch.ones(self.n_items, 1, device=h.device)))
                self.incidence, self.incidence_t = h, ht
                self.item_inverse_degree = torch.where(degree > 0, degree.clamp_min(1).rsqrt(), torch.zeros_like(degree))

    def propagate_items(self, x):
        self.prepare_graphs()
        inv = self.item_inverse_degree
        return inv*torch.sparse.mm(self.incidence, torch.sparse.mm(self.incidence_t, inv*x))

    def propagate_users(self, x):
        return (torch.sparse.mm(self.interaction, torch.sparse.mm(self.interaction_t, x))
                - self.user_degree*x)/self.neighbor_degree


# Dataset-specific loss weights from the recorded experiments in
# reports/comparison_sources.json. Shared benchmark settings remain in build_parser.
DATASET_LOSS_DEFAULTS = {
    'Tiktok': {'user_loss_ratio': 0.03, 'item_loss_ratio': 0.07},
    'Sports': {'user_loss_ratio': 0.3, 'item_loss_ratio': 0.7},
    'Clothing': {'user_loss_ratio': 0.3, 'item_loss_ratio': 0.7},
}


def resolve_dataset_loss_defaults(args):
    defaults = DATASET_LOSS_DEFAULTS[args.dataset]
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args


def main():
    parser = build_parser('MMHCL')
    parser.add_argument('--ui-layers', '--UI_layers', type=int, default=2)
    parser.add_argument('--user-layers', '--User_layers', type=int, default=3)
    parser.add_argument('--item-layers', '--Item_layers', type=int, default=2)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--user-loss-ratio', '--user_loss_ratio', type=float, default=None,
                        help='Default: Tiktok=0.03, Sports/Clothing=0.3')
    parser.add_argument('--item-loss-ratio', '--item_loss_ratio', type=float, default=None,
                        help='Default: Tiktok=0.07, Sports/Clothing=0.7')
    parser.add_argument('--cf-model', '--cf_model', choices=('LightGCN', 'NGCF', 'MF'), default='LightGCN')
    parser.add_argument('--graph-block-size', type=int, default=256)
    parser.add_argument('--ssl-chunk-size', type=int, default=128)
    args = resolve_dataset_loss_defaults(parser.parse_args())
    seed_everything(args.seed)
    data = RecommendationData(args.data_root, args.dataset, args.core, args.modalities)
    model = MMHCL(data, args.embedding_dim, args.reg_weight, args.ui_layers, args.user_layers, args.item_layers,
                  args.topk, args.temperature, args.user_loss_ratio, args.item_loss_ratio, args.cf_model,
                  args.graph_block_size, args.ssl_chunk_size)
    args.output_embedding_dim = model.output_dim
    args.hypergraph_operator = 'factorized_symmetric_H_Ht_positive_binary_knn'
    args.ssl_scope = 'all_users_and_all_items_per_training_batch'
    run_experiment(model, data, args)


if __name__ == '__main__':
    main()
