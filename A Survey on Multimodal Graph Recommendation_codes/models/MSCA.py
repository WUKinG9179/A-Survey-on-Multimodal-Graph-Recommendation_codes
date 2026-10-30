"""MSCA utility adapter, adapted from recomall/MSCA (GPL-3.0).

Source: https://github.com/recomall/MSCA/blob/main/src/models/msca.py
Multi-view Semantic Contrastive Alignment for Multimodal Recommendation.
See utility/README.md for graph safeguards and audio extension.
"""
from collections import Counter, defaultdict
import heapq
import torch
from torch import nn
from torch.nn import functional as F
from MGCN import MGCN
from utility.data import RecommendationData
from utility.experiment import build_parser, run_experiment, seed_everything


@torch.no_grad()
def structural_graph(train, num_items, topk):
    """Binary top-k co-occurrence (>1 shared users), plus a self-loop per item."""
    item_users = defaultdict(list)
    for user,items in train.items():
        for item in items:
            item_users[item].append(user)
    rows,cols = [],[]
    for item in range(num_items):
        counts = Counter()
        for user in item_users[item]:
            counts.update(train[user])
        counts.pop(item,None)
        candidates = (other for other,count in counts.items() if count>1)
        selected = heapq.nsmallest(topk,candidates,key=lambda other:(-counts[other],other))
        neighbors = [item]+selected
        rows.extend([item]*len(neighbors))
        cols.extend(neighbors)
    indices = torch.tensor([rows,cols],dtype=torch.long)
    degree = torch.bincount(indices[0],minlength=num_items).float()+1e-7
    values = degree[indices[0]].rsqrt()*degree[indices[1]].rsqrt()
    return torch.sparse_coo_tensor(indices,values,(num_items,num_items)).coalesce()


class MSCA(MGCN):
    def __init__(self,data,embedding_dim=64,reg_weight=1e-4,n_layers=3,ii_topk=10,
                 fusion_coeff=.3,tau=.2,cl_weight=.005,graph_block_size=256):
        if n_layers<1 or not 0<=fusion_coeff<=1:
            raise ValueError('n-layers must be positive; fusion-coeff must be in [0,1]')
        super().__init__(data,embedding_dim,reg_weight,1,n_layers,tau,cl_weight,ii_topk,graph_block_size)
        # MSCA uses feature purification but no MGCN preference gates.
        del self.preference_gates
        self.fusion_coeff = fusion_coeff
        self.train_dict = data.train
        self.model_variant = 'MSCA-utility' if len(data.features)==2 else 'MSCA-utility-multimodal'
        self.register_buffer('struct_graph',None,persistent=False)

    def prepare_graphs(self):
        super().prepare_graphs()
        if self.struct_graph is None:
            with torch.inference_mode(False),torch.no_grad():
                self.struct_graph = structural_graph(self.train_dict,self.num_items,self.ii_topk).to(self.norm_adj.device)

    def semantic_encode(self,adjacency,items):
        items = torch.sparse.mm(adjacency,items)
        return torch.cat((torch.sparse.mm(self.interaction,items),items))

    def fuse_views(self,views):
        weights = torch.cat([self.query_common(view) for view in views],dim=1).softmax(dim=1)
        redundant = sum(weights[:,index:index+1]*view for index,view in enumerate(views))
        return sum(views)-redundant

    def forward(self):
        self.prepare_graphs()
        x = torch.cat((self.user_embedding.weight,self.item_embedding.weight))
        layers = []
        # Unlike LightGCN defaults, the author excludes the unpropagated layer.
        for _ in range(self.n_ui_layers):
            x = torch.sparse.mm(self.norm_adj,x)
            layers.append(x)
        collaborative = torch.stack(layers).mean(dim=0)
        structure = self.semantic_encode(self.struct_graph,self.item_embedding.weight)
        modalities = []
        for name,embedding in self.feature_embeddings.items():
            features = self.projections[name](embedding.weight)
            items = self.item_embedding.weight*self.purifiers[name](features)
            modalities.append(self.semantic_encode(getattr(self,name+'_graph'),items))
        views = modalities+[structure]
        final = collaborative+self.fusion_coeff*self.fuse_views(views)
        users,items = final.split((self.num_users,self.num_items))
        return users,items,collaborative,views

    def encode(self):
        users,items,_,_ = self.forward()
        return users,items

    def compute_loss(self,users,positive,negative):
        ue,ie,collaborative,views = self.forward()
        margin = (ue[users]*(ie[positive]-ie[negative])).sum(dim=1)
        loss = F.softplus(-margin).mean()
        # Preserve author's unnormalized L2 over all model parameters.
        if self.reg_weight:
            loss = loss+self.reg_weight*sum(p.square().sum() for p in self.parameters())
        if self.ssl_alpha:
            nodes = positive+self.num_users
            for view in views:
                loss = loss+self.ssl_alpha*(self.info_nce(collaborative[users],view[users])
                       +self.info_nce(collaborative[nodes],view[nodes]))
        return loss


def main():
    parser = build_parser('MSCA')
    parser.add_argument('--n-layers',type=int,default=3)
    parser.add_argument('--ii-topk',type=int,default=10)
    parser.add_argument('--fusion-coeff',type=float,default=.3)
    parser.add_argument('--tau',type=float,default=.2)
    parser.add_argument('--cl-weight',type=float,default=.005)
    parser.add_argument('--graph-block-size',type=int,default=256)
    args = parser.parse_args()
    seed_everything(args.seed)
    data = RecommendationData(args.data_root,args.dataset,args.core,args.modalities)
    model = MSCA(data,args.embedding_dim,args.reg_weight,args.n_layers,args.ii_topk,args.fusion_coeff,
                 args.tau,args.cl_weight,args.graph_block_size)
    args.model_variant = model.model_variant
    args.output_embedding_dim = model.output_dim
    args.regularization = 'unnormalized_sum_all_parameter_squares'
    args.structural_graph = 'train_item_cooccurrence_gt1_binary_topk_plus_self'
    run_experiment(model,data,args)


if __name__=='__main__':
    main()
