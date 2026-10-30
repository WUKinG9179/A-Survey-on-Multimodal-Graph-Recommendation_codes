"""COHESION adapter; source: https://github.com/Jinfeng-Xu/COHESION.

Default embedding_dim budgets the concatenated representation, not each branch.
See utility/README.md for author-code fixes and multimodal generalization.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from DualGNN import build_user_neighbors
from MGCN import training_graphs
from utility.data import RecommendationData
from utility.experiment import build_parser, run_experiment, seed_everything


class GCNLayer(nn.Module):
    def __init__(self, users, feature_dim, width, layers):
        super().__init__()
        self.preference = nn.Parameter(torch.empty(users, width))
        nn.init.xavier_normal_(self.preference)
        self.projection = nn.Sequential(nn.Linear(feature_dim,4*width),nn.LeakyReLU(),nn.Linear(4*width,width))
        self.layers = layers

    def forward(self, features, ids, adjacency):
        projected = self.projection(features)
        items = ((ids.square()+projected.square())*.5+1e-8).sqrt()
        initial = F.normalize(torch.cat((self.preference,items)),dim=1)
        x, result = initial, initial
        for _ in range(self.layers):
            x = torch.sparse.mm(adjacency,x)
            x = F.cosine_similarity(x,initial,dim=1)[:,None]*x
            result = result+x
        return result


@torch.no_grad()
def binary_knn(features,k,block_size):
    x = F.normalize(features,dim=1)
    n,k = len(x),min(k,len(x))
    cols = torch.cat([(x[start:start+block_size]@x.T).topk(k,dim=1).indices.reshape(-1)
                      for start in range(0,n,block_size)])
    rows = torch.arange(n,device=x.device).repeat_interleave(k)
    # Equal outgoing degree k: author's symmetric degree normalization.
    return torch.sparse_coo_tensor(torch.stack((rows,cols)),
                x.new_full((len(cols),),1/(k+1e-7)),(n,n)).coalesce()


class COHESION(nn.Module):
    def __init__(self,data,embedding_dim=64,reg_weight=1e-4,n_layers=2,n_mm_layers=1,
                 ii_topk=10,uu_topk=40,image_weight=.1,edge_dropout=0.,
                 graph_block_size=256,dimension_mode='total',detach_id_branch=False,seed=2026):
        super().__init__()
        count = len(data.features)+1
        if min(embedding_dim,ii_topk,uu_topk,graph_block_size)<1 or min(n_layers,n_mm_layers)<0:
            raise ValueError('Invalid dimensions, topk, block size or layers')
        if reg_weight<0 or not 0<=image_weight<=1 or not 0<=edge_dropout<1:
            raise ValueError('Invalid regularization, image weight or dropout')
        if dimension_mode not in ('total','per-branch') or (dimension_mode=='total' and embedding_dim<count):
            raise ValueError('Total embedding dimension must allow at least one coordinate per branch')
        self.num_users,self.num_items = data.num_users,data.num_items
        self.modalities = list(data.features)
        self.widths = ([embedding_dim]*count if dimension_mode=='per-branch' else
                       [embedding_dim//count+(i<embedding_dim%count) for i in range(count)])
        self.output_dim = sum(self.widths)
        self.model_variant = 'COHESION-utility-'+dimension_mode
        self.reg_weight,self.n_mm_layers = reg_weight,n_mm_layers
        self.ii_topk,self.k,self.graph_block_size = ii_topk,uu_topk,graph_block_size
        self.image_weight,self.edge_dropout = image_weight,edge_dropout
        self.detach_id_branch = detach_id_branch
        self.id_features = nn.Parameter(torch.empty(self.num_items,max(self.widths)))
        nn.init.xavier_normal_(self.id_features)
        self.id_branch = GCNLayer(self.num_users,max(self.widths),self.widths[0],n_layers)
        self.branches = nn.ModuleDict()
        for name,width in zip(self.modalities,self.widths[1:]):
            feature = data.features[name]
            self.register_buffer(name+'_features',feature,persistent=False)
            self.branches[name] = GCNLayer(self.num_users,feature.shape[1],width,n_layers)
        if detach_id_branch:
            self.id_branch.requires_grad_(False)
        graph,_ = training_graphs(data)
        self.register_buffer('norm_adj',graph,persistent=False)
        self.register_buffer('masked_adj',None,persistent=False)
        self.register_buffer('train_pairs',torch.from_numpy(data.train_pairs.copy()).T,persistent=False)
        pu,pi = self.train_pairs
        ud = torch.bincount(pu,minlength=self.num_users).float()+1e-7
        ideg = torch.bincount(pi,minlength=self.num_items).float()+1e-7
        self.register_buffer('sampling_weights',ud[pu].rsqrt()*ideg[pi].rsqrt(),persistent=False)
        self.register_buffer('mm_adj',None,persistent=False)
        self.register_buffer('epoch_user_graph',torch.zeros(self.num_users,self.k,dtype=torch.long))
        self.register_buffer('user_weight_matrix',torch.zeros(self.num_users,self.k))
        self.register_buffer('user_graph_ready',torch.tensor(False))
        self.train_dict = data.train
        self.user_neighbors = None
        self.rng = np.random.default_rng(seed)

    def prepare_graphs(self):
        if self.mm_adj is None:
            with torch.inference_mode(False),torch.no_grad():
                graphs = [binary_knn(getattr(self,name+'_features'),self.ii_topk,self.graph_block_size) for name in self.modalities]
                if 'image' in self.modalities and len(graphs)>1:
                    weights = [self.image_weight if name=='image' else (1-self.image_weight)/(len(graphs)-1) for name in self.modalities]
                else:
                    weights = [1/len(graphs)]*len(graphs)
                merged = weights[0]*graphs[0]
                for weight,graph in zip(weights[1:],graphs[1:]):
                    merged = merged+weight*graph
                self.mm_adj = merged.coalesce()
        if not bool(self.user_graph_ready):
            self.sample_user_graph(random_padding=False)

    @torch.no_grad()
    def sample_user_graph(self,random_padding=True):
        if self.user_neighbors is None:
            self.user_neighbors = build_user_neighbors(self.train_dict,self.num_users,self.k)
        indices = np.zeros((self.num_users,self.k),dtype=np.int64)
        weights = np.zeros((self.num_users,self.k),dtype=np.float32)
        for user,(neighbors,counts) in enumerate(self.user_neighbors):
            if not neighbors:
                continue
            selected = list(range(len(neighbors)))
            if len(selected)<self.k:
                extra = (self.rng.integers(len(neighbors),size=self.k-len(selected)) if random_padding else
                         np.arange(self.k-len(selected))%len(neighbors))
                selected.extend(extra.tolist())
            indices[user] = np.asarray(neighbors)[selected]
            scores = np.asarray(counts,dtype=np.float64)[selected]
            exp = np.exp(scores-scores.max())
            weights[user] = exp/exp.sum()
        self.epoch_user_graph.copy_(torch.from_numpy(indices).to(self.epoch_user_graph.device))
        self.user_weight_matrix.copy_(torch.from_numpy(weights).to(self.user_weight_matrix.device))
        self.user_graph_ready.fill_(True)

    def pre_epoch_processing(self):
        self.prepare_graphs()
        self.sample_user_graph()
        if not self.edge_dropout:
            self.masked_adj = self.norm_adj
            return
        keep = int(self.train_pairs.shape[1]*(1-self.edge_dropout))
        selected = torch.multinomial(self.sampling_weights,keep,replacement=False) if keep else self.train_pairs.new_empty(0)
        users,items = self.train_pairs[:,selected]
        ud = torch.bincount(users,minlength=self.num_users).float()+1e-7
        idg = torch.bincount(items,minlength=self.num_items).float()+1e-7
        values = ud[users].rsqrt()*idg[items].rsqrt()
        nodes = items+self.num_users
        indices = torch.stack((torch.cat((users,nodes)),torch.cat((nodes,users))))
        self.masked_adj = torch.sparse_coo_tensor(indices,values.repeat(2),self.norm_adj.shape).coalesce()

    def encode(self):
        self.prepare_graphs()
        adjacency = self.masked_adj if self.training and self.masked_adj is not None else self.norm_adj
        ids = self.id_branch(self.id_features,self.id_features[:,:self.widths[0]],adjacency)
        if self.detach_id_branch:
            ids = ids.detach()
        views = [ids]+[branch(getattr(self,name+'_features'),self.id_features[:,:width],adjacency)
                        for (name,branch),width in zip(self.branches.items(),self.widths[1:])]
        users,items = torch.cat(views,dim=1).split((self.num_users,self.num_items))
        user_side = (users[self.epoch_user_graph]*self.user_weight_matrix[:,:,None]).sum(1)
        item_side = items
        for _ in range(self.n_mm_layers):
            item_side = torch.sparse.mm(self.mm_adj,item_side)
        return users+user_side,items+item_side

    def forward(self):
        return self.encode()

    def adaptive_margin(self,u,p,n):
        margins = torch.stack([x.sum(1) for x in (u*(p-n)).split(self.widths,dim=1)],dim=1)
        weights = 1-margins.softmax(dim=1).detach()
        return (weights*margins).sum(1)

    def compute_loss(self,users,positive,negative):
        ue,ie = self.encode()
        margin = self.adaptive_margin(ue[users],ie[positive],ie[negative])
        loss = F.softplus(-margin).mean()/math.log(2)
        reg = sum(branch.preference[users].square().mean() for branch in self.branches.values())
        return loss+self.reg_weight*reg


def main():
    parser = build_parser('COHESION')
    parser.add_argument('--n-layers',type=int,default=2)
    parser.add_argument('--n-mm-layers',type=int,default=1)
    parser.add_argument('--ii-topk',type=int,default=10)
    parser.add_argument('--uu-topk',type=int,default=40)
    parser.add_argument('--image-weight',type=float,default=.1)
    parser.add_argument('--edge-dropout',type=float,default=0.)
    parser.add_argument('--graph-block-size',type=int,default=256)
    parser.add_argument('--dimension-mode',choices=('total','per-branch'),default='total')
    parser.add_argument('--detach-id-branch',action='store_true')
    args = parser.parse_args()
    seed_everything(args.seed)
    data = RecommendationData(args.data_root,args.dataset,args.core,args.modalities)
    model = COHESION(data,args.embedding_dim,args.reg_weight,args.n_layers,args.n_mm_layers,args.ii_topk,
                     args.uu_topk,args.image_weight,args.edge_dropout,args.graph_block_size,args.dimension_mode,
                     args.detach_id_branch,args.seed)
    args.output_embedding_dim = model.output_dim
    args.branch_dimensions = dict(zip(['id']+model.modalities,model.widths))
    args.model_variant = model.model_variant
    args.evaluation_graph = 'full_training_UI_and_checkpoint_sampled_UU'
    run_experiment(model,data,args)


if __name__=='__main__':
    main()
