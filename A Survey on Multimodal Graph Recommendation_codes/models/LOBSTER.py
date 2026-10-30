"""LOBSTER utility adapter, based on Jinfeng-Xu/LOBSTER.

Reference: Bilateral Global Semantic Enhancement for Multimedia Recommendation.
Default total dimension budget and audio extension are documented in README.
"""
import torch
from torch import nn
from torch.nn import functional as F
from utility.data import RecommendationData
from utility.experiment import build_parser, run_experiment, seed_everything


class LayerGCN(nn.Module):
    def __init__(self,feature_dim,width,n_layers=4):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(feature_dim,width),nn.Tanh())
        self.n_layers = n_layers

    def forward(self,adj,users,items,factor_u,factor_i):
        initial = torch.cat((users+factor_u.sum(0),self.mlp(items)+factor_i.sum(0)))
        x,result = initial,initial
        for _ in range(self.n_layers):
            x = torch.sparse.mm(adj,x)
            x = x*F.cosine_similarity(x,initial,dim=1)[:,None]
            result = result+x
        return result


def normalized_ui(pairs,num_users,num_items):
    users,items = pairs
    ud = torch.bincount(users,minlength=num_users).float()+1e-7
    ideg = torch.bincount(items,minlength=num_items).float()+1e-7
    values = ud[users].rsqrt()*ideg[items].rsqrt()
    nodes = items+num_users
    indices = torch.stack((torch.cat((users,nodes)),torch.cat((nodes,users))))
    size = num_users+num_items
    return torch.sparse_coo_tensor(indices,values.repeat(2),(size,size)).coalesce(),values


class LOBSTER(nn.Module):
    def __init__(self,data,embedding_dim=64,reg_weight=1e-4,n_layers=4,factor_num_u=4,
                 factor_num_i=4,dropout=0.,dimension_mode='total'):
        super().__init__()
        branches = len(data.features)+1
        if min(embedding_dim,factor_num_u,factor_num_i)<1 or n_layers<0 or reg_weight<0 or not 0<=dropout<1:
            raise ValueError('Invalid dimensions, factors, layers, regularization or dropout')
        if dimension_mode not in ('total','per-branch') or (dimension_mode=='total' and embedding_dim<branches):
            raise ValueError('Total embedding dimension must allocate at least one coordinate to every branch')
        self.num_users,self.num_items = data.num_users,data.num_items
        self.widths = ([embedding_dim]*branches if dimension_mode=='per-branch' else
                       [embedding_dim//branches+(i<embedding_dim%branches) for i in range(branches)])
        self.output_dim = sum(self.widths)
        self.model_variant = 'LOBSTER-utility-'+dimension_mode
        self.reg_weight,self.dropout = reg_weight,dropout
        width = max(self.widths)
        self.user_embds = nn.Parameter(torch.empty(self.num_users,width))
        self.item_embds = nn.Parameter(torch.empty(self.num_items,width))
        self.factor_u = nn.Parameter(torch.empty(factor_num_u,width))
        self.factor_i = nn.Parameter(torch.empty(factor_num_i,width))
        for parameter in (self.user_embds,self.item_embds,self.factor_u,self.factor_i):
            nn.init.xavier_uniform_(parameter)
        self.modalities = list(data.features)
        self.models = nn.ModuleDict({'id':LayerGCN(width,self.widths[0],n_layers)})
        for name,branch_width in zip(self.modalities,self.widths[1:]):
            self.register_buffer(name+'_features',data.features[name],persistent=False)
            self.models[name] = LayerGCN(data.features[name].shape[1],branch_width,n_layers)
        pairs = torch.from_numpy(data.train_pairs.copy()).T
        graph,values = normalized_ui(pairs,self.num_users,self.num_items)
        self.register_buffer('train_pairs',pairs,persistent=False)
        self.register_buffer('norm_adj',graph,persistent=False)
        self.register_buffer('edge_values',values,persistent=False)
        self.register_buffer('masked_adj',None,persistent=False)
        self.register_buffer('pruning_random',torch.tensor(False))

    @torch.no_grad()
    def pre_epoch_processing(self):
        if not self.dropout:
            self.masked_adj = self.norm_adj
            return
        count = self.train_pairs.shape[1]
        keep = int(count*(1-self.dropout))
        if not keep:
            selected = self.train_pairs.new_empty(0)
        elif bool(self.pruning_random):
            selected = torch.randperm(count,device=self.train_pairs.device)[:keep]
        else:
            selected = torch.multinomial(self.edge_values,keep,replacement=False)
        self.pruning_random.logical_not_()
        self.masked_adj,_ = normalized_ui(self.train_pairs[:,selected],self.num_users,self.num_items)

    def encode(self):
        adjacency = self.masked_adj if self.training and self.masked_adj is not None else self.norm_adj
        views = []
        for (name,model),width in zip(self.models.items(),self.widths):
            features = self.item_embds if name=='id' else getattr(self,name+'_features')
            views.append(model(adjacency,self.user_embds[:,:width],features,
                               self.factor_u[:,:width],self.factor_i[:,:width]))
        return torch.cat(views,dim=1).split((self.num_users,self.num_items))

    def forward(self):
        return self.encode()

    def adaptive_margin(self,users,positive,negative):
        margins = torch.stack([part.sum(1) for part in (users*(positive-negative)).split(self.widths,dim=1)],dim=1)
        weights = 1-margins.softmax(dim=1).detach()
        return (weights*margins).sum(1)

    def compute_loss(self,users,positive,negative):
        ue,ie = self.encode()
        margin = self.adaptive_margin(ue[users],ie[positive],ie[negative])
        bpr = F.softplus(-margin).sum()
        reg = .5*sum(value.square().sum() for value in
                      (self.user_embds[users],self.item_embds[positive],self.item_embds[negative]))
        return bpr+self.reg_weight*reg


def main():
    parser = build_parser('LOBSTER')
    parser.add_argument('--n-layers',type=int,default=4)
    parser.add_argument('--factor-num-u',type=int,default=4)
    parser.add_argument('--factor-num-i',type=int,default=4)
    parser.add_argument('--dropout',type=float,default=0.)
    parser.add_argument('--dimension-mode',choices=('total','per-branch'),default='total')
    args = parser.parse_args()
    seed_everything(args.seed)
    data = RecommendationData(args.data_root,args.dataset,args.core,args.modalities)
    model = LOBSTER(data,args.embedding_dim,args.reg_weight,args.n_layers,args.factor_num_u,args.factor_num_i,
                    args.dropout,args.dimension_mode)
    args.model_variant = model.model_variant
    args.output_embedding_dim = model.output_dim
    args.branch_dimensions = dict(zip(['id']+model.modalities,model.widths))
    args.loss_reduction = 'sum_BPR_and_half_sum_sampled_ID_L2'
    run_experiment(model,data,args)


if __name__=='__main__':
    main()
