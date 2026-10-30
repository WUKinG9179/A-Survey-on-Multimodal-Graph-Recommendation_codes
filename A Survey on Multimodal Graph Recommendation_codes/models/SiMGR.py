"""SiMGR-inspired implicit-feedback adaptation, authorized by the user.

No explicit dislikes are present locally. Signed pseudo edges are semantic
estimates, not observed negative feedback. This is NOT a full paper reproduction.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from MGCN import MGCN
from utility.data import RecommendationData
from utility.experiment import build_parser,run_experiment,seed_everything


def rating_signs(data,path,threshold_std=0.):
    """Optional train-only JSON {user: {item: rating}}; exact train coverage."""
    ratings=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    converted={}
    for user,values in ratings.items():
        if not isinstance(values,dict):
            raise ValueError('Ratings must map each user to {item: rating}')
        for item,value in values.items():
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<=0:
                raise ValueError('Ratings must be positive finite numbers')
            key=(int(user),int(item))
            if key in converted:
                raise ValueError('Duplicate rating IDs')
            converted[key]=float(value)
    expected={tuple(pair) for pair in data.train_pairs.tolist()}
    if set(converted)!=expected:
        raise ValueError('Ratings must cover exactly train pairs; validation/test ratings are not accepted')
    thresholds={}
    for user,items in data.train.items():
        values=np.asarray([converted[(user,item)] for item in items])
        if len(values):
            thresholds[user]=values.mean()+threshold_std*values.std()
    return np.asarray([converted[(u,i)]>=thresholds[u] for u,i in data.train_pairs],dtype=bool)


def signed_adjacency(pairs,num_users,num_items,device):
    pairs=torch.as_tensor(pairs,dtype=torch.long,device=device).reshape(-1,2)
    users,items=pairs.T
    nodes=items+num_users
    indices=torch.stack((torch.cat((users,nodes)),torch.cat((nodes,users))))
    size=num_users+num_items
    raw=torch.sparse_coo_tensor(indices,torch.ones(indices.shape[1],device=device),(size,size)).coalesce()
    rows,cols=raw.indices()
    degree=torch.bincount(rows,minlength=size).float().clamp_min(1)
    return torch.sparse_coo_tensor(raw.indices(),degree[rows].rsqrt()*degree[cols].rsqrt(),raw.shape).coalesce()


@torch.no_grad()
def semantic_pseudo_edges(data,per_user,candidate_count,seed):
    """Static train-history cosine heuristic; balanced, disjoint unobserved pairs."""
    positive,negative=[],[]
    if per_user==0:
        return np.empty((0,2),dtype=np.int64),np.empty((0,2),dtype=np.int64)
    rng=np.random.default_rng(seed)
    allowed=np.unique(data.train_pairs[:,1]) if data.core=='0' else np.arange(data.num_items)
    features=[F.normalize(feature,dim=1) for feature in data.features.values()]
    # CPU preprocessing avoids keeping user-by-candidate-by-feature tensors on GPU.
    for user,history in sorted(data.train.items()):
        seen=set(history)
        available=len(allowed)-len(seen)
        take=min(candidate_count,available)
        k=min(per_user,take//2)
        if not k:
            continue
        selected=set()
        for _ in range(10):
            selected.update(int(item) for item in rng.choice(allowed,size=max(2*take,16)) if item not in seen)
            if len(selected)>=take:
                break
        if len(selected)<take:
            remaining=np.asarray([item for item in allowed if item not in seen and item not in selected])
            selected.update(rng.choice(remaining,size=take-len(selected),replace=False).tolist())
        candidates=np.asarray(sorted(selected))
        if len(candidates)>take:
            candidates=rng.choice(candidates,size=take,replace=False)
        candidates.sort()
        score=torch.zeros(len(candidates))
        for feature in features:
            profile=F.normalize(feature[list(history)].mean(0),dim=0)
            score+=feature[candidates]@profile
        # Stable tie ordering and separate ends ensure no edge has both signs.
        order=np.lexsort((candidates,score.numpy()))
        if score[order[-k]].item() <= score[order[k-1]].item():
            continue  # No semantic separation: do not invent opposite signs.
        negative.extend((user,int(item)) for item in candidates[order[:k]])
        positive.extend((user,int(item)) for item in candidates[order[-k:]])
    return np.asarray(positive,dtype=np.int64).reshape(-1,2),np.asarray(negative,dtype=np.int64).reshape(-1,2)


class SiMGR(MGCN):
    def __init__(self,data,embedding_dim=64,reg_weight=1e-4,n_layers=2,ii_topk=10,
                 pseudo_per_user=1,pseudo_candidates=64,sign_weight=.1,ssl_alpha=.01,
                 ssl_temp=.2,graph_block_size=256,seed=2026,ratings_file=None,threshold_std=0.):
        if pseudo_per_user<0 or pseudo_candidates<2*max(pseudo_per_user,1) or sign_weight<0 or not np.isfinite(threshold_std):
            raise ValueError('Invalid pseudo-edge count, candidates, sign weight or threshold')
        super().__init__(data,embedding_dim,reg_weight,1,n_layers,ssl_temp,ssl_alpha,ii_topk,graph_block_size)
        self.model_variant='SiMGR-inspired-Implicit' if ratings_file is None else 'SiMGR-inspired-RatingGraph'
        self.sign_weight=sign_weight
        self.negative_mlp=nn.Sequential(nn.Linear(embedding_dim,embedding_dim),nn.ReLU(),
                                       nn.Linear(embedding_dim,embedding_dim))
        self.sign_queries=nn.Parameter(torch.empty(2,embedding_dim))
        nn.init.xavier_uniform_(self.sign_queries)
        self._sign_data=data
        self._pseudo_options=(pseudo_per_user,pseudo_candidates,seed)
        self._observed_positive=(np.ones(len(data.train_pairs),dtype=bool) if ratings_file is None else
                                 rating_signs(data,ratings_file,threshold_std))
        self.register_buffer('positive_graph',None,persistent=False)
        self.register_buffer('negative_graph',None,persistent=False)
        self.sign_stats=None

    def prepare_signs(self):
        if self.positive_graph is not None:
            return
        with torch.inference_mode(False),torch.no_grad():
            data=self._sign_data
            pp,pn=semantic_pseudo_edges(data,*self._pseudo_options)
            observed=data.train_pairs
            positive=np.concatenate((observed[self._observed_positive],pp))
            negative=np.concatenate((observed[~self._observed_positive],pn))
            self.positive_graph=signed_adjacency(positive,self.num_users,self.num_items,self.norm_adj.device)
            self.negative_graph=signed_adjacency(negative,self.num_users,self.num_items,self.norm_adj.device)
            self.sign_stats={'observed_positive':int(self._observed_positive.sum()),
                             'observed_negative':int((~self._observed_positive).sum()),
                             'pseudo_positive':len(pp),'pseudo_negative':len(pn)}
            print('SiMGR signed graph: '+json.dumps(self.sign_stats),flush=True)

    def signed_enhancement(self,content):
        self.prepare_signs()
        x=torch.cat((self.user_embedding.weight,self.item_embedding.weight))
        layers=[x]
        for _ in range(self.n_ui_layers):
            x=torch.sparse.mm(self.positive_graph,x)
            layers.append(x)
        positive=torch.stack(layers).mean(0)
        message=torch.sparse.mm(self.negative_graph,positive)
        negative=self.negative_mlp(message)
        # Empty negative neighborhoods must not fabricate a negative message via bias.
        active=torch.zeros(len(content),dtype=torch.bool,device=content.device)
        active[self.negative_graph.indices()[0]]=True
        negative=negative*active[:,None]
        # Stable learned softmax replaces the paper's potentially singular ratio.
        logits=torch.stack(((positive*self.sign_queries[0]).sum(1),
                            (negative*self.sign_queries[1]).sum(1)),dim=1)
        weight=logits.softmax(1)
        return content+self.sign_weight*F.relu(weight[:,:1]*positive+weight[:,1:]*negative)

    def forward(self):
        _,_,side,content=super().forward()
        enhanced=self.signed_enhancement(content)
        users,items=(enhanced+side).split((self.num_users,self.num_items))
        return users,items,side,enhanced


def main():
    parser=build_parser('SiMGR (authorized implicit-feedback adaptation)')
    parser.add_argument('--n-layers',type=int,default=2)
    parser.add_argument('--ii-topk',type=int,default=10)
    parser.add_argument('--pseudo-per-user',type=int,default=1)
    parser.add_argument('--pseudo-candidates',type=int,default=64)
    parser.add_argument('--sign-weight',type=float,default=.1)
    parser.add_argument('--ssl-alpha',type=float,default=.01)
    parser.add_argument('--ssl-temp',type=float,default=.2)
    parser.add_argument('--graph-block-size',type=int,default=256)
    parser.add_argument('--ratings-file',type=Path)
    parser.add_argument('--threshold-std',type=float,default=0.)
    args=parser.parse_args()
    seed_everything(args.seed)
    data=RecommendationData(args.data_root,args.dataset,args.core,args.modalities)
    model=SiMGR(data,args.embedding_dim,args.reg_weight,args.n_layers,args.ii_topk,args.pseudo_per_user,
                args.pseudo_candidates,args.sign_weight,args.ssl_alpha,args.ssl_temp,args.graph_block_size,
                args.seed,args.ratings_file,args.threshold_std)
    args.model_variant=model.model_variant
    args.output_embedding_dim=model.output_dim
    args.sign_source='train_history_semantic_pseudo_edges; NOT observed dislikes'
    args.paper_reproduction=False
    print(model.model_variant+': authorized adaptation, NOT full paper SiMGR.',flush=True)
    run_experiment(model,data,args)


if __name__=='__main__':
    main()
