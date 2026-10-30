"""MHRec: modality-aware hyperedge diffusion with shared benchmark utility.

Based on the local model and Ricardo-Ping/ChaoRec, with device-safe diffusion,
stable sparse attention and deterministic checkpoint evaluation. See README.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from DualGNN import build_user_neighbors
from MGCN import training_graphs
from utility.data import RecommendationData
from utility.experiment import build_parser, run_experiment, seed_everything


class HypergraphAttentionLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.a = nn.Parameter(torch.empty(2*width,1))
        nn.init.xavier_uniform_(self.a)

    def forward(self,h,x):
        h = h.coalesce()
        node,edge = h.indices()
        # Some PyTorch versions cannot transpose sparse tensors in inference mode.
        with torch.inference_mode(False),torch.no_grad():
            transpose = h.transpose(0,1).coalesce()
        latent = torch.sparse.mm(transpose,x)
        width = x.shape[1]
        logits = (x@self.a[:width]).squeeze(1)[node]+(latent@self.a[width:]).squeeze(1)[edge]
        maximum = logits.new_full((len(x),),-torch.inf)
        maximum.scatter_reduce_(0,node,logits.detach(),reduce='amax',include_self=True)
        numerator = (logits-maximum[node]).exp()
        denominator = logits.new_zeros(len(x)).index_add_(0,node,numerator)
        weights = numerator/denominator[node].clamp_min(1e-16)
        return torch.zeros_like(x).index_add_(0,node,weights[:,None]*latent[edge])


class Denoise(nn.Module):
    def __init__(self,nodes,hidden=1000,time_dim=10,dropout=.5):
        super().__init__()
        self.time_dim = time_dim
        self.time_projection = nn.Linear(time_dim,time_dim)
        self.encoder = nn.Linear(nodes+time_dim,hidden)
        self.decoder = nn.Linear(hidden,nodes)
        self.dropout = nn.Dropout(dropout)
        for layer in (self.time_projection,self.encoder,self.decoder):
            nn.init.normal_(layer.weight,std=math.sqrt(2/sum(layer.weight.shape)))
            nn.init.normal_(layer.bias,std=.001)

    def forward(self,x,t,mess_dropout=True):
        half = self.time_dim//2
        frequencies = torch.exp(-math.log(10000)*torch.arange(half,device=x.device)/max(half,1))
        angles = t[:,None]*frequencies[None,:]
        emb = torch.cat((angles.cos(),angles.sin()),dim=1)
        emb = F.pad(emb,(0,self.time_dim-emb.shape[1]))
        if mess_dropout:
            x = self.dropout(x)
        return self.decoder(torch.tanh(self.encoder(torch.cat((x,self.time_projection(emb)),dim=1))))


class GaussianDiffusion(nn.Module):
    def __init__(self,steps=5,noise_scale=.1,noise_min=.0001,noise_max=.02):
        super().__init__()
        if steps<2 or not 0<noise_scale*noise_min<=noise_scale*noise_max<1:
            raise ValueError('Invalid diffusion schedule')
        self.steps = steps
        variance = torch.linspace(noise_scale*noise_min,noise_scale*noise_max,steps,dtype=torch.float64)
        target = 1-variance
        betas = torch.cat((1-target[:1],1-target[1:]/target[:-1])).clamp_max(.999)
        betas[0] = .0001
        alphas = 1-betas
        abar = alphas.cumprod(0)
        previous = torch.cat((torch.ones(1,dtype=torch.float64),abar[:-1]))
        for name,value in {'abar':abar,'coef1':betas*previous.sqrt()/(1-abar),
                           'coef2':(1-previous)*alphas.sqrt()/(1-abar)}.items():
            self.register_buffer(name,value.float())

    def q_sample(self,x,t,noise=None):
        if noise is None:
            noise = torch.randn_like(x)
        alpha = self.abar[t,None]
        return alpha.sqrt()*x+(1-alpha).sqrt()*noise

    def training_loss(self,model,x):
        t = torch.randint(self.steps,(len(x),),device=x.device)
        predicted = model(self.q_sample(x,t),t)
        snr = self.abar/(1-self.abar)
        weight = torch.where(t==0,torch.ones_like(t,dtype=x.dtype),snr[(t-1).clamp_min(0)]-snr[t])
        return (weight*(predicted-x).square().mean(1)).mean()

    @torch.no_grad()
    def reconstruct(self,model,x):
        t = torch.full((len(x),),self.steps-1,device=x.device,dtype=torch.long)
        current = self.q_sample(x,t)
        for step in reversed(range(self.steps)):
            t.fill_(step)
            predicted = model(current,t,mess_dropout=False)
            current = self.coef1[step]*predicted+self.coef2[step]*current
        return current


@torch.no_grad()
def item_neighbors(features,k,block_size):
    x = F.normalize(features,dim=1)
    k = min(k,max(0,len(x)-1))
    parts = []
    for start in range(0,len(x),block_size):
        similarity = x[start:start+block_size]@x.T
        rows = torch.arange(len(similarity),device=x.device)
        similarity[rows,rows+start] = -torch.inf
        parts.append(similarity.topk(k,dim=1).indices)
    return torch.cat(parts)


def incidence_from_nodes(nodes,num_nodes):
    edges = torch.arange(len(nodes),device=nodes.device)[:,None].expand_as(nodes)
    valid = nodes>=0
    indices = torch.stack((nodes[valid],edges[valid]))
    h = torch.sparse_coo_tensor(indices,torch.ones(indices.shape[1],device=nodes.device),
                                (num_nodes,len(nodes))).coalesce()
    return torch.sparse_coo_tensor(h.indices(),torch.ones_like(h.values()),h.shape).coalesce()


class MHRec(nn.Module):
    def __init__(self,data,embedding_dim=64,reg_weight=1e-4,ii_topk=10,uu_topk=10,
                 num_hypernodes=2,n_layers=3,h_layers=2,ssl_temp=.1,ssl_alpha=.01,
                 beta1=.5,beta2=.5,diffusion_steps=5,denoise_hidden=1000,diffusion_batch_size=1024,
                 diffusion_lr=.001,graph_block_size=256,ssl_chunk_size=128):
        super().__init__()
        if min(embedding_dim,ii_topk,uu_topk,num_hypernodes,denoise_hidden,diffusion_batch_size,
               graph_block_size,ssl_chunk_size)<1 or min(n_layers,h_layers)<0:
            raise ValueError('Invalid widths, neighborhood sizes, batch sizes or layers')
        if min(reg_weight,ssl_alpha,beta1,beta2)<0 or min(ssl_temp,diffusion_lr)<=0:
            raise ValueError('Invalid objective weights, temperature or learning rate')
        self.num_users,self.num_items = data.num_users,data.num_items
        self.num_nodes = self.num_users+self.num_items
        self.output_dim,self.reg_weight = embedding_dim,reg_weight
        self.n_layers,self.h_layers = n_layers,h_layers
        self.ssl_temp,self.ssl_alpha,self.beta1,self.beta2 = ssl_temp,ssl_alpha,beta1,beta2
        self.ii_topk,self.uu_topk = ii_topk,uu_topk
        self.num_hypernodes = min(num_hypernodes,self.num_nodes)
        self.diffusion_batch_size,self.diffusion_lr = diffusion_batch_size,diffusion_lr
        self.graph_block_size,self.ssl_chunk_size = graph_block_size,ssl_chunk_size
        self.modalities = list(data.features)
        self.model_variant = 'MHRec-utility' if len(self.modalities)==2 else 'MHRec-utility-multimodal'
        self.train_dict = data.train
        self.register_buffer('train_pairs',torch.from_numpy(data.train_pairs.copy()),persistent=False)
        graph,_ = training_graphs(data)
        self.register_buffer('adj',graph,persistent=False)
        self.user_embedding = nn.Embedding(self.num_users,embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items,embedding_dim)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)
        self.feature_embeddings,self.projections,self.user_modalities = nn.ModuleDict(),nn.ModuleDict(),nn.ModuleDict()
        self.attention,self.denoisers = nn.ModuleDict(),nn.ModuleDict()
        for name,features in data.features.items():
            self.register_buffer(name+'_original',features,persistent=False)
            self.register_buffer(name+'_raw_nodes',None,persistent=False)
            self.register_buffer(name+'_generated_nodes',torch.zeros(len(data.train_pairs),self.num_hypernodes,dtype=torch.long))
            self.feature_embeddings[name] = nn.Embedding.from_pretrained(features.clone(),freeze=False)
            self.projections[name] = nn.Linear(features.shape[1],embedding_dim)
            nn.init.xavier_uniform_(self.projections[name].weight)
            self.user_modalities[name] = nn.Embedding(self.num_users,embedding_dim)
            nn.init.xavier_normal_(self.user_modalities[name].weight)
            self.attention[name] = nn.ModuleList([HypergraphAttentionLayer(embedding_dim) for _ in range(h_layers)])
            self.denoisers[name] = Denoise(self.num_nodes,denoise_hidden)
        self.modal_weight = nn.Parameter(torch.full((len(self.modalities),),.5))
        self.diffusion = GaussianDiffusion(diffusion_steps)
        self.register_buffer('generated_ready',torch.tensor(False))
        self._denoise_optimizers = None
        self._incidence_cache = {}

    def prepare_raw_hyperedges(self):
        if getattr(self,self.modalities[0]+'_raw_nodes') is not None:
            return
        neighbors = build_user_neighbors(self.train_dict,self.num_users,self.uu_topk)
        # No synthetic links to user 0; missing neighbors are padded by -1.
        users = torch.full((self.num_users,self.uu_topk),-1,dtype=torch.long,device=self.adj.device)
        for user,(near,_) in enumerate(neighbors):
            if near:
                users[user,:len(near)] = torch.tensor(near,device=users.device)
        u,i = self.train_pairs.T
        with torch.inference_mode(False),torch.no_grad():
            for name in self.modalities:
                items = item_neighbors(getattr(self,name+'_original'),self.ii_topk,self.graph_block_size)
                raw = torch.cat((u[:,None],users[u],i[:,None]+self.num_users,items[i]+self.num_users),dim=1)
                setattr(self,name+'_raw_nodes',raw)

    def edge_batches(self,name,shuffle=False):
        raw = getattr(self,name+'_raw_nodes')
        order = torch.randperm(len(raw),device=raw.device) if shuffle else torch.arange(len(raw),device=raw.device)
        for start in range(0,len(raw),self.diffusion_batch_size):
            ids = order[start:start+self.diffusion_batch_size]
            nodes = raw[ids]
            rows = torch.arange(len(ids),device=raw.device)[:,None].expand_as(nodes)
            valid = nodes>=0
            dense = torch.zeros(len(ids),self.num_nodes,device=raw.device)
            dense[rows[valid],nodes[valid]] = 1
            yield ids,dense

    def pre_epoch_processing(self):
        self.prepare_raw_hyperedges()
        if self._denoise_optimizers is None:
            self._denoise_optimizers = {name:torch.optim.Adam(model.parameters(),lr=self.diffusion_lr)
                                       for name,model in self.denoisers.items()}
        for name,denoiser in self.denoisers.items():
            denoiser.train()
            optimizer = self._denoise_optimizers[name]
            total,count = 0.,0
            for _,dense in self.edge_batches(name,shuffle=True):
                optimizer.zero_grad(set_to_none=True)
                loss = self.diffusion.training_loss(denoiser,dense)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Non-finite hyperedge diffusion loss')
                loss.backward()
                optimizer.step()
                total += loss.detach().item()*len(dense)
                count += len(dense)
            optimizer.zero_grad(set_to_none=True)
            print(f'  Diffusion {name}: loss={total/count:.6f}',flush=True)
            with torch.no_grad():
                for ids,dense in self.edge_batches(name):
                    predictions = self.diffusion.reconstruct(denoiser,dense)
                    getattr(self,name+'_generated_nodes')[ids] = predictions.topk(self.num_hypernodes,dim=1).indices
        self.generated_ready.fill_(True)
        self._incidence_cache.clear()

    def load_state_dict(self,*args,**kwargs):
        result = super().load_state_dict(*args,**kwargs)
        self._incidence_cache.clear()
        return result

    def hypergraph(self,name):
        if not bool(self.generated_ready):
            self.prepare_raw_hyperedges()
            return incidence_from_nodes(getattr(self,name+'_raw_nodes'),self.num_nodes)
        cached = self._incidence_cache.get(name)
        if cached is None or cached.device!=self.adj.device:
            with torch.inference_mode(False),torch.no_grad():
                cached = incidence_from_nodes(getattr(self,name+'_generated_nodes'),self.num_nodes)
            self._incidence_cache[name] = cached
        return cached

    def gcn(self,x):
        layers = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.adj,x)
            layers.append(x)
        return torch.stack(layers).mean(0)

    def forward(self):
        views = {}
        for name in self.modalities:
            features = self.projections[name](self.feature_embeddings[name].weight)
            initial = torch.cat((self.user_modalities[name].weight,F.normalize(features,dim=1)))
            h = self.hypergraph(name)
            x,layers = initial,[initial]
            for layer in self.attention[name]:
                if torch.is_grad_enabled():
                    update = checkpoint(layer,h,x,use_reentrant=False)
                else:
                    update = layer(h,x)
                x = F.dropout(x+update,.5,training=self.training)
                layers.append(x)
            views[name] = torch.stack(layers).mean(0)+self.beta1*self.gcn(initial)
        modal = sum(weight*views[name] for weight,name in zip(self.modal_weight.softmax(0),self.modalities))
        behavior = self.gcn(torch.cat((self.user_embedding.weight,self.item_embedding.weight)))
        result = behavior+self.beta2*F.normalize(modal,dim=1)
        users,items = result.split((self.num_users,self.num_items))
        return users,items,views,behavior

    def encode(self):
        users,items,_,_ = self.forward()
        return users,items

    def contrast_loss(self,first,second,nodes):
        first,second = F.normalize(first,dim=1),F.normalize(second,dim=1)
        total = first.new_zeros(())
        for start in range(0,len(nodes),self.ssl_chunk_size):
            selected = nodes[start:start+self.ssl_chunk_size]
            def block(q,c,indices):
                return (torch.logsumexp(q@c.T/self.ssl_temp,dim=1)-(q*c[indices]).sum(1)/self.ssl_temp).sum()
            q = first[selected]
            total += checkpoint(block,q,second,selected,use_reentrant=False) if torch.is_grad_enabled() else block(q,second,selected)
        return total/len(nodes)

    def compute_loss(self,users,positive,negative):
        ue,ie,views,behavior = self.forward()
        u,p,n = ue[users],ie[positive],ie[negative]
        margin = (u*(p-n)).sum(1)
        # Stable equivalent to the local -log(sigmoid(margin)+1e-5).
        loss = -torch.logaddexp(F.logsigmoid(margin),margin.new_tensor(math.log(1e-5))).mean()
        initial_users = torch.cat([self.user_embedding(users)]+[self.user_modalities[name](users) for name in self.modalities],dim=1)
        def initial_items(ids):
            return torch.cat([self.item_embedding(ids)]+[self.projections[name](self.feature_embeddings[name](ids)) for name in self.modalities],dim=1)
        reg = sum(value.square().mean() for value in (u,p,n,initial_users,initial_items(positive),initial_items(negative)))
        loss += self.reg_weight*reg
        if self.ssl_alpha:
            for view in views.values():
                loss += self.ssl_alpha*(self.contrast_loss(behavior[:self.num_users],view[:self.num_users],users)
                         +self.contrast_loss(behavior[self.num_users:],view[self.num_users:],positive))
        return loss


def main():
    parser = build_parser('MHRec')
    for name,default in [('ii-topk',10),('uu-topk',10),('num-hypernodes',2),('n-layers',3),('h-layers',2),
                         ('diffusion-steps',5),('denoise-hidden',1000),('graph-block-size',256),('ssl-chunk-size',128)]:
        parser.add_argument('--'+name,type=int,default=default)
    for name,default in [('ssl-temp',.1),('ssl-alpha',.01),('beta1',.5),('beta2',.5)]:
        parser.add_argument('--'+name,type=float,default=default)
    parser.add_argument('--diffusion-batch-size',type=int,default=None)
    args = parser.parse_args()
    seed_everything(args.seed)
    data = RecommendationData(args.data_root,args.dataset,args.core,args.modalities)
    args.diffusion_batch_size = args.batch_size if args.diffusion_batch_size is None else args.diffusion_batch_size
    model = MHRec(data,args.embedding_dim,args.reg_weight,args.ii_topk,args.uu_topk,args.num_hypernodes,
                  args.n_layers,args.h_layers,args.ssl_temp,args.ssl_alpha,args.beta1,args.beta2,args.diffusion_steps,
                  args.denoise_hidden,args.diffusion_batch_size,args.lr,args.graph_block_size,args.ssl_chunk_size)
    args.model_variant = model.model_variant
    args.output_embedding_dim = model.output_dim
    args.diffusion_schedule = 'local_5_step_default_SNR_weighted_x0'
    args.diffusion_optimizer = 'persistent_per_modality_Adam'
    run_experiment(model,data,args)


if __name__=='__main__':
    main()
