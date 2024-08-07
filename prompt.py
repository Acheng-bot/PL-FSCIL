import torch
import torch.nn as nn

class FSCILPrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=True, prompt_key_init='uniform',
                 num_layers=1, use_prefix_tune_for_f_prompt=False, num_heads=-1, same_key_value=False,):
        super().__init__()

        self.length = length # 5
        self.prompt_pool = prompt_pool # True
        self.embedding_key = embedding_key # cls
        self.prompt_init = prompt_init # uniform
        self.prompt_key = prompt_key # True
        self.pool_size = pool_size # 10
        self.top_k = top_k # 1
        self.batchwise_prompt = batchwise_prompt # True
        self.num_layers = num_layers # 3
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_f_prompt # True
        self.num_heads = num_heads # 12  # number of attention heads
        self.same_key_value = same_key_value # False

        if self.prompt_pool: 
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0
                if self.same_key_value: 
                    prompt_pool_shape = (self.num_layers, 1, self.pool_size, self.length, 
                                        self.num_heads, embed_dim // self.num_heads)

                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1, 1)
                else: 
                    prompt_pool_shape = (self.num_layers, 2, self.pool_size, self.length, 
                                        self.num_heads, embed_dim // self.num_heads)
                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape)) # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.prompt, -1, 1) 
            else:
                prompt_pool_shape=(self.num_layers, self.pool_size, self.length, embed_dim)
                if prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
                    
        # if using learnable prompt keys
        if prompt_key: # default=True
            key_shape = (self.pool_size, embed_dim) # 10*768
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=[0, 2])
            self.prompt_key = prompt_mean 
            
    def l2_normalize(self, x, dim=None, epsilon=1e-12): 
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed, prompt_mask=None, cls_features=None): # 24*197*768  24*1  24*768
        out = dict()
        if self.prompt_pool: 
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            prompt_key_norm = self.l2_normalize(self.prompt_key, dim=-1) 
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1) 
            

            similarity = torch.matmul(prompt_key_norm, x_embed_norm.t()) 
            similarity = similarity.t() 

            (similarity_top_k, idx) = torch.topk(similarity, k=self.top_k, dim=1) 
            out['similarity'] = similarity 

            if self.batchwise_prompt: 
                prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True) 
                # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                # Unless dimension is specified, this will be flattend if it is not already 1D.
                if prompt_id.shape[0] < self.pool_size: 
                    prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                    id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                major_prompt_id = prompt_id[major_idx] # top_k
                # expand to batch
                idx = major_prompt_id.expand(x_embed.shape[0], -1).contiguous() # B, top_k
            
            if prompt_mask is not None: 
                idx = prompt_mask 
            
            out['prompt_idx'] = idx
            if self.use_prefix_tune_for_e_prompt: 
                batched_prompt_raw = self.prompt[:,:,idx]  
                num_layers, dual, batch_size, top_k, length, num_heads, heads_embed_dim = batched_prompt_raw.shape
                batched_prompt = batched_prompt_raw.reshape( 
                    num_layers, batch_size, dual, top_k * length, num_heads, heads_embed_dim
                )
            else:
                batched_prompt_raw = self.prompt[:,idx]
                num_layers, batch_size, top_k, length, embed_dim = batched_prompt_raw.shape
                batched_prompt = batched_prompt_raw.reshape(
                    num_layers, batch_size, top_k * length, embed_dim
                )

            batched_key_norm = prompt_key_norm[idx] 

            out['prompt_key_norm'] = prompt_key_norm 
            out['selected_key'] = batched_key_norm 
            out['x_embed_norm'] = x_embed_norm 

            # Put pull_constraint loss calculation inside
            x_embed_norm = x_embed_norm.unsqueeze(1) 
            sim = batched_key_norm * x_embed_norm 
            reduce_sim = torch.sum(sim) / x_embed.shape[0] 
            # print(reduce_sim)
            # out['reduce_sim'] = 1-reduce_sim
            out['reduce_sim'] = reduce_sim
        else:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.length, 
                                        self.num_heads, embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1)
                else:
                    prompt_pool_shape = (self.num_layers, 2, self.length, 
                                        self.num_heads, embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape)) 
                        nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1, -1)
            else:
                prompt_pool_shape = (self.num_layers, self.length, embed_dim)
                if self.prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif self.prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1)
        
        out['batched_prompt'] = batched_prompt 

        return out

def ortho_loss(G, E):
    # G.shape: (num_g_prompt, 1, g_prompt_length, num_heads, embed_dim // num_heads)
    # E.shape: (num_e_prompt, 1, e_prompt_length, num_heads, embed_dim // num_heads)
    
    # Assume G and E have been reshaped or unsqueezed to ensure their dimensions align as described above
    
    loss = 0.0
    for i in range(G.shape[3]): # Loop over num_heads
        g_head = G[:, :, :, i, :]
        e_head = E[:, :, :, i, :]
        
        # Reshape tensors for matrix multiplication
        g_head = g_head.reshape(g_head.shape[0]*g_head.shape[2], g_head.shape[3])  # Shape: (num_g_prompt * g_prompt_length, embed_dim // num_heads)
        e_head = e_head.reshape(e_head.shape[0]*e_head.shape[2], e_head.shape[3])  # Shape: (num_e_prompt * e_prompt_length, embed_dim // num_heads)
        
        loss += torch.norm(torch.matmul(g_head, e_head.t()), p='fro') / (g_head.shape[0] * e_head.shape[0])
        
    return loss / G.shape[3]