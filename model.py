import torch
from torch import nn

class SelfAttention(nn.Module):
    def __init__(self,embed_size,heads):
        super(SelfAttention,self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size), 'Embed size needs to be div by heads'
        
        self.values = nn.Linear(self.head_dim,self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim,self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim,self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)
        
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        values_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        #split embedding into self.heads pieces
        values = values.reshape(N,values_len,self.heads,self.head_dim)
        keys = keys.reshpe(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N,key_len, self.heads,self.head_dim)
        
        energy = torch.einsum()