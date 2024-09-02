import os 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"



###########################################
## MULTI-HEAD ATTENTION
###########################################


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention extends the basic attention mechanism by running multiple attention mechanisms in parallel, 
    allowing the model to focus on different parts of the sequence simulteniously. 
    """
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        self.embed_size = embed_size ## dimentionality of the input embeddings 
        self.num_heads = num_heads ## num of attention heads , each part will attend to different parts of the input sequence. 
        self.head_dim = embed_size // num_heads ## dimension of each attention head
        
        assert (self.head_dim * num_heads == embed_size), "Embed size must be divisible by num_heads"
        
        ## Linear layers: (values, keys, queries) these linear layers are used to project the input embeddings into different spaces, 
        ## corresponding to the values, keys and queries for each attention head.

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)        
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False) 

        ## After attention has been applied across all heads, the outputs are concatenated and passed through this linear layer to 
        ## combine them back into single output_vector of size embed_size
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_size)
        
    
    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]
        
        ## split the embeddings into self.num_heads different pieces
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)        
        queries = queries.reshape(N, query_len, self.num_heads, self.head_dim)        
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        # Check shapes before einsum
        # print("    Queries shape:", queries.shape, "Keys shape:", keys.shape)
        
        ## calculate the attention scores
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # (N, num_heads, query_len, key_len)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim = 3)
        
        ## multiply by values
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.num_heads * self.head_dim)
        
        out = self.fc_out(out)
        
        return out

###########################################
## ENCODER LAYER
###########################################
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, forward_expansion, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion), 
            nn.ReLU(), 
            nn.Linear(forward_expansion, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention = self.attention(x,x,x,mask)
        x = self.dropout(self.norm1(attention + x))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
    
class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, num_layers, num_heads, forward_expansion, dropout):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_size, num_heads, forward_expansion, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

###########################################
## DECODER LAYER
###########################################

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, forward_expansion, dropout):
        super(TransformerDecoderLayer, self).__init__()

        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion), 
            nn.ReLU(), 
            nn.Linear(forward_expansion, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
        self.encoder_attention = MultiHeadAttention(embed_size, num_heads)
        
    def forward(self, x, value, key, src_mask, tgt_mask):
        attention = self.attention(x,x,x, tgt_mask)
        x = self.dropout(self.norm1(attention + x))
        
        attention = self.attention(x,key,value, src_mask)
        x = self.dropout(self.norm2(attention + x))
        
        forward = self.feed_forward(x)
        out = self.dropout(self.norm3(forward + x))
        
        return out        
    
class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, num_layers, num_heads, forward_expansion, dropout):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_size, num_heads, forward_expansion, dropout) for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(embed_size, embed_size)
        
        
    def forward(self, x, enc_out, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, tgt_mask)
            
        return x

###########################################
## TRANSFORMER MODEL
###########################################

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_encoder_layers, num_decoder_layers, forward_expansion, dropout, max_len):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, embed_size))
        
        self.encoder = TransformerEncoder(embed_size, num_encoder_layers, num_heads, forward_expansion, dropout)
        self.decoder = TransformerDecoder(embed_size, num_decoder_layers, num_heads, forward_expansion, dropout)
        
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, target, src_mask, tgt_mask):
        print(f"x shape: {x.size()}, positional_encoding shape: {self.positional_encoding[:, :x.size(1), :].size()}")
        print(f"target shape: {target.size()}, positional_encoding shape: {self.positional_encoding[:, :target.size(1), :].size()}")
        
        # Ensure positional encoding can handle the input sequence length
        if x.size(1) > self.positional_encoding.size(1):
            raise ValueError(f"Input sequence length ({x.size(1)}) exceeds maximum positional encoding length ({self.positional_encoding.size(1)}).")
        ### embedding and positional encoding
        embed_x = self.dropout(self.embedding(x) + self.positional_encoding[:, :x.size(1), :])
        embed_target = self.dropout(self.embedding(target) + self.positional_encoding[:, :target.size(1), :])
        
        ### encoder and decoder
        enc_output = self.encoder(embed_x, src_mask)
        output = self.decoder(embed_target, enc_output, src_mask, tgt_mask)
        
        ### final output layer
        out = self.fc_out(output)
        return out