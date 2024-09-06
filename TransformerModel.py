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

    Attention score is calculated using the dot product between query and keys. 
    This gives us the score that indicates how much focus a query should place on each key..

    Resulting energy tensor element indicates the importance of the particular key for the particulat query.

    To prevent the model from attending certain positios (eg. padding tokens), mask tensor is applied here setting those positions to a 
    very negative value (-1e20) so they contribute nearly zero to the attention dimesion. 

    Attention scores are scaled by square root of embed_size (stabilization technique) and passes through the softmax function to get the
    probability distribution. This tells model to how much attention to pay to each key for a given query. 

    Attention weights are used to calculate the weighted sum of values, which gives us the final output for the each head. This allows model 
    to  focus on specific parts of the input based on the attention score. 

    The outputs of all attention heads are concatenated and passed through the final linear layer to produce the final output of
    the multi-head attention layer. 
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
        ## e.g. if num heads is 8 and embed_size is 512 then each head will have dimentionality of 512 // 8 = 64
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)        
        queries = queries.reshape(N, query_len, self.num_heads, self.head_dim)        
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        # Check shapes before einsum
        # print("k Queries shape:", queries.shape, "Keys shape:", keys.shape)
        
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
    """
    TransformerEncoder is a single layer in transformer encoder. Each encoder layer processes the input sequence. Applying self-attention and FFNN then normalizes and adds
    the output to the original input. 
    
    embed_size : size of input embeddings
    num_heads : num of attention heads used in MultiHeadAttention layer
    forward_expansion : factor by which the hidden layer is expanded in FFNN. if embed_size is 512 and forward_expansion is 4 then there will be 512 * 4 = 2048 units in the FFNN.


    
    """
    def __init__(self, embed_size, num_heads, forward_expansion, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(embed_size, num_heads) ## allows model to focus on the different parts of input sequence when encoding it. 
        self.norm1 = nn.LayerNorm(embed_size) ## stabilizes and accelerates the training of the model by normalizing the inputs accross the features. This helps in stablity
        self.norm2 = nn.LayerNorm(embed_size) ## also helps in scaling stability 


        ## gets applied to each position in the sequence independently, consist of 2 linear layers with ReLU activation in between. 
        ## it expands the dimentionality and then projects back to embed_size.
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion), 
            nn.ReLU(), 
            nn.Linear(forward_expansion, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention = self.attention(x,x,x,mask) ## self attention applied to the input x since the self attention the queries, keys and values all come from the same input x
        x = self.dropout(self.norm1(attention + x)) ## output of the attention is added back to the original input information while allowing the model to learn modifications 
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x)) 
        return out
    
class TransformerEncoder(nn.Module):

    """
    TransformerEncoder class is composed with multiple TransformerEncoderLayer layers stacked on top of each other. Each layer processes the input sequence, 
    allowing the model to build complex representations. 

    num_layers: num of TransformerEncoderLayer layers to stack, each layer adds more depth and complexity to the model, allowing it to learn more complex representations. 

    The model applies each layer sequentially, passing the output of one layer as a input to the next. 

    output: x is returned where output is rich, context aware representation of the input sequence where the each position has been encoded with information from the other 
    relevent positions in the sequence. 
    """
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
    """
    A single layer of TransformerDecoderLayer extends the functionality of TransformerEncoderLayer by adding 2nd attention mechanism that allows the decoder to 
    attend to the encoder's output. 

    """
    def __init__(self, embed_size, num_heads, forward_expansion, dropout):
        super(TransformerDecoderLayer, self).__init__()

        ## This is 1st multi head attention layer which applies self attention to decoder's input (that is, previously generated tokens). It helps the model understand the relationship 
        ## between different parts of the output sequence. 


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
        ## This is 2nd MultiHeadAttention layer which allows decoder to attend the output of encoder. it enables the model to focus on relevant parts of input sequence when generating the output. 
        self.encoder_attention = MultiHeadAttention(embed_size, num_heads)
        
    def forward(self, x, value, key, src_mask, tgt_mask):
        attention = self.attention(x,x,x, tgt_mask)

        ## output of self-Attention layer is added back to the original input and then normalized using LayerNorm. This is ia residual connection that helps preserve the original input information
        ## while allowing model to learn the modifications. 
        x = self.dropout(self.norm1(attention + x))
        ## Tis attends to the encoders output, which helps decoder to use information from the input sequence when generating each token of the output sequence. 
        attention = self.attention(x,key,value, src_mask)
        x = self.dropout(self.norm2(attention + x))
        
        forward = self.feed_forward(x)
        out = self.dropout(self.norm3(forward + x))
        
        return out        
    
class TransformerDecoder(nn.Module):
    """
    TransformerDecoder is componsed of multiple TransformerDecoderLayer stacked on top of each other. Each layer process input sequence from the previous layer, allowing the model to build complex representations and generate more accurate outputs. 


    """
    def __init__(self, embed_size, num_layers, num_heads, forward_expansion, dropout):
        super(TransformerDecoder, self).__init__()
        ## Model applies each layer sequentially, passing output of one layer as the input to the next
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_size, num_heads, forward_expansion, dropout) for _ in range(num_layers)
        ])
        ## After passing through all decoder layers final output is passed through a fully connected layer. This maps the embed size to the size of vocab allowing the model to predict next token in the sequence. 
        self.fc_out = nn.Linear(embed_size, embed_size)
        
        
    def forward(self, x, enc_out, src_mask, tgt_mask):
        """
        Here x is embedded and partially decoded sequence is passed through each TransformerDecoderLayer. 
        Output represents the generated sequence enriched with context from the entire input sequence and any previously generated tokens.
        """
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