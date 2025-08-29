import torch
import torch.nn as nn
import math

class Embedding(nn.Module):
    #turns tokens into vectors
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model #d_model = dimensions of the vectors(512)
        self.vocab_size = vocab_size #vocab size = number of tokens in model vocabulary
        self.embedding = nn.Embedding(vocab_size, d_model) #initialization

    def forward(self, x):
        #tokens get turned into vector and then normalized
        return self.embedding(x) * math.sqrt(self.d_model)

class Positional_Encoding(nn.Module):

    def __init__(self, d_model: int, seq_length: int, dropout: float) -> None:
        super().__init__
        self.d_model = d_model #dimensions of vector
        self.seq_length = seq_length #amount of tokens in sequence
        self.dropout = nn.Dropout(dropout) #dropping/deactivating specific neurons 
        #positional encoding formula different for odd & even tokens

        pe = torch.zeros(seq_length, d_model) #creating empty matrix[seq length * d_model], will be filled later with values

        #one term
        position = torch.arange(0,seq_length, dtype= torch.float).unsqueeze(1)

        #two oterm       
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000) / d_model))
        
        #for even
        pe[:, 0::2] = torch.sin(position * div_term)

        #for odd
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)
    
    def forward(self, x):
        #adding positional embedding
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)
    
#layer normalization: for each of the embedding vectors -> adjust values so they have mean: 0 and variance: 1
#then shift rest of the vector
#stabilize training and make optimization easier
#Norm(x) = alpha * (x - mean)/(std + eps) + bias
class Layer_Norm(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init()
        self.eps = eps #avoid division by zero
        self.alpha = nn.Parameter(torch.ones(1))#alpha is a trainable scalable factor
        self.bias = nn.Parameter(torch.zeros(1))#bias is a trainable offset

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean)/ (std + self.eps) + self.bias

#feed_forward: update token value based on itself(per-token non-linear transformation)

class Feed_Forward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
        #turns linear -> RELU -> dropout -> linear
        #RELU is the non-linear
    
#multihead attention: looking at information from mult. perspectives
class MultiHead_Attention(nn.Module):
    #several heads(parallel attention layers) can learn diff kind of relationships at once
    #Split into 3 Parts: Q(Query), K(Key), V(value)
    #Query: "What other words are important to me"
    #Key: "This is who I am"
    #Attention Weights: Compares Q with each K to produce attention weights
    #Attention weights decide how much of Value to mix into 'cats' new representation
    #Multi-Head: various perspectives(like grammar or meaning)
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model# dimensions of embeddings
        self.h = h #number of heads
        #makes sure that each head gets even amount of dimensions
        assert d_model % h == 0, "d_model ain't divisible by h"

        self.d_k = d_model // h #dimensions per head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # how much does word A care about word B
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9) # don't pay attention to words that don't matter at all
        attention_scores = attention_scores.softmax(dim = -1) #turning scores into probabilities

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        v = self.w_v(v)

        #splitting into multiple heads 
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        #updating embedding vectors based on the attention weights, x is the updated vector
        x, self.attention_scores = MultiHead_Attention.attention(query, key, value, mask, self.dropout)

        #putting heads back all together
        x = x.transpose(1,2).continguous().view(x.shape[0], -1, self.h * self.d_k)
        
        #returning the output(x)
        return self.w_o(x)


#by passing through all these attention layers, can lose og data that's important -> preserving og data
class Residual_Connection(nn.Module):
    def __init__(self, dropout):
        self.dropout = dropout
        self.norm = Layer_Norm()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class Encoder_Block(nn.Module):
    #order: input_embedding positional embeddimg, multhead self attention, residual connection & layer norm, feed forward, res connect and layer norm again

    def __init__(self,self_attention_block: MultiHead_Attention, feed_forward_block: Feed_Forward, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([Residual_Connection(dropout) for _ in range(2)])
    

    def forward(self, x, src_mask):
        #run through multihead then residual
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x, src_mask))
        #run through feed forward then residual
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    #layers the encoder blocks
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init()
        self.layers = layers
        self.norm = Layer_Norm()
    
    #for each layer mask and then att end apply norm to it
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

#Decoder: generates the output sequence
class Decoder_Block(nn.Module):

    def __init__(self, self_attention_block: MultiHead_Attention, cross_attention_block:MultiHead_Attention, feed_forward_block:Feed_Forward, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block #looks at past target tokens
        self.cross_attention_block = cross_attention_block #looks at output from encoder
        self.feed_forward_block = feed_forward_block #adds non-linearity

        self.residual_connections = nn.Module([Residual_Connection(dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask)) #masked self_attention
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)) #uses encoder output(cross attention)
        x = self.residual_connections[2](x, self.feed_forward_block) #feed forward, processes independently
        return x

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = Layer_Norm()
    #layers each decoder block
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    