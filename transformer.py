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