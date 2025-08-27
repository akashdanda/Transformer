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
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = dropout
