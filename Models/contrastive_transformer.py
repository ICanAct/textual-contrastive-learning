import torch
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np
import os
from pathlib import Path
import math

data_dir = os.path.join(Path(__file__).resolve().parents[1], 'Embeddings')

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 256):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0)]
        x = x.transpose(0, 1)
        return self.dropout(x)

class contrastive_transformer(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, num_layers, dim_feedforward=1024, dropout=0.5, custom_embeddings=True):
        super(contrastive_transformer, self).__init__()
        self.custom_embeddings = custom_embeddings
        self.hidden_size = hidden_size
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers
        self.heads = num_heads
        self.dropout = dropout
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.custom_embeddings:
            self.load_embeddings()
            self.embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(self.embeddings).float(), freeze=True)
            self.input_size = self.embeddings.shape[1]
        else:
            self.embedding = torch.nn.Embedding(1000001, self.hidden_size, padding_idx=0)
            self.input_size = self.hidden_size
        
        transformer_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.input_size, dim_feedforward=self.dim_feedforward, nhead=self.heads, batch_first=True, dropout=self.dropout, activation='gelu')
        self.transformer_encoder = torch.nn.TransformerEncoder(transformer_encoder_layer, num_layers=self.num_layers)
        self.pos_enc = PositionalEncoding(self.input_size)
   
    def load_embeddings(self):
        # load the embeddings here
        #self.embeddings = np.load(os.path.join(data_dir, "embs_fasttext.npy"), allow_pickle=True)
        self.embeddings = np.load(os.path.join(data_dir, "embs_glove.npy"), allow_pickle=True)
    
    def forward(self, x, padding_mask):
        x = self.embedding(x)
        x = self.pos_enc(x)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        return x
    
