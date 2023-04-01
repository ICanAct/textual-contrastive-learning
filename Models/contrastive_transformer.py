import torch
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path

data_dir = os.path.join(Path(__file__).resolve().parents[1], 'Embeddings')

class contrastive_transformer(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, num_layers, dropout=0.5, custom_embeddings=False):
        super(contrastive_transformer, self).__init__()
        self.custom_embeddings = custom_embeddings
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.heads = num_heads
        self.dropout = dropout
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.custom_embeddings:
            self.load_embeddings()
            self.embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(self.embeddings).float(), freeze=True)
            self.input_size = self.embeddings.shape[1]
            
        self.embedding = torch.nn.Embedding(1000001, 300, padding_idx=0)
        self.input_size = 300
        self.transformer_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.input_size, nhead=self.heads, batch_first=True, dropout=self.dropout, activation='gelu')
        self.transformer_encoder = torch.nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=self.num_layers)
    
    def load_embeddings(self):
        # load the embeddings here
        self.embeddings = np.load(os.path.join(data_dir, "embs_fasttext.npy"), allow_pickle=True)
    
    def forward(self, x, padding_mask):
        if self.custom_embeddings:
            x = self.embedding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        return x
    
