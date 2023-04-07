import torch
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path

data_dir = os.path.join(Path(__file__).resolve().parents[1], 'Embeddings')

class ConTra_mlp(torch.nn.Module):
    def __init__(self, contrastive_model, hidden_size, num_classes):
        super(ConTra_mlp, self).__init__()
        self.contrastive_model = contrastive_model
        self.hidden_size = hidden_size
        
        self.num_classs = num_classes
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.fc = torch.nn.Linear(self.hidden_size, self.num_classs)
            
    
    def forward(self, x, padding_mask):
        x = self.contrastive_model(x, padding_mask=padding_mask)
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return x
    
