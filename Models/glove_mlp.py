import torch
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path

data_dir = os.path.join(Path(__file__).resolve().parents[1], 'Embeddings')

class glove_mlp(torch.nn.Module):
    def __init__(self,hidden_size, num_classes):
        super(glove_mlp, self).__init__()
        
        self.load_embeddings()
        self.hidden_size = hidden_size
        self.num_classs = num_classes
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(self.embeddings).float(), freeze=True)
        self.fc = torch.nn.Linear(self.hidden_size, self.num_classs)
    
    def load_embeddings(self):
        # load the embeddings here
        #self.embeddings = np.load(os.path.join(data_dir, "embs_fasttext.npy"), allow_pickle=True)
        self.embeddings = np.load(os.path.join(data_dir, "embs_glove.npy"), allow_pickle=True)
    
    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return x
    
