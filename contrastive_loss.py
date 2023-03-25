import torch.nn as nn
import torch
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
            
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        similarity_matrix = similarity_matrix / self.temperature
        similarity_matrix.fill_diagonal_(float("-inf"))
        
        labels = torch.arange(z_i.shape[0]).to(similarity_matrix.device)
        labels = torch.cat([labels+labels.shape[0], labels])
        loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
        
        return loss