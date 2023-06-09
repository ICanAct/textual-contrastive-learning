import torch.nn as nn
import torch
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
        self.ce_loss = nn.CrossEntropyLoss()
            
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        """

        representations = torch.cat([emb_i, emb_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        similarity_matrix = similarity_matrix / self.temperature
        similarity_matrix.fill_diagonal_(float("-inf"))
        
        labels = torch.arange(emb_i.shape[0]).to(similarity_matrix.device)
        labels = torch.cat([labels+labels.shape[0], labels])
        loss = self.ce_loss(similarity_matrix, labels)
        
        return loss
