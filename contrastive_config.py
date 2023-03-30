import torch
class Config:
    def __init__(self):
        self.batch_size = 2
        self.num_workers = 1
        self.epochs = 20
        self.hidden_size = 768
        self.num_heads = 12
        self.num_layers = 6
        self.dropout = 0.1
        self.temperature = 0.07
        self.learning_rate = 0.0001
        self.custom_embeddings = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')