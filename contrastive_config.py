import torch
class Config:
    def __init__(self):
        self.batch_size = 128
        self.num_workers = 1
        self.epochs = 10000
        self.hidden_size = 768
        self.num_heads = 12
        self.num_layers = 6
        self.dim_feedforward = 2048
        self.dropout = 0.1
        self.temperature = 0.1
        self.learning_rate = 0.0001
        self.custom_embeddings = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
