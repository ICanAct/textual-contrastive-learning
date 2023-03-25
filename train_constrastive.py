from Dataset.contrastive_dataset import ConstrastiveDataset
from Models.contrastive_transformer import contrastive_transformer
from constrastive_config import Config
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
from contrastive_loss import ContrastiveLoss


class custom_transformers_trainer():
    def __init__(self, model, train_dataset):
        self.model = model
        self.config = Config()
        self.train_dataset = train_dataset
        self.create_optimizer()
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=self.collate_fn)
        self.criterion = ContrastiveLoss(self.config.batch_size,self.config.temperature)
        
    def create_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
    
    def collate_fn(self, batch):
        original_data, contrastive_data = zip(*batch)
        original_data = torch.nn.utils.rnn.pad_sequence(original_data, batch_first=True, padding_value=0)
        contrastive_data = torch.nn.utils.rnn.pad_sequence(contrastive_data, batch_first=True, padding_value=0)
        return original_data, contrastive_data
        
    def train(self):
        
        loss_total = 0
        loss_num = 0
        # this is 2.0 specific. Remove this if you are not using torch 2.0
        model = torch.compile(self.model)
        model = model.to(self.config.device)
        model.train()
        for epoch in range(self.config.epochs):
            for step, (original, augmented) in enumerate(self.train_loader):
                # creating mask
                self.optimizer.zero_grad()
                
                original, augmented = original.to(self.config.device), augmented.to(self.config.device)
                original_mask = torch.zeros((original.shape[1], original.shape[1]), device=self.config.device).type(torch.bool)
                original_padding_mask = (original == 0)
                original_padding_mask = original_padding_mask.to(self.config.device)
                
                augmented_mask = torch.zeros((augmented.shape[1], augmented.shape[1]), device=self.config.device).type(torch.bool)
                augmented_padding_mask = (augmented == 0)
                augmented_padding_mask = augmented_padding_mask.to(self.config.device)
                
                original_embs = self.model(original, original_mask, original_padding_mask)
                augmented_embs = self.model(augmented, augmented_mask, augmented_padding_mask)
                
                # mean pooling for sentence representation.
                original_embs = torch.mean(original_embs, dim=1)
                augmented_embs = torch.mean(augmented_embs, dim=1)
                
                loss = self.criterion(original_embs, augmented_embs)
                
                loss.backward()
                self.optimizer.step()
                loss_total += loss.item()
                loss_num += 1
                if step % 100 == 0 and step!=0:
                    print(f"Epoch: {epoch}, Step: {step}, Loss: {loss_total/loss_num}")
            
            print(f"Epoch: {epoch}, Loss: {loss_total/loss_num}")

    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    
if __name__ == '__main__':
    # create the dataset
    train_data = ConstrastiveDataset('sample_with_none.csv')
    config = Config()
    model = contrastive_transformer(config.hidden_size, config.num_heads, config.num_layers, config.dropout, config.custom_embeddings)
    trainer = custom_transformers_trainer(model, train_data)
    
        
    trainer.train()
    # path to save the weights. 
    trainer.save_model('transformer_weights.pt')