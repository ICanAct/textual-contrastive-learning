import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from Dataset.contrastive_dataset import ConstrastiveDataset
from Models.contrastive_transformer import contrastive_transformer
from contrastive_config import Config
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
        original_data = pad_sequence(original_data, batch_first=True, padding_value=0)
        contrastive_data = pad_sequence(contrastive_data, batch_first=True, padding_value=0)
        return original_data, contrastive_data
        
    def train(self):
        
        # this is 2.0 specific. Remove this if you are not using torch 2.0
        model = torch.compile(self.model)
        model = model.to(self.config.device)
        model.train()
        for epoch in range(self.config.epochs):
            loss_total = 0
            total_steps = 0
            for step, (original, augmented) in enumerate(self.train_loader):
                # creating mask
                self.optimizer.zero_grad()
                
                original, augmented = original.to(self.config.device), augmented.to(self.config.device)

                original_padding_mask = (original == 0).to(self.config.device)
                augmented_padding_mask = (augmented == 0).to(self.config.device)
                
                original_embs = self.model(original, padding_mask=original_padding_mask)
                augmented_embs = self.model(augmented, padding_mask=augmented_padding_mask)
                
                # mean pooling for sentence representation.
                original_embs = torch.mean(original_embs, dim=1)
                augmented_embs = torch.mean(augmented_embs, dim=1)
                
                loss = self.criterion(original_embs, augmented_embs)
                
                loss.backward()
                self.optimizer.step()
                loss_total += loss.item()
                total_steps += 1
                if step % 100 == 0 and step!=0:
                    print(f"Epoch: {epoch}, Step: {step}, Loss: {loss_total/total_steps}")
            
            # save the model after 25 epochs
            if epoch % 25 == 0 and epoch!=0:
                self.save_model(f"transformer_weights_{self.train_dataset.filename}.pt")
                
            print(f"Epoch: {epoch}, Loss: {loss_total/total_steps}")

    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    
if __name__ == '__main__':
    # create the dataset
    filename = "con_sub_unique.zstd"
    train_data = ConstrastiveDataset(filename)
    config = Config()
    model = contrastive_transformer(config.hidden_size, config.num_heads, config.num_layers, config.dropout, config.custom_embeddings)
    trainer = custom_transformers_trainer(model, train_data)
    
        
    trainer.train()
    # path to save the weights. 
    trainer.save_model(f'transformer_weights_final_{filename}.pt')
