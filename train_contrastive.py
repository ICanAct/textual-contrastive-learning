import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import lr_scheduler

from Dataset.contrastive_dataset import ConstrastiveDataset
from Models.contrastive_transformer import contrastive_transformer
from contrastive_config import Config
from contrastive_loss import ContrastiveLoss


filename = "chained_augmentations"

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
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[i*50 for i in range(1,1000)], gamma=0.5)
    
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
            self.scheduler.step()
            loss_total = 0
            batches_processed = 0
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
                batches_processed += 1
            
            print(f"Epoch: {epoch}, Loss: {loss_total/batches_processed}")
            if epoch % 50 == 0:
                self.save_model(f'checkpoints/longfinal{filename}.pt')
                print(f"Learning rate: {self.scheduler.get_last_lr()}")

    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    
if __name__ == '__main__':
    # create the dataset
    filename_ext = f"{filename}.zstd"
    train_data = ConstrastiveDataset(filename_ext)
    config = Config()
    model = contrastive_transformer(config.hidden_size, config.num_heads, config.num_layers, config.dim_feedforward, config.dropout, config.custom_embeddings)
    chkpt_path = f"checkpoints/longfinal{filename}.pt"
    model.load_state_dict(torch.load(chkpt_path))
    trainer = custom_transformers_trainer(model, train_data)
    print(trainer.model)
    
        
    trainer.train()
    # path to save the weights. 
    trainer.save_model(f'checkpoints/longfinal{filename}.pt')
