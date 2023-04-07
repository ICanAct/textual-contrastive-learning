import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
from pathlib import Path
from contrastive_config import Config
from Dataset.fine_tuning_dataset import FineTuningDataset
from Models.glove_mlp import glove_mlp
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score

model_saves_path = os.path.join(Path(__file__).resolve().parent, 'model_saves')

class fine_tuning_trainer():
    def __init__(self, model, train_dataset, val_dataset, test_dataset):
        self.model = model
        self.config = Config()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.create_optimizer()
        self.create_dataloader()
        
        
        self.criterion = torch.nn.CrossEntropyLoss()
      
    def create_dataloader(self):
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=self.collate_fn)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=self.collate_fn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=self.collate_fn)
        
    def create_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
    
    def collate_fn(self, batch):
        original_data, labels = zip(*batch)
        original_data = pad_sequence(original_data, batch_first=True, padding_value=0)
        labels = torch.tensor(labels)
        return original_data, labels
    

    def train(self):
        max_val_acc = 0
        # this is 2.0 specific. Remove this if you are not using torch 2.0
        #self.model = torch.compile(self.model)
        self.model = self.model.to(self.config.device)
        
        for epoch in range(self.config.epochs):
            
            self.model.train()
            loss_total = 0
            total_steps = 0
            for step, (original, labels) in enumerate(self.train_loader):
                # creating mask
            
                original, labels = original.to(self.config.device), labels.to(self.config.device)
                
                outputs = self.model(original)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_total += loss.item()
                total_steps += 1
                if step % 100 == 0 and step!=0:
                    print(f"Epoch: {epoch}, Step: {step}, Loss: {loss_total/total_steps}")
                
            print(f"Epoch: {epoch}, Loss: {loss_total/total_steps}")
            val_loss, val_acc, val_f1 = self.evaluation(data_set='val')
            
            print("Validation Loss: {}, Validation Accuracy: {}, Validation F1: {}".format(val_loss, val_acc, val_f1))
            if val_acc > max_val_acc:
                print("Improved Validation Accuracy. Saving model...")
                max_val_acc = val_acc
                self.save_model("fine_tune_model_mlp.pt")

    
    def evaluation(self, data_set='val'):
        
        if data_set == 'val':
            data_loader = self.val_loader
        elif data_set == 'test':
            data_loader = self.test_loader
            
        total_logits = []
        total_labels = []
        self.model.eval()
        loss_total = 0
        loss_num = 0
        
        for data, targets in data_loader:
            with torch.no_grad():
                data, targets = data.to(self.config.device), targets.to(self.config.device)
                logits = self.model(data)
                total_logits.append(logits)
                loss = self.criterion(logits, targets)
                total_labels += targets.tolist()
                loss_total += loss.item()
                loss_num += 1
        
        total_loss = loss_total/loss_num
        total_logits = F.softmax(torch.cat(total_logits, dim=0).detach(), dim=1)
        total_labels = torch.tensor(total_labels, device=self.config.device)
        accuracy = multiclass_accuracy(total_logits, total_labels, num_classes=4)
        f1_score = multiclass_f1_score(total_logits, total_labels, average='macro', num_classes=4)
        
        return total_loss, accuracy, f1_score
    
    def save_model(self, file_name):
        path = os.path.join(model_saves_path, file_name)
        torch.save(self.model.state_dict(), path)

    
if __name__ == '__main__':
    # create the dataset
    
    data_dir = os.path.join(Path(__file__).resolve().parent, "Data")
    
    train_data = FineTuningDataset('train.zstd')
    val_data = FineTuningDataset('val.zstd')
    test_data = FineTuningDataset('test.zstd')
    config = Config()
    mlp = glove_mlp(300, 4)
    trainer = fine_tuning_trainer(mlp, train_data, val_data, test_data)
    trainer.train()
    # eval on test set
    print("Evaluating on test set...")
    test_loss, test_acc, test_f1 = trainer.evaluation(data_set='test')
    print("Test Loss: {}, Test Accuracy: {}, Test F1: {}".format(test_loss, test_acc, test_f1))
    print("Smolboi!")
