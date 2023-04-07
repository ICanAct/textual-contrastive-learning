import torch
import os
import torch.optim as optim
from pathlib import Path
from Dataset.fine_tuning_dataset import FineTuningDataset
from torch.utils.data import DataLoader, random_split
from contrastive_config import Config
from Models.distill_bert_mlp import DistillBERTMLP
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score

data_dir = os.path.join(Path(__file__).resolve().parent, "Data")
model_saves_path = os.path.join(Path(__file__).resolve().parent, 'model_saves')

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class distillbert_trainer():
    def __init__(self, model, train_dataset, test_dataset,epochs, batch_size, learning_rate, device, val_dataset=None):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.criterion = torch.nn.CrossEntropyLoss()
        self.load_tokenizer()
        self.create_data_loaders()
        self.create_optimizer()
    
    def collate_fn(self, batch):
        data, target = zip(*batch)
        output = self.tokenizer(data, truncation=True, padding='max_length', max_length=256, return_tensors="pt")
        target = torch.tensor(target)
        return output, target
    
    def load_tokenizer(self):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    def create_data_loaders(self):
        if self.val_dataset is None:
            train_set, valid_set  = random_split(self.train_dataset, [0.8, 0.2])
        else:
            train_set = self.train_dataset
            valid_set = self.val_dataset
        
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        self.valid_loader = DataLoader(valid_set, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
    
    def create_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
    
    
    def train(self):
        max_val_acc = 0
        self.model = torch.compile(self.model)
        self.model = self.model.to(self.device)
        
        for epoch in range(self.epochs):
            loss_total = 0
            loss_num = 0
            self.model.train()
            
            for step, (data, target) in enumerate(self.train_loader):
                input_ids, attention_mask, target = data['input_ids'].to(self.device), data['attention_mask'].to(self.device), target.to(self.device)
                output = self.model(input_ids, attention_mask)
                loss = self.criterion(output, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_total += loss.item()
                loss_num += 1
                if step % 100 == 0 and step!=0:
                    print(f"Epoch: {epoch}, Step: {step}, Loss: {loss_total/loss_num}")
            
            print(f"Epoch: {epoch}, Loss: {loss_total/loss_num}")
            print("Evaluating on validation set")
            val_loss,val_acc, val_f1 = self.evaluation(data_set='val')
            
            print(f"Epoch: {epoch}, Val Loss: {val_loss}, Val Acc: {val_acc}, Val F1: {val_f1}")
            
            if val_acc > max_val_acc:
                print("Validation accuracy improved, saving model")
                self.save_model('distill_bert_fine_tuned_ckpt.pt')
                max_val_acc = val_acc
        
    
    def evaluation(self, data_set='val'):
        
        if data_set == 'val':
            data_loader = self.valid_loader
        elif data_set == 'test':
            data_loader = self.test_loader
            
            
        total_logits = []
        total_labels = []
        self.model.eval()
        loss_total = 0
        loss_num = 0
        with torch.no_grad():
            for step, (data, target) in enumerate(data_loader):
                input_ids, attention_mask, token_type_ids, target = data['input_ids'].to(self.device), data['attention_mask'].to(self.device),data['token_type_ids'].to(self.device), target.to(self.device)
                logits = self.model(input_ids, attention_mask, token_type_ids)
            
                total_logits.append(logits)
                loss = self.criterion(logits, target)
                total_labels += target.tolist()
                loss_total += loss.item()
                loss_num += 1
            
            total_loss = loss_total/loss_num
            total_logits = F.softmax(torch.cat(total_logits, dim=0).detach(), dim=1)
            total_labels = torch.tensor(total_labels, device=self.device)
            accuracy = multiclass_accuracy(total_logits, total_labels)
            f1_score = multiclass_f1_score(total_logits, total_labels, num_classes=self.model.num_classes, average='macro')
            
            
        return total_loss, accuracy, f1_score
    
    def save_model(self, file_name):
        path = os.path.join(model_saves_path, file_name)
        torch.save(self.model.state_dict(), path)


if __name__ == '__main__':
    # create the dataset
    
    data_dir = os.path.join(Path(__file__).resolve().parent, "Data")
    
    train_data = FineTuningDataset('train.zstd', bert_model=True)
    val_data = FineTuningDataset('val.zstd', bert_model=True)
    test_data = FineTuningDataset('test.zstd', bert_model=True)
    config = Config()
    bert = DistillBERTMLP(num_classes=4) 
    trainer = distillbert_trainer(bert, train_data, test_data, config.epochs, config.batch_size, config.learning_rate, config.device, val_data)
    trainer.train()
    # eval on test set
    trainer.evaluation(data_set='test')