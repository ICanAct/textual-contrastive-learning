import pandas as pd
import os
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import torch

dir_path = os.path.join(Path(__file__).resolve().parents[1], 'Embeddings')
data_dir = os.path.join(Path(__file__).resolve().parents[1], 'Data')
vocab_path = os.path.join(dir_path, 'vocab_glove.npy')

class FineTuningDataset(Dataset):
    def __init__(self, file_name, bert_model=False):
        super().__init__()
        self.file_name= file_name
        csv_path = os.path.join(data_dir, file_name)
        self.contrastive_frame = pd.read_parquet(csv_path)
        self.original_data_list = self.contrastive_frame['Description'].values.tolist()
        self.contrastive_frame['Class Index'] = self.contrastive_frame['Class Index'] - 1
        self.labels = self.contrastive_frame['Class Index'].values.tolist()
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        self.bert_model = bert_model
        self.num_classes = len(set(self.labels))
        
        if not bert_model:
            self.load_embedding()
        
        
    def __len__(self):
        return len(self.original_data_list)

    def __getitem__(self, index):
        if self.bert_model:
            original_data = self.original_data_list[index]
        else:
            original_data = self.convert_text_to_input_ids(self.original_data_list[index])
        
        labels = self.labels[index]
        return original_data, labels
    
    def load_embedding(self):
        
        vocab = np.load(vocab_path, allow_pickle=True)
        self.word2idx = {term:idx for idx,term in enumerate(vocab)}
        self.idx2word = {idx:word for word,idx in self.word2idx.items()}
    
    
    def convert_text_to_input_ids(self, text):
        words = text.strip().split()
        for i in range(len(words)):
            if words[i] not in self.word2idx:
                words[i] = self.word2idx[self.unk_token]
            else:
                words[i] = self.word2idx[words[i]]
        return torch.Tensor(words).long()
    



