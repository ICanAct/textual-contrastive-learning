import pandas as pd
import os
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import random 
import torch

dir_path = os.path.join(Path(__file__).resolve().parents[1], 'Embeddings')
data_dir = os.path.join(Path(__file__).resolve().parents[1], 'Data')
vocab_path = os.path.join(dir_path, 'vocab_fasttext.npy')

class ConstrastiveDataset(Dataset):
    def __init__(self, file_name):
        super().__init__()
        csv_path = os.path.join(data_dir, file_name)
        self.contrastive_frame = pd.read_csv(csv_path)
        self.original_data_list = self.contrastive_frame['Description'].values.tolist()
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        self.load_embedding()
        self.load_contrastive_data()
        
    def __len__(self):
        return len(self.original_data_list)

    def __getitem__(self, index):
        
        original_data = self.original_data_list[index]
        contrastive_data = random.choice(self.augmented_data_list[index])
        original_data = self.convert_text_to_input_ids(original_data)
        contrastive_data = self.convert_text_to_input_ids(contrastive_data)
        
        return original_data, contrastive_data
    
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
    
    def load_contrastive_data(self):
        self.augmented_data_list = self.contrastive_frame[self.contrastive_frame.columns.to_list()[2:]].values.tolist()
        for i in range(len(self.augmented_data_list)):
            self.augmented_data_list[i] = [x for x in self.augmented_data_list[i] if str(x) != 'nan']


