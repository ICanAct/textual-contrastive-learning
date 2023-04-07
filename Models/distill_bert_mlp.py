import torch
from transformers import DistilBertModel

class DistillBERTMLP(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.linear = torch.nn.Linear(768, self.num_classes)
    
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    return_dict=True
                )
        output = output[0]
        #output = self.dropout_layer(output)
        # This is to get a sense of the whole sentence embedding. (because we are using a classification task)
        output = torch.mean(output, dim=1)
        output = self.linear(output)
        return output