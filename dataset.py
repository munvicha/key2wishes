import torch
from torch.data.utils import Dataset
import config

class myDataset(Dataset):
 
  def __init__(self, data, tokenizer):
    text, keywords = list(), list()
    for index, row in data.iterrows():
      text.append(row['Text'])
      keywords.append(row['Key Words'])

    self.tokenizer = config.TOKENIZER
    self.text = text
    self.keywords = keywords
    
  
  def __getitem__(self, index):
    kws = self.keywords[index]
    input = 'k2w: ' +  kws 
    keywords_encoding = self.tokenizer.encode_plus(input, truncation = True, max_length = config.MAX_LEN, padding = 'max_length', return_tensors = 'pt')
    
    text =  self.text[index] 
    text_encoding = self.tokenizer.encode_plus(text, truncation = True, max_length = config.MAX_LEN, padding = 'max_length', return_tensors = 'pt')
    
    input_ids = keywords_encoding['input_ids'].flatten().to(config.DEVICE)
    input_mask = keywords_encoding['attention_mask'].flatten().to(config.DEVICE)
    target_mask = text_encoding['attention_mask'].flatten().to(config.DEVICE)
    labels = text_encoding['input_ids'].flatten().to(config.DEVICE)
    labels [labels==0] = -100

    sample =  {
        'input_ids': input_ids,
        'input_mask': input_mask,
        "target_mask": target_mask,
         "labels":labels
    }
    return sample

  def __len__(self):
    return len(self.text)