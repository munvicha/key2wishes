import torch
from tqdm import tqdm

def train(train_loader, model, optimizer):
  epoch_loss = 0
  model.train()
  for batch in tqdm(train_loader):
    optimizer.zero_grad()
    output = model(input_ids=batch["input_ids"],
            attention_mask=batch["input_mask"],
            decoder_attention_mask=batch['target_mask'],
            labels=batch['labels'])
    loss = output[0]
    epoch_loss += loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

  return (epoch_loss/len(train_loader))

def evaluate(valid_loader, model):
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for batch in tqdm(valid_loader):
          output = model(input_ids=batch["input_ids"],
                        attention_mask=batch["input_mask"],
                        decoder_attention_mask=batch['target_mask'],
                        labels=batch['labels'])
          loss = output[0]
          valid_loss += loss

    return (valid_loss/len(valid_loader))