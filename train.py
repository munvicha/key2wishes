from email.mime import base
from lib2to3.pgen2.tokenize import tokenize
import config, dataset, model, engine
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import pandas as pd
import numpy as np
from math import inf

def split_data(data, size = 0.8):
  data = data.sample(frac = 1).reset_index(drop = True)
  train_index = int(np.floor(len(data)*size))
  train_df = data[:train_index]
  valid_df = data[train_index:]

  return train_df, valid_df

def run():
    df = pd.read_csv(config.TRAINING_FILE)
    kw = []
    for s in df['Key Words']:
        s = s.replace('[', '')
        s = s.replace(']', '')
        s = s.replace("'", '')
        s = s.replace(',', '')
        kw.append(s)

    df.drop(['Unnamed: 0', 'Key Words'], axis = 1, inplace = True)
    df['Key Words'] = kw

    train_df, valid_df = split_data(df)
    train_dataset = dataset.myDataset(train_df, config.TOKENIZER)
    valid_dataset = dataset.myDataset(valid_df, config.TOKENIZER)
    train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE,  shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.VALID_BATCH_SIZE, shuffle=True)

    model = model.model
    model.to(config.DEVICE)
    optimizer = AdamW(model.parameters, lr = 4e-4, eps = 1e-9)

    base_loss = inf
    for epoch in range(config.EPOCHS):
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, config.EPOCHS))
        train_loss = engine.train(train_loader=train_loader, model=model, optimizer=optimizer)
        valid_loss = engine.evaluate(valid_loader=valid_loader, model = model, optimizer=optimizer)
        print(f'Train loss: {train_loss} | Valid loss: {valid_loss}')

        #save model
        if valid_loss < base_loss:
            base_loss = valid_loss
            model.save_pretrained(config.MODEL_PATH)

if __name__ == '__main__':
    run()
