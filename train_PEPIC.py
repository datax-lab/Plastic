import sys
import pandas as pd
import os
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold
import sklearn
from sklearn import preprocessing
from sklearn.metrics import auc, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything, utilities
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import time
import warnings
import random
import argparse
import copy

encodings = {'Miyazawa_energies': 20}


class CustomDataset(Dataset):
    
    def __init__(self, mode, encoding):
        self.df = pd.DataFrame(data = np.load(f'./folder/data.npy', allow_pickle=True), columns = ['Sequence', 'Label', 'fold'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return torch.Tensor(self.df['Sequence'].iloc[idx]).float(), torch.Tensor(self.df['Label'].iloc[idx]).float()
    
class CustomTest(Dataset):
    
    def __init__(self, mode, encoding):
        self.df = pd.DataFrame(data = np.load(f'./folder/curated_data.npy', allow_pickle=True), columns = ['Sequence', 'Label'])
        self.df.Label = self.df.Label.apply(lambda x : x[:-1])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return torch.Tensor(self.df['Sequence'].iloc[idx]).float(), torch.Tensor(self.df['Label'].iloc[idx]).float()
    
    
class ECDataModule(pl.LightningDataModule):

    def __init__(self, batch_size, train_fold, val_fold, test_fold, mode, encoding):
        super().__init__()
        self.batch_size = batch_size
        self.df = CustomDataset(mode, encoding)
        self.test = CustomTest(mode, encoding)
        self.train_fold = train_fold
        self.val_fold = val_fold
        self.test_fold = test_fold

    def prepare_data(self) : 
        pass

    def setup(self, stage=None):
        self.trainset = torch.utils.data.Subset(self.df, self.df.df[self.df.df.fold.isin(self.train_fold)].index)
        self.valset = torch.utils.data.Subset(self.df, self.df.df[self.df.df.fold.isin(self.val_fold)].index)
        self.testset = torch.utils.data.Subset(self.df, self.df.df[self.df.df.fold.isin(self.test_fold)].index)
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size,
                                          shuffle=True, num_workers=2, drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valset, batch_size=self.batch_size,
                                           num_workers=2)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size,
                                           shuffle=False, num_workers=2)


class Model(pl.LightningModule):

    def __init__(self, dimension, lr):
        super().__init__()
        
        self.norm1 = nn.BatchNorm1d(128)
        self.norm2 = nn.BatchNorm1d(128)
        self.norm3 = nn.BatchNorm1d(128)
        self.norm4 = nn.BatchNorm1d(512)
        self.norm5 = nn.BatchNorm1d(512)
        
        self.conv1 = nn.Conv1d(dimension, 128, 4)
        self.conv2 = nn.Conv1d(dimension, 128, 8)
        self.conv3 = nn.Conv1d(dimension, 128, 16)
        
        self.max1 = nn.MaxPool1d(997)
        self.max2 = nn.MaxPool1d(993)
        self.max3 = nn.MaxPool1d(985)
        
        self.dense1 = nn.Linear(128*3, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 11)
            
        self.micro_aucpr = [0, 0]
        self.macro_aucpr = [0, 0]
        self.lr = lr
        
    def forward(self, x):
        x1 = self.max1(F.relu(self.conv1(x)))
        x2 = self.max2(F.relu(self.conv2(x)))
        x3 = self.max3(F.relu(self.conv3(x)))
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x3 = torch.flatten(x3, 1)
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        x3 = self.norm3(x3)
        x = torch.cat((x1, x2, x3), dim=1)
        x = F.relu(self.norm4(self.dense1(x)))
        x = F.relu(self.norm5(self.dense2(x)))
        x = self.dense3(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.transpose(1, 2).contiguous()
        y_hat = self(x)
        loss = nn.BCEWithLogitsLoss()(y_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.transpose(1, 2).contiguous()
        y_hat = self(x)
        val_loss = nn.BCEWithLogitsLoss()(y_hat, y).cpu().detach().numpy()
        pred = torch.sigmoid(y_hat)
        return pred, y
     
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.transpose(1, 2).contiguous()
        y_hat = self(x)
        val_loss = nn.BCEWithLogitsLoss()(y_hat, y).cpu().detach().numpy()
        pred = torch.sigmoid(y_hat)
        return pred, y

encoding = 'Miyazawa_energies'
dimension = 21

data_module = ECDataModule(16, [0, 1, 2, 3, 4, 5, 6, 7], [8], [9], encoding, experiment)
    
model = Model(dimension, 1e-4)
trainer = Trainer(max_epochs=50, gpus=1, deterministic=True, enable_progress_bar=False, num_sanity_val_steps=0)
trainer.fit(model, data_module)