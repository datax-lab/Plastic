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

warnings.filterwarnings('ignore')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
torch.set_float32_matmul_precision('medium')


class CustomDataset(Dataset):
    
    def __init__(self, mode, encoding):
        self.df = pd.DataFrame(data = np.load(f'./data/{encoding}_{mode}.npy', allow_pickle=True), columns = ['Sequence', 'Label', 'fold'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return torch.Tensor(self.df['Sequence'].iloc[idx]).float(), torch.Tensor(self.df['Label'].iloc[idx]).float()
    
class CustomTest(Dataset):
    
    def __init__(self, mode, encoding):
        self.df = pd.DataFrame(data = np.load(f'./data/{encoding}_test.npy', allow_pickle=True), columns = ['Sequence', 'Label'])
        self.df.Label = self.df.Label.apply(lambda x : x[:-1])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return torch.Tensor(self.df['Sequence'].iloc[idx]).float(), torch.Tensor(self.df['Label'].iloc[idx]).float()
    
    
class ECDataModule(pl.LightningDataModule):

    def __init__(self, batch_size, train_fold, test_fold, mode, encoding):
        super().__init__()
        self.batch_size = batch_size
        self.df = CustomDataset(mode, encoding)
        self.test = CustomTest(mode, encoding)
        self.train_fold = train_fold
        self.test_fold = test_fold

    def prepare_data(self) : 
        pass

    def setup(self, stage=None):
        self.trainset = torch.utils.data.Subset(self.df, self.df.df[self.df.df.fold.isin(self.train_fold)].index)
        self.valset = torch.utils.data.Subset(self.df, self.df.df[self.df.df.fold.isin(self.test_fold)].index)
        self.testset = torch.utils.data.Subset(self.df, self.df.df.index)
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size,
                                          shuffle=True, num_workers=1, drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valset, batch_size=self.batch_size,
                                           num_workers=1)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size,
                                           shuffle=False, num_workers=1)

# In[4]:


class EMATracker:
    
    def __init__(self, alpha: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self._value = None

    def update(self, new_value):
        if self._value is None:
            self._value = new_value
        else:
            self._value = (new_value * self.alpha + self._value * (1-self.alpha))

    @property
    def value(self):
        return self._value


def integrate(x, y):
    sm = 0
    for i in range(1, len(x)):
        h = x[i-1] - x[i]
        sm += h * (y[i] + y[i-1]) / 2

    return sm


class Model(pl.LightningModule):

    def __init__(self, dimension, mode):
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
        self.dense3 = nn.Linear(512, 13)
            
        self.micro_aucpr = [0, 0]
        self.macro_aucpr = [0, 0]
        self.micro_f1 = [0, 0]
        self.macro_f1 = [0, 0]
        self.mode = mode
        self.threshold = -1
        
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
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
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
    
    def validation_epoch_end(self, validation_step_outputs):
        pred = torch.cat([i[0] for i in validation_step_outputs], dim=0).cpu().detach().numpy()
        y = torch.cat([i[1] for i in validation_step_outputs], dim=0).cpu().detach().numpy()
        mask = y.sum(axis=0).astype('bool')
        pred = pred[:,mask]
        y = y[:,mask]

        macro_auc = []
        for category in range((y.shape[1])):
            precision, recall, thresholds = precision_recall_curve(y[:,category], pred[:,category])
            area = integrate(recall, precision)
            macro_auc.append(area)
            
        precision, recall, _ = precision_recall_curve(y.ravel(), pred.ravel())
        micro_auc = integrate(recall, precision)
        
        if self.threshold > 0: 
            
            thresholds = [i/1000 for i in range(1001)]

            for threshold in thresholds : 
                if threshold > self.threshold : 
                    macro_f1 = sklearn.metrics.f1_score(y, (pred > threshold).astype('int'), average='macro')
                    micro_f1 = sklearn.metrics.f1_score(y, (pred > threshold).astype('int'), average='micro')

                    if macro_f1 > self.macro_f1[0] : 
                        self.macro_f1[0] = macro_f1
                        self.micro_f1[0] = micro_f1
            
        self.micro_aucpr[0] = min(micro_auc, 1.0) 
        self.macro_aucpr[0] = min(np.mean(macro_auc), 1.0)
        return 
    
     
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.transpose(1, 2).contiguous()
        y_hat = self(x)
        val_loss = nn.BCEWithLogitsLoss()(y_hat, y).cpu().detach().numpy()
        pred = torch.sigmoid(y_hat)
        return pred, y
    
    def test_epoch_end(self, test_step_outputs):
        pred = torch.cat([i[0] for i in test_step_outputs], dim=0).cpu().detach().numpy()
        y = torch.cat([i[1] for i in test_step_outputs], dim=0).cpu().detach().numpy()
        mask = y.sum(axis=0).astype('bool')
        pred = pred[:,mask]
        y = y[:,mask]

        macro_auc = []
        for category in range((y.shape[1])):
            precision, recall, thresholds = precision_recall_curve(y[:,category], pred[:,category])
            area = integrate(recall, precision)
            macro_auc.append(area)
            
        precision, recall, _ = precision_recall_curve(y.ravel(), pred.ravel())
        micro_auc = integrate(recall, precision)
        
        thresholds = [i/1000 for i in range(1001)]
        
        for threshold in thresholds : 
            if threshold > self.threshold : 
                macro_f1 = sklearn.metrics.f1_score(y, (pred > threshold).astype('int'), average='macro')
                micro_f1 = sklearn.metrics.f1_score(y, (pred > threshold).astype('int'), average='micro')
                
                if macro_f1 > self.macro_f1[1] : 
                    self.best_threshold = threshold
                    self.macro_f1[1] = macro_f1
                    self.micro_f1[1] = micro_f1
            
        self.micro_aucpr[1] = min(micro_auc, 1.0) 
        self.macro_aucpr[1] = min(np.mean(macro_auc), 1.0)
        
        return 
    
    def find_threshold(self, device) : 
        
        print('Thresholding') 
        df = pd.DataFrame(data = np.load(f'./data/{encoding}_binary.npy', allow_pickle=True), columns = ['Sequence', 'Label', 'Fold']).drop(columns=['Fold'])
        #df.Label = df.Label.apply(lambda x : 1 - x) 
        df = df[df.Label == 0].reset_index()
        
        preds = torch.zeros(len(df), 13)
        
        for index, row in tqdm(df.iterrows(), total=len(df)) : 
            data = torch.Tensor([row['Sequence']]).to(device).transpose(1, 2).contiguous()
            pred = self(data)
            pred = pred.detach().cpu().squeeze()
            preds[index] = torch.sigmoid(pred)
            
        thresholds = [i/1000 for i in range(1001)]
        
        for threshold in thresholds : 
            if (preds > threshold).sum().numpy() / (13 * len(df)) < 0.01 : 
                self.threshold = threshold
                return
            

encodings = {'One_hot_6_bit': 6, 'Binary_5_bit': 5, 'Hydrophobicity_matrix': 20, 'Meiler_parameters': 7, 'Acthely_factors': 5, 'PAM250': 20, 'BLOSUM62': 20, 'Miyazawa_energies': 20, 'Micheletti_potentials': 20, 'AESNN3': 3, 'ANN4D': 4, 'ProtVec': 100, 'one_hot': 21}

with open('./positive_only_1fdr.txt', 'w+') as f : 
        f.write(f'encoding testing set : macro aucpr  micro aucpr macro f1 micro f1\n')

for encoding, dimension in encodings.items() : 
    mode = 'labels'
    val_macro_aucprs = []
    val_micro_aucprs = []
    val_macro_f1s = []
    val_micro_f1s = []
    test_macro_aucprs = []
    test_micro_aucprs = []
    test_macro_f1s = []
    test_micro_f1s = []
    for fold in [0, 1, 2, 3, 4] : 
        
        device = torch.device('cuda')

        folds = [0, 1, 2, 3, 4]
        folds.remove(fold)
        data_module = ECDataModule(16, folds, [fold], mode, encoding)

        model = Model(dimension, mode)
        trainer = Trainer(max_epochs=15, gpus=1, deterministic=True, enable_progress_bar=True, num_sanity_val_steps=0)
        trainer.fit(model, data_module)
        model.eval()
        model.to(device)
        model.find_threshold(device) 
        trainer.validate(model, data_module)
        trainer.test(model, data_module)
        
        val_macro_aucprs.append(model.macro_aucpr[0])
        val_micro_aucprs.append(model.micro_aucpr[0])
        val_macro_f1s.append(model.macro_f1[0])
        val_micro_f1s.append(model.micro_f1[0])
        
        test_macro_aucprs.append(model.macro_aucpr[1])
        test_micro_aucprs.append(model.micro_aucpr[1])
        test_macro_f1s.append(model.macro_f1[1])
        test_micro_f1s.append(model.micro_f1[1])

    with open('./positive_only_1fdr.txt', 'a+') as f : 
        f.write(f'{encoding} val : {np.mean(val_macro_aucprs)}±{np.std(val_macro_aucprs)} {np.mean(val_micro_aucprs)}±{np.std(val_micro_aucprs)} {np.mean(val_macro_f1s)}±{np.std(val_macro_f1s)} {np.mean(val_micro_f1s)}±{np.std(val_micro_f1s)}\n')
        f.write(f'{encoding} test : {np.mean(test_macro_aucprs)}±{np.std(test_macro_aucprs)} {np.mean(test_micro_aucprs)}±{np.std(test_micro_aucprs)} {np.mean(test_macro_f1s)}±{np.std(test_macro_f1s)} {np.mean(test_micro_f1s)}±{np.std(test_micro_f1s)}\n')