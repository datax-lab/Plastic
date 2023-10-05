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
import math

warnings.filterwarnings('ignore')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
torch.set_float32_matmul_precision('medium')


class CustomDataset(Dataset):
    
    def __init__(self, mode):
        self.df = pd.DataFrame(data = np.load(f'./data/transformer_{mode}.npy', allow_pickle=True), columns = ['Sequence', 'Label', 'fold'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return torch.Tensor(self.df['Sequence'].iloc[idx]).int(), torch.Tensor(self.df['Label'].iloc[idx]).float()
    
class CustomTest(Dataset):
    
    def __init__(self, mode):
        self.df = pd.DataFrame(data = np.load(f'./data/transformer_test.npy', allow_pickle=True), columns = ['Sequence', 'Label'])
        self.df.Label = self.df.Label.apply(lambda x : x[:-1])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return torch.Tensor(self.df['Sequence'].iloc[idx]).float(), torch.Tensor(self.df['Label'].iloc[idx]).float()
    
    
class ECDataModule(pl.LightningDataModule):

    def __init__(self, batch_size, train_fold, test_fold, mode):
        super().__init__()
        self.batch_size = batch_size
        self.df = CustomDataset(mode)
        self.test = CustomTest(mode)
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


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
class TransformerEmbedding(nn.Module): 

    def __init__(self, vocab_size, embed_size, dropout):
        super().__init__()
        self.token = nn.Embedding(vocab_size, embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim, max_len=1000)
        self.dropout = nn.Dropout(dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        x = self.token(sequence) + self.position(sequence)
        return self.dropout(x)
    
class MultiHeadedAttention(nn.Module):

    def __init__(self, h=8, d_model=256, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        
        self.d_k = d_model // h
        self.h = h
        self.dim = d_model
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.h, self.d_k)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, mask=None):
        
        x = self.norm(x)
        
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        query = self.transpose_for_scores(query)
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) 
        scores = scores / math.sqrt(query.size(-1))

        p_attn = self.softmax(scores)

        x = torch.matmul(p_attn, value)
        
        batch_size = query.size(0)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        
        x = self.dropout(x)
        
        return x
    
    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    
class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
    
class SublayerConnection(nn.Module):
    
    def __init__(self, size, feed_forward_hidden, dropout):
        super(SublayerConnection, self).__init__()
        self.feed_forward = PositionwiseFeedForward(d_model=size, d_ff=feed_forward_hidden, dropout=dropout)
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x +  self.dropout(self.feed_forward(self.norm(x)))
    
class TransformerBlock(nn.Module):

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(attn_heads, hidden, dropout)
        self.output_sublayer = SublayerConnection(hidden, feed_forward_hidden, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x1 = self.attention(x, mask)
        x = x + x1
        x = self.output_sublayer(x)
        return self.dropout(x)
    
class Transformer(nn.Module) :
    
    def __init__(self, mode, vocab_size=23, dimension=256, attn_heads=2, dropout=0.1) :
        super().__init__()
        self.embeddings = TransformerEmbedding(vocab_size, dimension, dropout)
        self.enc_1 = TransformerBlock(dimension, attn_heads, dimension*4, dropout)
        self.enc_2 = TransformerBlock(dimension, attn_heads, dimension*4, dropout)
        self.enc_3 = TransformerBlock(dimension, attn_heads, dimension*4, dropout)
        self.enc_4 = TransformerBlock(dimension, attn_heads, dimension*4, dropout)
        self.classifier = nn.Linear(dimension, 13)
        
    def forward(self, x):
        mask = None
        emb = self.embeddings(x)
        x = self.enc_1(emb, mask)
        x = self.enc_2(x, mask)
        x = self.enc_3(x, mask)
        x = self.enc_4(x, mask)
        x = x[:,0,:]
        x = self.classifier(x)
        return x

    
def integrate(x, y):
    sm = 0
    for i in range(1, len(x)):
        h = x[i-1] - x[i]
        sm += h * (y[i] + y[i-1]) / 2

    return sm


class Model(pl.LightningModule):

    def __init__(self, mode):
        super().__init__()
        
        self.model = Transformer(mode)
            
        self.micro_aucpr = [0, 0]
        self.macro_aucpr = [0, 0]
        self.micro_f1 = [0, 0]
        self.macro_f1 = [0, 0]
        self.mode = mode
        self.threshold = -1
        
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.BCEWithLogitsLoss()(y_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
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
        df = pd.DataFrame(data = np.load(f'./data/transformer_binary.npy', allow_pickle=True), columns = ['Sequence', 'Label', 'Fold']).drop(columns=['Fold'])
        #df.Label = df.Label.apply(lambda x : 1 - x) 
        df = df[df.Label == 0].reset_index()
        
        preds = torch.zeros(len(df), 13)
        
        for index, row in tqdm(df.iterrows(), total=len(df)) : 
            data = torch.Tensor([row['Sequence']]).to(device).int()
            pred = self(data)
            pred = pred.detach().cpu().squeeze()
            preds[index] = torch.sigmoid(pred)
            
        thresholds = [i/1000 for i in range(1001)]
        
        for threshold in thresholds : 
            if (preds > threshold).sum().numpy() / (13 * len(df)) < 0.05 : 
                self.threshold = threshold
                return
            

with open('./positive_only_trans_5fdr.txt', 'w+') as f : 
        f.write(f'testing set : macro aucpr  micro aucpr macro f1 micro f1\n')
        
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
    data_module = ECDataModule(16, folds, [fold], mode)

    model = Model(mode)
    trainer = Trainer(max_epochs=25, gpus=1, deterministic=True, enable_progress_bar=True, num_sanity_val_steps=0)
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

with open('./positive_only_trans_5fdr.txt', 'a+') as f : 
    f.write(f'val : {np.mean(val_macro_aucprs)}±{np.std(val_macro_aucprs)} {np.mean(val_micro_aucprs)}±{np.std(val_micro_aucprs)} {np.mean(val_macro_f1s)}±{np.std(val_macro_f1s)} {np.mean(val_micro_f1s)}±{np.std(val_micro_f1s)}\n')
    f.write(f'test : {np.mean(test_macro_aucprs)}±{np.std(test_macro_aucprs)} {np.mean(test_micro_aucprs)}±{np.std(test_micro_aucprs)} {np.mean(test_macro_f1s)}±{np.std(test_macro_f1s)} {np.mean(test_micro_f1s)}±{np.std(test_micro_f1s)}\n')