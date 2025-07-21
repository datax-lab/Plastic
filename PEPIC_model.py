import sys
import pandas as pd
import os
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences  
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

class Model(pl.LightningModule):

    def __init__(self, dimension, mode, lr):
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