import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import math
import string
import numpy as np 

from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

from dataset.PBCdataset_multi_label import PlasticDataset, get_csv_file, PlasticDatasetBRNN
from config.cfg_multi_label import opt

from .aslloss import AsymmetricLossOptimized

class Trainer():
    
    def train_model(self, loader, model, optimizer, class_weight, batch_size):
        train_loss = 0
        
        class_weight = torch.FloatTensor(class_weight)
        class_weight = class_weight.cuda()
        criterion = nn.BCEWithLogitsLoss(weight = class_weight)

        count = 0
        model.train()
        for tr_i, data in enumerate(loader):
            torch.cuda.empty_cache()

            
            seq, label = data
            seq = seq.view(seq.size(0), seq.size(2), seq.size(1))
            seq = seq.cuda()
            label = label.cuda()

            
            optimizer.zero_grad()
            output = model(seq)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss / len(loader)

        return model, train_loss

    def val_model(self, loader, model, class_weight, batch_size):
        val_loss = 0
        
        class_weight = torch.FloatTensor(class_weight).cuda()

        criterion = nn.BCEWithLogitsLoss(weight = class_weight)

        model.eval()
        for va_i, data in enumerate(loader):
            torch.cuda.empty_cache()

            seq, label = data
            seq = seq.view(seq.size(0), seq.size(2), seq.size(1))
            seq = seq.cuda()
            label = label.cuda()

            output = model(seq)

            loss = criterion(output, label)
            val_loss += loss.item()
        val_loss = val_loss / len(loader)

        return model, val_loss

    def test_model(self, loader, model, class_weight, batch_size):
        test_pred_list = list()
        test_label_list = list()
        weight_pred_list = list()
        sigmoid = nn.Sigmoid()
        model.eval()
        for te_i, data in enumerate(loader):
            torch.cuda.empty_cache()
            seq, label = data
            seq = seq.view(seq.size(0), seq.size(2), seq.size(1))
            seq = seq.cuda()
            label = label.cuda()

            pred = model(seq)
            pred = sigmoid(pred)
            test_pred_list.extend(pred.detach().cpu().tolist())
            test_label_list.extend(label.tolist())

        return model, test_pred_list, test_label_list

    def set_data(self, batch_size, mth, fold):
        # multi pet pu pha ne
        train_path = f'./data/multi_data/cnn_brnn/{fold}_fold_train.csv' #amplified data from BLAST
        val_path = f'./data/multi_data/cnn_brnn/{fold}_fold_test.csv' #amplified data from BLAST
        test_path = f'./data/multi_data/cnn_brnn/1_test_data.csv' #plastic biodegradation gene data 
        
        train_seq, train_label = get_csv_file(train_path)
        val_seq, val_label = get_csv_file(val_path)
        test_seq, test_label = get_csv_file(test_path)
    
        traindataset = PlasticDataset(train_seq, train_label, mth)
        valdataset = PlasticDataset(val_seq, val_label, mth)
        testdataset = PlasticDataset(test_seq, test_label, mth)

        train_dataloader = DataLoader(traindataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 1, drop_last = True)
        val_dataloader = DataLoader(valdataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 1, drop_last = True)
        test_dataloader = DataLoader(testdataset, batch_size = 2, shuffle = True, pin_memory = True, num_workers = 1, drop_last = True)
        
        train_count = self.count_class_data_binary(train_label)
        val_count = self.count_class_data_binary(val_label)
        evl_count = self.count_class_data_binary(test_label)
        
        train_count = self.count_class_data(train_label)
        val_count = self.count_class_data(val_label)
        evl_count = self.count_class_data(test_label)

        tr_weight = compute_class_weight(class_weight = 'balanced', classes = np.unique(train_count), y = train_count)
        val_weight = compute_class_weight(class_weight = 'balanced', classes = np.unique(val_count), y = val_count)
        te_weight = compute_class_weight(class_weight = 'balanced', classes = np.unique(evl_count), y = evl_count)

        return train_dataloader, val_dataloader, test_dataloader, tr_weight, val_weight, te_weight
    
    def count_class_data_binary(self, csv_data):
        class_data = ['NYLON', 'PBAT', 'PBS', 'PBSA', 'PCL', 'PE', 'PEA', 'PES', 'PET', 'PHA', 'PHB', 'PLA', 'PU', 'PVA', 'NEGATIVE']
        count_data = np.zeros(2, dtype=np.int64)
        for i in csv_data:
            output_string = i.translate(str.maketrans('', '', string.punctuation))
            output_string = output_string.split(" ")
            
            for z in output_string:
                if int(z) == 13:
                    count_data[0] += 1
                else:
                    count_data[1] += 1
        
        label_count = 0
        label_list = list()
        for y in count_data:
            label_list.extend([label_count] * int(y))
            label_count += 1
            
        return label_list
    