import gc
import torch
import string
import numpy as np
import pandas as pd

from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

LABEL_LIST = ['NYLON', 'PBAT', 'PBS', 'PBSA', 'PCL', 'PE', 'PEA', 'PES', 'PET', 'PHA', 'PHB', 'PLA', 'PU', 'NEGATIVE']

def get_data(negative, positive):
    neg_data = list(SeqIO.parse(open(negative), 'fasta'))
    neg_label = np.zeros(len(neg_data), dtype = np.int)

    pos_data = list(SeqIO.parse(open(positive), 'fasta'))
    pos_label = np.ones(len(pos_data), dtype = np.int)
    
    neg_seq_list = list()
    for i in neg_data:
        neg_seq_list.append(i.seq)
        
    pos_seq_list = list()
    for i in pos_data:
        pos_seq_list.append(i.seq)
        
    total_data = neg_seq_list + pos_seq_list
    total_label = np.concatenate((neg_label, pos_label), axis = 0)
    
    del neg_data, pos_data, neg_seq_list, neg_label, pos_seq_list, pos_label
    gc.collect()
    
    return total_data, total_label

def get_csv_file(path):
    
    read_data = pd.read_csv(path)
    
    seq_data = read_data['seq']
    labal_data = read_data['label']
    
    del read_data
    gc.collect()
    
    return seq_data, labal_data

class PlasticDataset:
    def __init__(self, seq, label, encoding):
        self.seq = seq
        self.label = label
        self.encoding = encoding
        
    def __len__(self):
        return len(self.seq)
    
    def __getitem__(self, idx):
        
        if self.encoding.__name__ == 'pssm':
            seq = self.encoding(self.seq[idx], LABEL_LIST.index(self.label[idx]))
        else:
            seq = self.encoding(self.seq[idx])

        label = torch.FloatTensor(make_multi_label(self.label[idx]))

        seq = np.array(seq)
        seq = torch.FloatTensor(seq)

        return seq, label

def make_multi_label(data):
    data = data.translate(str.maketrans('', '', string.punctuation))
    data = data.split(' ')
    label_np = np.zeros(len((LABEL_LIST)))
    
    for i in range(len(data)):
        label_np[int(data[i])] = 1
    return label_np
        