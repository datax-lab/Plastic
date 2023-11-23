import os
import gc

import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from core.PBCmodel_multi_label import PlasticModel
from sklearn.metrics import roc_auc_score
from config.cfg_multi_label import opt

from tools.save_result import save_AUC, save_loss_plot, save_pred_result, create_logger
from core.trainer_multi_label import Trainer
import sklearn
from sklearn.metrics import precision_recall_curve, precision_score, recall_score

from datetime import datetime

today = str(datetime.now())
today = today[:10]
RESULT_MAIN_PATH = f'result_multi_label/2023-10-30'

def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def integrate(x, y):
    sm = 0
    for i in range(1, len(x)):
        h = x[i-1] - x[i]
        sm += h * (y[i] + y[i-1]) / 2
    return sm

        
def get_result(pred, label, ma_f1, mi_f1, ma_au, mi_au, isVal = True):
    idx = 1
    threshold = 0
    thresholds = [i/1000 for i in range(1001)]
    
    if isVal:
        idx = 0

    np_pred = np.array(pred)
    np_label = np.array(label)
    
    macro_auc = list()

    for category in range((np_label.shape[1])):
        precision, recall, thresholds = precision_recall_curve(np_label[:,category], np_pred[:,category])
        area = integrate(recall, precision)
        macro_auc.append(area)
        
    for cur_threshold in thresholds : 
        cur_macro_f1 = sklearn.metrics.f1_score(np_label, (np_pred > cur_threshold).astype('int'), average='macro')
        cur_micro_f1 = sklearn.metrics.f1_score(np_label, (np_pred > cur_threshold).astype('int'), average='micro')

        if ma_f1[idx] < cur_macro_f1 : 
            threshold = cur_threshold
            ma_f1[idx] = cur_macro_f1
            mi_f1[idx] = cur_micro_f1
                    
    precision, recall, _ = precision_recall_curve(np_label.ravel(), np_pred.ravel())
    micro_auc = integrate(recall, precision)
    
    ma_au[idx] = min(np.mean(macro_auc), 1.0)
    mi_au[idx] = min(micro_auc, 1.0) 
    
    return ma_f1, mi_f1, ma_au, mi_au, threshold
    
                
def train_model(batch_size, lr, fold):
    epoch = opt.epoch
    method_list = opt.method_list
    kr_size_list = opt.kr_size_list
    method_name_list = opt.method_name_list
    trainer = Trainer()
    fold_count = fold
    for mth, kr_size, mth_name in zip(method_list, kr_size_list, method_name_list):
        print(f'method : {mth_name}')
        check_val_loss = 1000
        
        model_path = f'{RESULT_MAIN_PATH}/CNN_NEW/'
        method_path = f'{model_path}{mth_name}/'
        fold_path = f'{method_path}{fold_count}_Fold/'
        hyperparameter_path = f'{fold_path}{lr}_batchsize_{batch_size}/'
        auc_path = f'{hyperparameter_path}AUC/'
        loss_path = f'{hyperparameter_path}Loss/'
        log_path = f'{hyperparameter_path}Log/'
        save_model_path = f'{hyperparameter_path}Model/'
        pred_path = f'{hyperparameter_path}PredResult/'

        log_save_name = f'{fold_count}_Fold_result{mth_name}.log'
        pred_file_name = f'{fold_count}_Fold_{mth_name}_result.csv'
        weighted_pred_file_name = f'{fold_count}_Fold_{mth_name}_weighted_result.csv'
        model_save_name = f'{fold_count}_Fold_{mth_name}.pth'
        
        train_dataloader, val_dataloader, test_dataloader, tr_weight, val_weight, te_weight = trainer.set_data(batch_size, mth, fold_count)

        model = PlasticModel(kr_size).cuda()
        model.load_state_dict(torch.load(save_model_path + model_save_name))
        model.eval()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 2)
        
        loss_save_name = f'{fold_count}_Fold_{mth_name}_Loss.png'
        epoch_list = list()
        train_loss_list = list()
        val_loss_list = list()
        test_loss_list = list()
        
        val_check_count = 0
        
        macro_f1_list = [0., 0.]
        micro_f1_list = [0., 0.]
        macro_aucpr = [0., 0.]
        micro_aucpr = [0., 0.]
        
        with torch.no_grad():
            model, val_pred_list, val_label_list = trainer.test_model(val_dataloader, model, 0, batch_size)
            model, test_pred_list, test_label_list = trainer.test_model(test_dataloader, model, te_weight, batch_size)
            
            
            macro_f1_list, micro_f1_list, macro_aucpr, micro_aucpr, val_threshold = get_result(val_pred_list, val_label_list, macro_f1_list, micro_f1_list, macro_aucpr, micro_aucpr, isVal = True)
            macro_f1_list, micro_f1_list, macro_aucpr, micro_aucpr, test_threshold = get_result(test_pred_list, test_label_list, macro_f1_list, micro_f1_list, macro_aucpr, micro_aucpr, isVal = False)

            fileHandler, streamHandler, logger = create_logger(hyperparameter_path + log_save_name)
            
            #binary
            logger.info(f'Validation {fold_count}Fold | Mi_F1 : {macro_f1_list[0]:.4f} | Ma_F1 : {micro_f1_list[0]:.4f} | Ma_AUC : {macro_aucpr[0]} | Mi_AUC : {macro_aucpr[0]} | Threshold : {val_threshold}')
            logger.info(f'Test {fold_count}Fold | Mi_F1 : {macro_f1_list[1]:.4f} | Ma_F1 : {micro_f1_list[1]:.4f} | Ma_AUC : {macro_aucpr[1]} | Mi_AUC : {macro_aucpr[1]} | Threshold : {test_threshold}')

            fileHandler.close()
            logger.removeHandler(fileHandler)
            logger.removeHandler(streamHandler)

        gc.collect()
        torch.cuda.empty_cache()
        print()

def F_score(output, label, threshold=0.5, beta=1): #Calculate the accuracy of the model
    prob = output > threshold
    label = label > threshold

    TP = (prob & label).sum(1).float()
    TN = ((~prob) & (~label)).sum(1).float()
    FP = (prob & (~label)).sum(1).float()
    FN = ((~prob) & label).sum(1).float()

    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    return F2.mean(0)

def main():
    batch_size_list = [16, 32, 64]
    lr_list = [1e-4, 1e-5, 1e-6]
    fold_list = [0, 1, 2, 3, 4]
    for fold in fold_list:
        for bt_size in batch_size_list:
            for lr in lr_list:
                train_model(bt_size, lr, fold)    
if __name__ == "__main__":
    main()
