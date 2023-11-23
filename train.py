import os
import gc

import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from core.PBCmodel_multi_label import PlasticModel
from sklearn.metrics import roc_auc_score
from config.cfg_multi_label import opt

from tools.save_result import save_loss_plot, save_pred_result, create_logger
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
        
        make_dir(RESULT_MAIN_PATH)
        make_dir(model_path)
        make_dir(method_path)
        make_dir(fold_path)
        make_dir(hyperparameter_path)
        make_dir(auc_path)
        make_dir(loss_path)
        make_dir(log_path)
        make_dir(save_model_path)
        make_dir(pred_path)

        train_dataloader, val_dataloader, test_dataloader, tr_weight, val_weight, te_weight = trainer.set_data(batch_size, mth, fold_count)
        model = PlasticModel(kr_size).cuda()
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 2)
        
        loss_save_name = f'{fold_count}_Fold_{mth_name}_Loss.png'
        epoch_list = list()
        train_loss_list = list()
        val_loss_list = list()
        test_loss_list = list()
        
        val_check_count = 0
        for epoch_idx in range(1, epoch + 1):
            auc_save_name = f'Epoch_{epoch_idx}_{mth_name}_AUC.png'
            weighted_auc_save_name = f'Epoch_{epoch_idx}_{mth_name}_weighted_AUC.png'
            log_save_name = f'{fold_count}_Fold_{mth_name}.log'
            pred_file_name = f'{fold_count}_Fold_{mth_name}_result.csv'
            weighted_pred_file_name = f'{fold_count}_Fold_{mth_name}_weighted_result.csv'
            model_save_name = f'{fold_count}_Fold_{mth_name}.pth'
            
            model.train()
            epoch_list.append(epoch_idx)
            
            model, train_loss = trainer.train_model(train_dataloader, model, optimizer, tr_weight, batch_size)
            
            model.eval()
            with torch.no_grad():
                model, val_loss = trainer.val_model(val_dataloader, model, val_weight, batch_size)
                model, test_pred_list, test_label_list = trainer.test_model(test_dataloader, model, te_weight, batch_size)
                
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            
            
            thresholds = [i/1000 for i in range(1001)]
            macro_f1 = 0
            micro_f1 = 0
            threshold = 0
            
            np_test_label_list = np.array(test_label_list)
            np_test_pred_list = np.array(test_pred_list)
            
            for cur_threshold in thresholds : 
                cur_macro_f1 = sklearn.metrics.f1_score(np_test_label_list, (np_test_pred_list > threshold).astype('int'), average='macro')
                cur_micro_f1 = sklearn.metrics.f1_score(np_test_label_list, (np_test_pred_list > threshold).astype('int'), average='micro')

                if macro_f1 < cur_macro_f1 : 
                    threshold = cur_threshold
                    macro_f1 = cur_macro_f1
                    micro_f1 = cur_micro_f1

            cur_lr = optimizer.param_groups[0]['lr']
            fileHandler, streamHandler, logger = create_logger(log_path + log_save_name)
            
            #binary
            logger.info(f'{fold_count}Fold Epoch : {epoch_idx} | Train_loss : {train_loss:.4f} | Val_loss : {val_loss:.4f} | Mi_F1 : {micro_f1:.4f} | Ma_F1 : {macro_f1:.4f} | Threshold : {threshold} | lr : {cur_lr:.7f}')

            fileHandler.close()
            logger.removeHandler(fileHandler)
            logger.removeHandler(streamHandler)

            save_pred_result(test_pred_list, test_label_list, pred_path + pred_file_name)

            if check_val_loss > val_loss:# and count_val_loss != 2:
                check_val_loss = val_loss
                torch.save(model.state_dict(), save_model_path + model_save_name)
                val_check_count = 0
                
            del test_pred_list, test_label_list, train_loss, val_loss
                
        save_loss_plot(train_loss_list, val_loss_list, epoch_list, loss_path + loss_save_name)
            
        del train_loss_list, val_loss_list, epoch_list
        gc.collect()
        torch.cuda.empty_cache()
        print()

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