import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve
from sklearn import metrics
from sklearn.utils.class_weight import compute_sample_weight

def save_loss_plot(train, val, epoch, file_name):

    plt.plot(epoch, train, label = 'trapn_loss')
    plt.plot(epoch, val, label = 'val_loss')
    
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc = 'upper right')
    plt.title('Train Val Test Loss')
    plt.savefig(file_name)
    plt.close()
    
def create_logger(file_path):
    import logging
    import logging.handlers

    logger = logging.getLogger(__name__)

    formatter = logging.Formatter('[%(asctime)s] - %(message)s')

    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(file_path)

    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)

    logger.setLevel(level=logging.DEBUG)

    return fileHandler, streamHandler, logger

def gene_save_pred_result(seq, pred, label, path):
    import pandas as pd
    
    csv_dict = {"seq" : seq, "prediction" : pred, "label" : label}
    df_data = pd.DataFrame(csv_dict)
    
    df_data.to_csv(path, index=False)
    
def save_pred_result(pred, label, path):
    import pandas as pd
    
    csv_dict = {"prediction" : pred, "label" : label}
    df_data = pd.DataFrame(csv_dict)
    
    df_data.to_csv(path, index=False)
