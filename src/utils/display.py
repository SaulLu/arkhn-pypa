import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

def generate_confusion_matrix(conf_matrix, labels_list, curr_time=None, curr_epoch=None, prefix=''):
    path_conf_mat = None

    if curr_epoch and curr_time:
        path_conf_mat = "data/parameters/img/" \
                                + prefix \
                                + curr_time \
                                + "_confusion_matrix" \
                                + '_epoch_' \
                                +  curr_epoch \
                                + ".png"
        
        path_conf_mat_pred = "data/parameters/img/" \
                                + prefix \
                                + curr_time \
                                + "_precision_matrix" \
                                + '_epoch_' \
                                +  curr_epoch \
                                + ".png"
        
        path_conf_mat_true = "data/parameters/img/" \
                                + prefix \
                                + curr_time \
                                + "_recall_matrix_" \
                                + '_epoch_' \
                                +  curr_epoch \
                                + ".png"
    
    df_conf_matrix = pd.DataFrame(conf_matrix, labels_list, labels_list)
    display_confusion_matrix(df_conf_matrix, path=path_conf_mat, title=f'Confusion Matrix - {prefix.capitalize()}')

    df_conf_matrix_true = (df_conf_matrix.T / df_conf_matrix.T.sum()).T * 100
    df_conf_matrix_true = df_conf_matrix_true.round(0)
    display_confusion_matrix(df_conf_matrix_true, path=path_conf_mat_true, 
        title=f'Recall Matrix (%) - {prefix.capitalize()}')
    
    df_conf_matrix_pred = df_conf_matrix / df_conf_matrix.sum() *100
    df_conf_matrix_pred = df_conf_matrix_pred.round(0)
    display_confusion_matrix(df_conf_matrix_pred, path=path_conf_mat_pred, 
        title=f'Precision Matrix (%) - {prefix.capitalize()}')

def display_confusion_matrix(df_conf_matrix, path=None, title=None):
    _, __ = plt.subplots()
    sn.set(font_scale=0.8)
    sn.heatmap(df_conf_matrix, annot=True, fmt='g')
    plt.ylabel('True Tag', va='center')
    plt.xlabel('Predicted Tag')
    plt.title(title)
    if path:
        try:
            plt.savefig(
                path,
                bbox_inches='tight',
                pad_inches=0.5)
        except Exception:
            print("Error building image!: " + path)
            raise
    plt.show()
    plt.close()