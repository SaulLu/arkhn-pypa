import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

def generate_confusion_matrix(conf_matrix, labels_list, curr_time=None, curr_epoch=None):
    path_conf_mat = None
    print(f"curr_epoch {curr_epoch}")
    print(f"curr_time {curr_time}")

    if curr_epoch and curr_time:
        path_conf_mat = "data/parameters/img/" \
                                + curr_time \
                                + "_confusion_matrix" \
                                + '_epoch_' \
                                +  curr_epoch \
                                + ".jpeg"
        
        path_conf_mat_pred = "data/parameters/img/" \
                                + curr_time \
                                + "_confusion_matrix_pred" \
                                + '_epoch_' \
                                +  curr_epoch \
                                + ".jpeg"
        
        path_conf_mat_true = "data/parameters/img/" \
                                + curr_time \
                                + "_confusion_matrix_true" \
                                + '_epoch_' \
                                +  curr_epoch \
                                + ".jpeg"
    
    df_conf_matrix = pd.DataFrame(conf_matrix, labels_list, labels_list)
    display_confusion_matrix(df_conf_matrix, path=path_conf_mat, title='Confusion Marix')

    df_conf_matrix_true = (df_conf_matrix.T / df_conf_matrix.T.sum()).T * 100
    display_confusion_matrix(df_conf_matrix_true, path=path_conf_mat_true, 
        title='Visualization of the distribution of predicted tags for a given real tag')
    
    df_conf_matrix_pred = df_conf_matrix / df_conf_matrix.sum() *100
    display_confusion_matrix(df_conf_matrix_pred, path=path_conf_mat_pred, 
        title='Visualization of the distribution of real tags for a given predicted tag')

def display_confusion_matrix(df_conf_matrix, path=None, title=None):
    _, __ = plt.subplots()
    sn.set(font_scale=0.8)
    sn.heatmap(df_conf_matrix, annot=True, fmt='d')
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
    plt.show()