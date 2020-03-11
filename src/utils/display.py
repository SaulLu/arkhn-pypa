import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

def display_confusion_matrix(conf_matrix, labels_list, path=None):
    f, ax = plt.subplots()
    df_conf_matrix = pd.DataFrame(conf_matrix, labels_list, labels_list)
    sn.set(font_scale=0.8)
    sn.heatmap(df_conf_matrix, annot=True, fmt='d')
    plt.ylabel('True Tag')
    plt.xlabel('Predicted Tag')
    if path:
        try:
            plt.savefig(
                path,
                bbox_inches='tight',
                pad_inches=0.5)
        except Exception:
            print("Error building image!: " + path)
    plt.show()