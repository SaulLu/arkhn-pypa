import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

def display_confusion_matrix(conf_matrix, labels_list):
    df_conf_matrix = pd.DataFrame(conf_matrix, labels_list, labels_list)
    sn.set(font_scale=1.4)
    sn.heatmap(df_conf_matrix, annot=True, annot_kws={"size": 16})
    plt.show()