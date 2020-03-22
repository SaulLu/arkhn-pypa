import glob
import os
import time
import re
import sys

def get_path_last_model():
    # path = 'data/parameters/intermediate/'
    path = 'data/results'
    list_of_files = []
    latest_file = None
    # r=root, f = files
    for r, _, f in os.walk(path):
        for file in f:
            if '.pt' in file:
                list_of_files.append(os.path.join(r, file))
    try:
        latest_file = max(list_of_files, key=os.path.getctime)
    except :
        print (f"there are no models already saved in the subfolders of path {path}")
        sys.exit(1)
    return(latest_file)

def set_saving_dir(path_previous_model, pretrained_model, data_path):
    if not path_previous_model:
        parent_path = "data/results"
        data_number = re.sub(r"\D", "", data_path)    
        dir_name = time.strftime("%Y%m%d_%H%M%S") + "_" + \
            pretrained_model + "_" + data_number
        path = os.path.join(parent_path, dir_name)
        path_int = os.path.join(path, "intermediate")
        path_img = os.path.join(path, "img")

        create_dir(path)
        create_dir(path_int)
        create_dir(path_img)
    else:
        path = os.path.dirname(os.path.dirname(path_previous_model))
        path_int = os.path.join(path, "intermediate")
        path_img = os.path.join(path, "img")

        if not os.path.isdir(path_int):
            create_dir(path_int)
        if not os.path.isdir(path_img):
            create_dir(path_img)
    return path

def create_dir(path_dir):
    try:
        os.mkdir(path_dir)
    except OSError:
        print (f"Creation of the directory {path_dir} failed")
        sys.exit(1)
    else:
        print (f"Successfully created the directory {path_dir} ")

if __name__ == "__main__":
    path_last_model = get_path_last_model()
    print(path_last_model)

    saving_dir = set_saving_dir("data/results/20200322_124708_bert-base-uncased_2009/intermediate/model_generate_conditionally_V2.pt", "bert-base-uncased", 'data/inputs/2009/dataframe_final_clean.csv')
    print(saving_dir)