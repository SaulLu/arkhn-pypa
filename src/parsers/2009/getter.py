
import os
import re

def get_annotations_1(folder_name):
    ids_annotations_1 = []
    corpus_annotations_1 = []

    sub_folder_tag = 'annotations'
    sub_path_tag = os.path.join(folder_name,sub_folder_tag)

    for doc_name in os.listdir(sub_path_tag):
        if doc_name[0] != ".":  # ignore hidden files
            id = re.sub('[^0-9]','', doc_name)
            ids_annotations_1.append(int(id))
            path = os.path.join(sub_path_tag,doc_name)
            with open(path) as f:
                content = f.read().splitlines()
                corpus_annotations_1.append(content)
    return ids_annotations_1, corpus_annotations_1

def get_annotations_2(folder_name):
    ids_annotations_2 = []
    corpus_annotations_2 = []

    sub_folder_val = 'annotations_val_test'
    sub_path_val = os.path.join(folder_name,sub_folder_val)

    for doc_name in os.listdir(sub_path_val):
        if doc_name[0] != ".":  # ignore hidden files
            id = re.split('\.', doc_name)[0]
            ids_annotations_2.append(int(id))
            path = os.path.join(sub_path_val,doc_name)
            with open(path) as f:
                content = f.read().splitlines()
                corpus_annotations_2.append(content)
    return ids_annotations_2, corpus_annotations_2

def get_records(folder_name):
    ids_records = []
    corpus_records = []
    sub_folder_rec = 'records'
    sub_path_dir_rec = os.path.join(folder_name,sub_folder_rec)
    for dir_name in os.listdir(sub_path_dir_rec):
        if dir_name[0] != ".":
            sub_path_rec = os.path.join(sub_path_dir_rec,dir_name)
            for filename in os.listdir(sub_path_rec):
                if filename[0] != ".":
                    ids_records.append(int(filename))
                    path = os.path.join(sub_path_rec,filename)
                    with open(path) as f:
                        #content = f.readlines()
                        content = f.read().splitlines()
                        corpus_records.append(content)
    return ids_records, corpus_records