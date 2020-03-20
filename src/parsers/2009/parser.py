import os
from nltk import pos_tag, RegexpParser
import pandas as pd
import numpy as np
import re

from getter import get_annotations_1, get_annotations_2, get_records


def main():
    folder_name = 'data' #change the folder

    ids_annotations_1, corpus_annotations_1 = get_annotations_1(folder_name)
    ids_annotations_2, corpus_annotations_2 = get_annotations_2(folder_name)
    ids_records, corpus_records = get_records(folder_name)

    intersection = list(set(ids_annotations_1) & set(ids_records))
    if len(intersection) == len(ids_annotations_1):
        print("Success: all train annotations have a corresponding entry.", len(intersection))

    intersection = list(set(ids_annotations_2) & set(ids_records))
    if len(intersection) == len(ids_annotations_2):
        print("Success: all valid annotations have a corresponding entry.", len(intersection))

    df_records = extract_corpus_in_df(ids_records, corpus_records)

    get_annotations(df_records, corpus_annotations_1, ids_annotations_1, 'yes')
    get_annotations(df_records, corpus_annotations_2, ids_annotations_2, 'yes')

    sub_folder_df = 'pandas_dataframe'
    file_name = 'dataframe_final.pkl'
    sub_path_dir_df = os.path.join(folder_name,sub_folder_df, file_name)
    df_records.to_pickle(sub_path_dir_df)

    file_name = 'dataframe_final.csv'
    sub_path_dir_df = os.path.join(folder_name,sub_folder_df, file_name)
    df_records.to_csv(sub_path_dir_df)

    file_name = 'dataframe_final.txt'
    sub_path_dir_df = os.path.join(folder_name,sub_folder_df, file_name)
    np.savetxt(sub_path_dir_df, df_records.values, fmt="%s")

    df_formated = df_records[df_records['annotated']=='yes']

    sentences = df_formated['sentence'].to_list()
    sentences_bis = [0,0]
    offset = 0
    for i in range(2,len(sentences)):
        if sentences[i] == 0 and sentences[i-1]!=0:
            offset += sentences[i-1] + 1
        sentences_bis.append(sentences[i] + offset)
    df_formated['sentence'] = sentences_bis
    df_formated = df_formated.rename(columns={
        "NER_tag":"tag"
    })

    sub_folder_df = 'pandas_dataframe'
    file_name = 'dataframe_final_clean.pkl'
    sub_path_dir_df = os.path.join(folder_name,sub_folder_df, file_name)
    df_formated.to_pickle(sub_path_dir_df)

    file_name = 'dataframe_final_clean.csv'
    sub_path_dir_df = os.path.join(folder_name,sub_folder_df, file_name)
    df_formated.to_csv(sub_path_dir_df)

    file_name = 'dataframe_final_clean.txt'
    sub_path_dir_df = os.path.join(folder_name,sub_folder_df, file_name)
    np.savetxt(sub_path_dir_df, df_formated.values, fmt="%s")

def extract_corpus_in_df(ids_records, corpus_records):
    entries_cols = ["id", "sentence", "row", "offset", "word", 'NER_tag', 'annotated']
    df_records = pd.DataFrame(columns=entries_cols)

    df_records = pd.DataFrame(columns=entries_cols)  # reset df
    tmp_list = []

    for doc_i, document in enumerate(corpus_records):
        word_sentence = 0
        for row_i, row in enumerate(document):
            row = re.sub('\t', ' ', row)
            row_split = row.split(" ")
            for word_i, word in enumerate(row_split):
                if word != '...':
                    if re.search('\.|\?|\!', word):
                        word_sentence += 1
                    word = word.rstrip(".")  # strip "." from end of word
                word = word.replace("\t", "")
                word_id = ids_records[doc_i]
                word_row = row_i+1  # 1-based indexing 
                word_offset = word_i # 0-based indexing
                
                if len(word) > 0 and "|" not in word:
                    tmp_list.append([word_id, word_sentence, word_row, word_offset, word, 'O', 'no'])

    df_records = pd.DataFrame(tmp_list, columns=entries_cols)
    return df_records

def set_tag_row_col(df, row_indexer, tag):
    df.loc[row_indexer,"NER_tag"] = tag
    return(df.loc[row_indexer,"word"].values[0])

def get_tag_row_col(df, id, row, offset):
    row_indexer = df[(df['id'] == id ) & (df['row'] == row) & (df['offset'] == offset)].index.values.astype(int)
    return row_indexer

def set_status_record(df, id, yes_no):
    row_indexer = df[(df['id'] == id )].index.values.astype(int)
    df.loc[row_indexer,"annotated"] = yes_no

def get_annotations(df, corpus, ids_corpus, train_val_test):
    n = len(corpus)

    for i, document in enumerate(corpus):
        print(f"annotation : {i}/{n}")
        set_status_record(df, ids_corpus[i], train_val_test)
        for row in document:
            row = row.split("||")
            # print(row, "\n")
            for tag in row: 
                tag = tag.split("=")
                #print(f"tag : {tag}")
                if ":" in tag[1]:
                    tag_label = tag[0].lstrip(" ")
                    #print(f"tag_label : {tag_label}")
                    info = tag[1].split('"')
                    info[:] = [x for x in info if x]
                    #print(f"info : {info}")
                    tag_word = info[0].lstrip(" ")
                    if tag_word != '...':
                        tag_word = tag_word.rstrip(".")
                    text_word = ''
                    #print(f"tag_word : {tag_word}")
                    coords = re.split(' |,', info[1])
                    coords[:] = [x for x in coords if x]
                    #print(f"coords: {coords}")
                    if len(coords)%2 !=0:
                        print("pb")
                    else:
                        BIO_tag = 'B-'
                        #print(f"{len(coords)//2}")
                        for ind in range(len(coords)//2):
                            out1 = coords[ind*2].split(":")
                            row_1, offset_1 = [int(x) for x in out1]
                            # print(f"row_1: {row_1}")
                            # print(f"offset_1: {offset_1}")
                            out2 = coords[ind*2+1].split(":")
                            row_2, offset_2 = [int(x) for x in out2]
                            # print(f"row_2: {row_2}")
                            # print(f"offset_2: {offset_2}")
                            keep_tag = True
                            while keep_tag:
                                # print(f"row_1: {row_1}")
                                # print(f"offset_1: {offset_1}")
                                tag = BIO_tag + tag_label
                                row_indexer = get_tag_row_col(df, ids_corpus[i], row_1, offset_1)
                                if (row_1 == row_2) and (offset_1 == (offset_2+1)):
                                    if text_word.lower().lstrip(" ") != tag_word:
                                        print(f"\nprobleme de correspondance")
                                        print(f"text_word: {text_word.lower()}")
                                        print(f"tag_word: {tag_word}")

                                    keep_tag = False
                                    # print("\n")
                                elif row_indexer:
                                    text_word = text_word + ' ' + set_tag_row_col(df, row_indexer, tag)
                                    offset_1 += 1
                                else:
                                    row_1 += 1
                                    offset_1 = 0
                                if offset_1 > 300:
                                    print("There is a problem, check the corresponding annotation and record files")
                                    print(f"tag_word: {tag_word}")
                                    break
                                
                                BIO_tag = 'I-'


