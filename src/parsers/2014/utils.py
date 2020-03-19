import os
import string
import xml.etree.ElementTree as ET
import re
import pandas as pd


def clean_word(word):
    s = "#/-_ \n\t,."
    w = re.sub(r'[.;$&]|#[0-9]{4}', '', word)
    w = w.rstrip(s)
    return w.lstrip(s)


def extract_words(text, tags, text_idx):
    tag = 'O'
    body = {
        'start': 0,
        'end': 0
    }
    # we need to have unique sentences id, hypothesis is that there is less than 1000 sentences per text
    current_sent = 1000 * text_idx
    if tags:
        tag, body = tags.pop(0)
    words = []
    word = ''
    for index, char in enumerate(text):

        if char == ' ' or char == '\n':
            if word:
                if index > int(body['start']) and index <= (int(body['end']) + 1):
                    if clean_word(word):
                        words.append((tag, clean_word(word), current_sent))
                else:
                    if clean_word(word):
                        words.append(('O', clean_word(word), current_sent))
            word = ''
        else:
            word += char

        if index > int(body['end']) and tags:
            tag, body = tags.pop(0)

        if char == '.':  # check end of sentence
            if index < len(text) - 1 and text[index+1] == ' ':
                current_sent += 1

    return words


def extract_tags_risk(path):
    tree = ET.parse(path)
    root = tree.getroot()

    tags = []
    text = ''

    for child in root:
        if child.tag == 'TEXT':
            text = child.text

        if child.tag == 'TAGS':
            for child2 in child:
                if child2.tag == 'PHI':
                    tags.append((child2.tag, child2.attrib))
                else:
                    tag_child = []

                    def is_included(new_attrib):
                        inc = True
                        to_remove = []
                        if not 'start' in new_attrib:
                            return False
                        for i in range(len(tag_child)):
                            _, attr = tag_child[i]

                            if attr['start'] <= new_attrib['start'] and attr['end'] >= \
                                    new_attrib['end']:
                                inc = False
                            elif attr['start'] >= new_attrib['start'] and attr['end'] <= \
                                    new_attrib['end']:
                                to_remove.append(i)

                        for i in sorted(to_remove, reverse=True):
                            del tag_child[i]

                        return inc

                    for child3 in child2:
                        ta, attrib = child3.tag, child3.attrib
                        if is_included(attrib):
                            tag_child.append((ta, attrib))

                tags += tag_child

    tags = sorted(tags, key=lambda x: int(x[1]['start']))
    return tags, text


def extract_tags_phi(path):
    tree = ET.parse(path)
    root = tree.getroot()
    text = ''
    tags = []

    for child in root:
        # print(child.tag)
        if child.tag == 'TEXT':
            text = child.text
            # print(len(text))
            # print(text[4680:4685])
            l = text.splitlines()

        if child.tag == 'TAGS':
            for tag in child:
                tags.append((tag.tag, tag.attrib))
    return tags, text


def map_output(words):
    sent = []
    word = []
    tag = []
    for i, (t, w, s) in enumerate(words):
        word.append(w)
        sent.append(s)

        if t == 'O':  # no tag case
            tag.append(t)
        elif tag:
            t0 = tag[-1].split('-')[-1]
            if t0 != t:
                tag.append('B-'+t)  # beginning of tag
            else:
                if i >= len(words) - 1:  # end of text
                    tag.append('O-'+t)
                else:
                    t_next, _, _ = words[i+1]
                    if t_next == t:
                        tag.append('I-'+t)
                    else:
                        tag.append('O-'+t)

        else:
            tag.append('B-'+t)

    return pd.DataFrame.from_dict(
        {
            'sentence': sent,
            'word': word,
            'tag': tag
        }
    )


def save_to_csv(path, filename, df):
    p = os.path.join(path, filename)
    df.to_csv(p, encoding='utf-8')