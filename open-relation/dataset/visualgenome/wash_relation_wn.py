#coding=utf-8
import os
import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn


def wash_relation_wn(relation_label2wn_path):
    legal_pos_list = ['IN', 'JJ', 'JJR', 'JJS', 'RP', 'TO',     # 介词，形容词，比较级，最高级，虚词，to
                      'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']  # 动词，过去式，现在分词，过去分词，现在非三单，现在三单
    with open(relation_label2wn_path, 'r') as label2wn_file:
        label2wn = json.load(label2wn_file)
    lemmatizer = WordNetLemmatizer()
    wn_stub = 'relation.x.01'
    for label in label2wn.keys():
        wns = label2wn[label]
        if len(wns) > 0:
            continue
        label = label.lower()  # to lower case
        words = nltk.word_tokenize(label)  # split by spaces
        word_pos_list = nltk.pos_tag(words)  # [(word, pos)]
        for word_pos in word_pos_list:
            pos = word_pos[1]
            word = word_pos[0]
            if pos in legal_pos_list:
                org_word = lemmatizer.lemmatize(word)  # reshape word to original
                synsets = wn.synsets(org_word)
                if len(synsets) > 0:
                    wns.append(synsets[0])
        if len(wns) == 0:
            wns.append(wn_stub)
        label2wn[label] = wns
    with open(relation_label2wn_path, 'w') as label2wn_file:
        json.dump(label2wn, label2wn_file)




