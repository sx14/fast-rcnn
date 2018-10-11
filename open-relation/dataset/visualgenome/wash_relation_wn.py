#coding=utf-8
import os
import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

legal_pos_list = ['IN', 'JJR', 'JJS', 'RP', 'TO',           # 介词，比较级，最高级，虚词，to
                  'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']  # 动词，过去式，现在分词，过去分词，现在非三单，现在三单


def wash_relation_label(org_anno_root, output_anno_root):
    anno_list = os.listdir(org_anno_root)
    anno_total = len(anno_list)
    lemmatizer = WordNetLemmatizer()
    for i in range(0, anno_total):  # collect
        anno_file_name = anno_list[i]
        print('washing[%d/%d] : %s' % (anno_total, (i + 1), anno_file_name))
        org_anno_path = os.path.join(org_anno_root, anno_file_name)
        with open(org_anno_path, 'r') as anno_file:
            anno = json.load(anno_file)
        relations = anno['relationships']
        for i in range(0, len(relations)):
            r = relations[i]
            predicate = r['predicate']
            new_label_words = []
            label_lower = predicate.lower()
            words = nltk.word_tokenize(label_lower)  # split by spaces
            word_pos_list = nltk.pos_tag(words)  # [(word, pos)]
            for word_pos in word_pos_list:
                word = word_pos[0]
                pos = word_pos[1]
                if pos in legal_pos_list:   # legal predicate words
                    if pos.startswith('VB'):
                        org_word = lemmatizer.lemmatize(word, pos='v')  # reshape word to original
                    else:
                        org_word = word
                    new_label_words.append(org_word)
            # merge word list to new predicate
            new_predicate = ' '.join(new_label_words)
            relations[i]['predicate'] = new_predicate
        anno['relations'] = relations
        output_anno_path = os.path.join(output_anno_root, anno_file_name)
        with open(output_anno_path, 'w') as anno_file:
            json.dump(anno, anno_file, sort_keys=False, indent=4)


def wash_relation_wn(relation_label2wn_path):
    """
    run after wash_relation_label
    try to supplement WordNet node
    :param relation_label2wn_path:
    :return:
    """
    with open(relation_label2wn_path, 'r') as label2wn_file:
        label2wn = json.load(label2wn_file)
    wn_stub = 'relation.x.01'
    for label in label2wn.keys():
        wns = label2wn[label]
        if len(''.join(wns)) > 0:
            continue
        wns = []
        words = nltk.word_tokenize(label)  # split by spaces
        word_pos_list = nltk.pos_tag(words)  # [(word, pos)]
        for word_pos in word_pos_list:
            pos = word_pos[1]
            word = word_pos[0]
            if pos in legal_pos_list:
                synsets = wn.synsets(word)
                if len(synsets) > 0:
                    wns.append(synsets[0].name())
        if len(wns) == 0:
            wns.append(wn_stub)
        label2wn[label] = wns
    with open(relation_label2wn_path, 'w') as label2wn_file:
        json.dump(label2wn, label2wn_file)




