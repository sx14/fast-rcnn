from nltk.corpus import wordnet as wn
import os
import json
from matplotlib import pyplot as plt


def get_distribution(anno_root, wn_type, total):
    # get the distribution of the synset depth
    a = []
    counter = 0
    for anno_file_name in os.listdir(anno_root):
        if counter > total:
            break
        counter = counter + 1
        print('processing[%d/%d] : %s\n' % (total, counter, anno_file_name))
        anno_file_path = os.path.join(anno_root, anno_file_name)
        with open(anno_file_path) as anno_file:
            anno = json.load(anno_file)
        objects = anno[wn_type][wn_type]
        for o in objects:
            o_synsets = o['synsets']
            for w in o_synsets:
                wn_node = wn.synset(w)
                depth = wn_node.max_depth()
                a.append(depth)
    plt.hist(a, 10)
    plt.show()


def collect_all_relationships(anno_root, total):
    relation_bags = dict()     # wordnet_node -> VS_relations
    wn_relation_set = set()    # wordnet synset collection
    relation_counter = dict()  # VS_relation -> appearance times of current VS_relation
    counter = 0
    anno_total = len(os.listdir(anno_root))
    for anno_file_name in os.listdir(anno_root):
        counter = counter + 1
        if counter > total:
            break
        print('processing[%d/%d] : %s\n' % (min(total, anno_total), counter, anno_file_name))
        anno_file_path = os.path.join(anno_root, anno_file_name)
        with open(anno_file_path) as anno_file:
            anno = json.load(anno_file)
        objects = anno['relationships']['relationships']
        for o in objects:
            relation = o['predicate']
            if relation not in relation_counter.keys():
                relation_counter[relation] = 1
            else:
                relation_counter[relation] = relation_counter[relation] + 1
            wn_relations = o['synsets']
            for wr in wn_relations:
                wn_relation_set.add(wr)
                if wr not in relation_bags.keys():
                    relation_bag = set()
                    # relation_bag = dict()
                    # relation_bag[relation] = o['subject']['name'] + ' - ' + o['object']['name']
                    relation_bag.add(relation)
                    relation_bags[wr] = relation_bag
                else:
                    relation_bag = relation_bags[wr]
                    # if relation not in relation_bag.keys():
                    #     relation_bag[relation] = o['subject']['name'] + ' - ' + o['object']['name']
                    relation_bag.add(relation)
                    relation_bags[wr] = relation_bag
    return relation_counter, wn_relation_set, relation_bags


def convert_relation_to_wn(anno_root, total):
    lines = []  # (VS_subject, VS_relation, VS_object) -> (WN_subject, WN_relation, WN_object)
    counter = 0
    anno_total = len(os.listdir(anno_root))
    for anno_file_name in os.listdir(anno_root):
        counter = counter + 1
        if counter > total:
            break
        print('processing[%d/%d] : %s\n' % (min(total, anno_total), counter, anno_file_name))
        anno_file_path = os.path.join(anno_root, anno_file_name)
        with open(anno_file_path) as anno_file:
            anno = json.load(anno_file)
        objects = anno['relationships']['relationships']
        for o in objects:
            o_synsets = o['synsets']
            for w in o_synsets:
                relation = o['predicate']
                object_name = o['object']['name']
                subject_name = o['subject']['name']
                wn_relation = w.split('.')[0]
                if len(o['object']['synsets']) > 0:
                    wn_object = o['object']['synsets'][0].split('.')[0]
                else:
                    wn_object = object_name
                if len(o['subject']['synsets']) > 0:
                    wn_subject = o['subject']['synsets'][0].split('.')[0]
                else:
                    wn_subject = subject_name
                relationship = '{} {} {} -> {} {} {}'.format(subject_name, relation, object_name, wn_subject, wn_relation, wn_object)
                lines.append(relationship)
    return lines


def export_json(relation_bags, json_file_path):
    relation_dict = dict()
    for wr in relation_bags.keys():
        wr_dict = dict()
        relations = relation_bags[wr]
        # relation_lines = []
        # for relation in relations.keys():
        #     relation_line = relation + ' : ' + relations[relation]
        #     relation_lines.append(relation_line)
        # wr_dict['relations'] = relation_lines
        wr_dict['relations'] = list(relations)
        wr_dict['definition'] = wn.synset(wr).definition()
        relation_dict[wr] = wr_dict
    with open(json_file_path, 'w') as json_file:
        json.dump(relation_dict, json_file, indent=4)


def export_txt(relation_counter, txt_file_path):
    lines = []
    relation_counter = sorted(relation_counter.items(), key=lambda d:d[1], reverse=True)
    with open(txt_file_path, 'w') as txt_file:
        for relation in relation_counter:
            line = relation[0] + '\t\t' + str(relation[1]) + '\n'
            lines.append(line)
        txt_file.writelines(lines)


def find_redundant_relationship(anno_root):
    output_file = open('redundant_relationship.txt', 'a')
    for anno_name in os.listdir(anno_root):
        img_info = '----'+anno_name+'----'
        print(img_info)
        output_file.write(img_info+'\n')
        anno_path = os.path.join(anno_root, anno_name)
        anno = json.load(open(anno_path, 'r'))
        sbj2obj = dict()
        sbjobj2relation = dict()
        relationships = anno['relationships']
        for relationship in relationships:
            sbj = relationship['subject']
            sbj_id = str(sbj['object_id'])
            obj = relationship['object']
            obj_id = str(obj['object_id'])
            if sbj_id in sbj2obj:
                if sbj2obj[sbj_id] == obj_id:
                    sbjobj_key = sbj_id+'-'+obj_id
                    existed_relation = sbjobj2relation[sbjobj_key]
                    if existed_relation != relationship['predicate']:
                        relationship1_str = sbj['name']+' | '+relationship['predicate']+' | '+obj['name']
                        relationship2_str = sbj['name']+' | '+existed_relation+' | '+obj['name']
                        print(relationship1_str)
                        output_file.write(relationship1_str + '\n')
                        print(relationship2_str)
                        output_file.write(relationship2_str + '\n')
            elif obj_id in sbj2obj:
                if sbj2obj[obj_id] == sbj_id:
                    sbjobj_key = obj_id + '-' + sbj_id
                    existed_relation = sbjobj2relation[sbjobj_key]
                    relationship1_str = sbj['name']+' | '+relationship['predicate']+' | '+obj['name']
                    relationship2_str = obj['name']+' | '+existed_relation+' | '+sbj['name']
                    print(relationship1_str)
                    output_file.write(relationship1_str + '\n')
                    print(relationship2_str)
                    output_file.write(relationship2_str + '\n')
            else:
                sbj2obj[sbj_id] = obj_id
                sbjobj_key = sbj_id+'-'+obj_id
                sbjobj2relation[sbjobj_key] = relationship['predicate']










if __name__  == '__main__':
    anno_root = '/media/sunx/Data/dataset/visual genome/anno'
    # total = 1000
    # get_distribution(anno_root, 'objects', total)
    # get_distribution(anno_root, 'relationships', total)
    # lines = convert_relation_to_wn(anno_root, 10)
    # relation_set = collect_all_relationships(anno_root, 10000)
    # relation_counter, wn_relation_set, relation_bags = collect_all_relationships(anno_root, 100000000)
    # relation_bag_json_path = '/media/sunx/Data/dataset/visual genome/relation_bag3.json'
    # export_json(relation_bags, relation_bag_json_path)
    # relation_counter_txt_path = '/media/sunx/Data/dataset/visual genome/relation_counter.txt'
    # export_txt(relation_counter, relation_counter_txt_path)

    find_redundant_relationship(anno_root)

