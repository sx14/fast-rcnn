import os
import json
import pickle
from open_relation1.vg_data_config import vg_config, vg_object_config, vg_relation_config


def vglabel2wnleaf(anno_root, vg2wn_path, target_key):
    vg2wn = dict()  # vg_label: [wn]
    anno_list = os.listdir(anno_root)
    anno_total = len(anno_list)
    for i in range(0, anno_total):  # collect
        anno_file_name = anno_list[i]
        print('processing vg2wn [%d/%d]' % (anno_total, (i+1)))
        anno_path = os.path.join(anno_root, anno_file_name)
        with open(anno_path, 'r') as anno_file:
            anno = json.load(anno_file)
        items = anno[target_key]
        for item in items:
            if target_key == 'objects':
                item_name = item['name']
            elif target_key == 'relationships':
                item_name = item['predicate']
            else:
                print('target_key is expected to be "objects" or "relationships"')
                return
            item_synsets = item['synsets']
            if item_name not in vg2wn:
                vg2wn[item_name] = dict()
                for syn in item_synsets:
                    vg2wn[item_name][syn] = 1
            else:
                existed_syns = vg2wn[item_name]
                for syn in item_synsets:  # count
                    if syn in existed_syns:
                        vg2wn[item_name][syn] += 1
                    else:
                        vg2wn[item_name][syn] = 1
    for vg_label in vg2wn:
        syn_counter = vg2wn[vg_label]
        if target_key == 'objects':
            max_time_syn = syn_counter.keys()[0]
            max_times = syn_counter[max_time_syn]
            for syn in syn_counter:
                if syn_counter[syn] > max_times:
                    max_times = syn_counter[syn]
                    max_time_syn = syn
            vg2wn[vg_label] = [max_time_syn]        # 1 - [1]
        elif target_key == 'relationships':
            vg2wn[vg_label] = syn_counter.keys()    # 1 - [n]
        else:
            print('target_key is expected to be "objects" or "relationships"')
            return
    pickle.dump(vg2wn, open(vg2wn_path, 'wb'))


if __name__ == '__main__':
    anno_root = vg_config['clean_anno_root']
    obj_vg2wn_path = vg_object_config['vg2wn_path']
    rlt_vg2wn_path = vg_relation_config['vg2wn_path']
    # object
    wn_obj_stub = 'entity.n.01'
    vglabel2wnleaf(anno_root, obj_vg2wn_path, 'objects')
    # relation
    # vglabel2wnleaf(anno_root, rlt_vg2wn_path, 'relationships')


