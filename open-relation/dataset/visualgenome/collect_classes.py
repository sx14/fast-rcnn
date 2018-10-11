import os
import json
import data_config
import wash_relation_wn



def vs_label2wn_leaf(anno_root, label2wn_path, target_key, stub_wn):
    label2wn = dict()  # vs_object_label: [wn]
    anno_list = os.listdir(anno_root)
    anno_total = len(anno_list)
    for i in range(0, anno_total):  # collect
        anno_file_name = anno_list[i]
        print('processing[%d/%d] : %s' % (anno_total, (i+1), anno_file_name))
        anno_path = os.path.join(anno_root, anno_file_name)
        with open(anno_path, 'r') as anno_file:
            anno = json.load(anno_file)
        objects = anno[target_key]
        for o in objects:
            if target_key == 'objects':
                o_name = o['names'][0]
            elif target_key == 'relationships':
                o_name = o['predicate']
            else:
                print('target_key is expected to be "objects" or "relationships"')
                return
            o_synsets = o['synsets']
            if o_name not in label2wn:
                label2wn[o_name] = dict()
                for s in o_synsets:
                    label2wn[o_name][s] = 1
            else:
                existed_syns = label2wn[o_name]
                for s in o_synsets:  # time statistic
                    if s in existed_syns:
                        label2wn[o_name][s] += 1
                    else:
                        label2wn[o_name][s] = 1
    for n in label2wn:
        syns = label2wn[n]
        if target_key == 'objects':
            max_times = 0
            max_time_syn = stub_wn  # for objects with no wn synset label
            for s in syns:
                if syns[s] > max_times:
                    max_times = syns[s]
                    max_time_syn = s
            label2wn[n] = [max_time_syn]   # 1 - [1]
        elif target_key == 'relationships':
            label2wn[n] = syns.keys()      # 1 - [n]
        else:
            print('target_key is expected to be "objects" or "relationships"')
            return
    with open(label2wn_path, 'w') as out:
        json.dump(label2wn, out, sort_keys=False, indent=4)


if __name__ == '__main__':
    org_anno_root = os.path.join(data_config.VS_ROOT, 'anno')
    washed_anno_root = os.path.join(data_config.VS_ROOT, 'washed_anno')
    obj_output_path = os.path.join(data_config.VS_ROOT, 'feature', 'object', 'prepare', 'label2wn.json')
    rlt_output_path = os.path.join(data_config.VS_ROOT, 'feature', 'relation', 'prepare', 'label2wn.json')
    # object
    stub_wn_object = 'entity.n.01'
    # vs_label2wn_leaf(org_anno_root, obj_output_path, 'objects', stub_wn_object)
    # relation
    wash_relation_wn.wash_relation_label(org_anno_root, washed_anno_root)
    # vs_label2wn_leaf(washed_anno_root, rlt_output_path, 'relationships', None)
    # wash_relation_wn.wash_relation_wn(rlt_output_path)

