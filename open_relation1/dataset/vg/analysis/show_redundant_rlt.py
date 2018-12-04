import os
import json

anno_root = '/media/sunx/Data/dataset/visual genome/anno'
output_path = '/media/sunx/Data/dataset/visual genome/analysis/redundant_rlts.txt'
anno_list = os.listdir(anno_root)
# anno_sum = len(anno_list)
anno_sum = 1000
output_file = open(output_path, 'w')
output_file.close()
output_file = open(output_path, 'a')
for i in range(0, anno_sum):
    anno_path = os.path.join(anno_root, anno_list[i])
    anno = json.load(open(anno_path, 'r'))
    objId2name = dict()
    obj_annos = anno['objects']
    for obj in obj_annos:
        oid = obj['object_id']
        oname = obj['names'][0]
        objId2name[oid] = oname
    rlt_dict = dict()
    rlt_annos = anno['relationships']
    for rlt in rlt_annos:
        sid = rlt['subject']['object_id']
        oid = rlt['object']['object_id']
        predicate = rlt['predicate']
        if sid in rlt_dict:
            obj_rlts = rlt_dict[sid]
            if oid in obj_rlts:
                rlts = obj_rlts[oid]
            else:
                rlts = list()
            # outer -> inner
            if (0, predicate) not in rlts:
                rlts.append((0, predicate))
            obj_rlts[oid] = rlts
        elif oid in rlt_dict:
            obj_rlts = rlt_dict[oid]
            if sid in obj_rlts:
                rlts = obj_rlts[sid]
            else:
                rlts = list()
            # inner -> outer
            if (1, predicate) not in rlts:
                rlts.append((1, predicate))
        else:
            obj_rlts = dict()
            obj_rlts[oid] = [(0, predicate)]
            rlt_dict[sid] = obj_rlts
    output_lines = []
    output_lines.append('---- '+anno_list[i]+' ----\n')
    for o1 in rlt_dict:
        o1_name = objId2name[o1]
        o2s = rlt_dict[o1]
        for o2 in o2s:
            o2_name = objId2name[o2]
            rlts = o2s[o2]
            if len(rlts) > 1:
                for rlt in rlts:
                    if rlt[0] == 0:
                        record = o1_name + ' | ' + rlt[1] + ' | ' + o2_name + '\n'
                    else:
                        record = o2_name + ' | ' + rlt[1] + ' | ' + o1_name + '\n'
                    output_lines.append(record)
    output_file.writelines(output_lines)
output_file.close()