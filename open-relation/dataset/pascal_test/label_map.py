import json


def _label2wn():
    label2wn_dict = dict()
    label2wn_dict['aeroplane'] = 'airplane.n.01'
    label2wn_dict['bicycle'] = 'bicycle.n.01'
    label2wn_dict['bird'] = 'bird.n.01'
    label2wn_dict['boat'] = 'boat.n.01'
    label2wn_dict['bottle'] = 'bottle.n.01'
    label2wn_dict['bus'] = 'bus.n.01'
    label2wn_dict['car'] = 'car.n.01'
    label2wn_dict['cat'] = 'cat.n.01'
    label2wn_dict['chair'] = 'chair.n.01'
    label2wn_dict['cow'] = 'cow.n.01'
    label2wn_dict['diningtable'] = 'dining_table.n.01'
    label2wn_dict['dog'] = 'dog.n.01'
    label2wn_dict['horse'] = 'horse.n.01'
    label2wn_dict['motorbike'] = 'motorcycle.n.01'
    label2wn_dict['person'] = 'person.n.01'
    label2wn_dict['pottedplant'] = 'plant.n.02'
    label2wn_dict['sheep'] = 'sheep.n.01'
    label2wn_dict['sofa'] = 'sofa.n.01'
    label2wn_dict['train'] = 'train.n.01'
    label2wn_dict['tvmonitor'] = 'television_monitor.n.01'
    return label2wn_dict


def label2wn_index(wn_synset_file_path):
    label2wn = _label2wn()
    with open(wn_synset_file_path, 'r') as wn_synset_file:
        wn_synsets = json.load(wn_synset_file)
    wn_synset_index = dict()
    for i in range(0, len(wn_synsets)):
        wn_synset_index[wn_synsets[i]] = i
    label2wn_index_dict = dict()
    for label in label2wn.keys():
        synset = label2wn[label]
        synset_index = wn_synset_index[synset]
        label2wn_index_dict[label] = synset_index
    return label2wn_index_dict