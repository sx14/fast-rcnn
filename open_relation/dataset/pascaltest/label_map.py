import json
import data_config


def generate_label2wn():
    label2wn_dict = dict()
    label2wn_dict['aeroplane'] = ['airplane.n.01']
    label2wn_dict['bicycle'] = ['bicycle.n.01']
    label2wn_dict['bird'] = ['bird.n.01']
    label2wn_dict['boat'] = ['boat.n.01']
    label2wn_dict['bottle'] = ['bottle.n.01']
    label2wn_dict['bus'] = ['bus.n.01']
    label2wn_dict['car'] = ['car.n.01']
    label2wn_dict['cat'] = ['cat.n.01']
    label2wn_dict['chair'] = ['chair.n.01']
    label2wn_dict['cow'] = ['cow.n.01']
    label2wn_dict['diningtable'] = ['dining_table.n.01']
    label2wn_dict['dog'] = ['dog.n.01']
    label2wn_dict['horse'] = ['horse.n.01']
    label2wn_dict['motorbike'] = ['motorcycle.n.01']
    label2wn_dict['person'] = ['person.n.01']
    label2wn_dict['pottedplant'] = ['plant.n.02']
    label2wn_dict['sheep'] = ['sheep.n.01']
    label2wn_dict['sofa'] = ['sofa.n.01']
    label2wn_dict['train'] = ['train.n.01']
    label2wn_dict['tvmonitor'] = ['television_monitor.n.01']
    return label2wn_dict


if __name__ == '__main__':
    label2wn = generate_label2wn()
    with open(data_config.LABEL2WN_PATH, 'w') as label2wn_file:
        json.dump(label2wn, label2wn_file)
