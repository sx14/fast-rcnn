import json


def vg_anno_2_dict(anno_file_path):
    mid_data = dict()
    with open(anno_file_path, 'r') as anno_file:
        anno = json.load(anno_file)
        mid_data['filename'] = str(anno['image_info']['image_id']) + '.jpg'
        mid_data['width'] = str(anno['image_info']['width'])
        mid_data['height'] = str(anno['image_info']['height'])
        mid_data['depth'] = '3'
        anno_objects = anno['objects']
        mid_objects = []
        for o in anno_objects:
            obj = dict()
            obj['xmin'] = str(o['x'])
            obj['ymin'] = str(o['y'])
            obj['xmax'] = str(int(o['x']) + int(o['w']) - 1)
            obj['ymax'] = str(int(o['y']) + int(o['h']) - 1)
            obj['name'] = o['name']
            obj['pose'] = 'left'
            obj['truncated'] = '0'
            obj['difficult'] = '0'
            obj['synsets'] = o['synsets']
            mid_objects.append(obj)
        mid_data['objects'] = mid_objects
        anno_relationships = anno['relationships']
        mid_relationships = []
        for r in anno_relationships:
            mid_relationship = dict()
            mid_relationship['predicate'] = r['predicate']
            mid_subject = dict()
            mid_object = dict()
            subject = r['subject']
            object = r['object']
            mid_subject['xmin'] = str(subject['x'])
            mid_subject['ymin'] = str(subject['y'])
            mid_subject['xmax'] = str(int(subject['x']) + int(subject['w']) - 1)
            mid_subject['ymax'] = str(int(subject['y']) + int(subject['h']) - 1)
            mid_object['xmin'] = str(object['x'])
            mid_object['ymin'] = str(object['y'])
            mid_object['xmax'] = str(int(object['x']) + int(object['w']) - 1)
            mid_object['ymax'] = str(int(object['y']) + int(object['h']) - 1)
            mid_relationship['subject'] = mid_subject
            mid_relationship['object'] = mid_object
            mid_relationships.append(mid_relationship)
        mid_data['relationships'] = mid_relationships
    return mid_data