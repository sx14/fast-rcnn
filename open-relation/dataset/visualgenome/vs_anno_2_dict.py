import json


def vs_anno_2_dict(anno_file_path):
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
            obj['name'] = o['names'][0]
            obj['pose'] = 'left'
            obj['truncated'] = '0'
            obj['difficult'] = '0'
            mid_objects.append(obj)
        mid_data['objects'] = mid_objects
    return mid_data