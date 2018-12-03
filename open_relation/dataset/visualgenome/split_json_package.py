import json
import os
import data_config

object_json_path = os.path.join(data_config.VS_ROOT, 'objects.json')
image_data_path = os.path.join(data_config.VS_ROOT, 'image_data.json')
relationship_path = os.path.join(data_config.VS_ROOT, 'relationships.json')
output_json_root = os.path.join(data_config.VS_ROOT, 'anno')


def split_json(json_path, output_json_root, key, has_key):
    with open(json_path, 'r') as data_file:
        image_data = json.load(data_file)
    for i in range(0, len(image_data)):
        print('processing ' + key + ' : ' + str(i))
        img_info = image_data[i]
        img_json_file_path = os.path.join(output_json_root, str(img_info['image_id']) + '.json')
        if os.path.exists(img_json_file_path):
            img_json_file = open(img_json_file_path, 'r')
            img_json_content = json.load(img_json_file)
        else:
            f = open(img_json_file_path, 'w')
            f.close()
            img_json_content = dict()
        if key not in img_json_content:
            if has_key:
            	img_json_content[key] = img_info[key]
            else:
                img_json_content[key] = img_info
        with open(img_json_file_path, 'w') as img_json_file:
            json.dump(img_json_content, img_json_file, sort_keys=False, indent=4)




if __name__ == '__main__':
    split_json(image_data_path, output_json_root, u'image_info', False)
    split_json(object_json_path, output_json_root, u'objects', True)
    split_json(relationship_path, output_json_root, u'relationships', True)

