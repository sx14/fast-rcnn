import os
import vs_anno_2_dict as org
import to_pascal_format as out

if __name__ == '__main__':
    anno_root = '/media/sunx/Data/dataset/visual genome/anno'
    output_root = '/media/sunx/Data/dataset/visual genome/Annotations'
    for anno_file_name in os.listdir(anno_root):
        output_xml_file_name = anno_file_name.split('.')[0] + '.xml'
        mid_data = org.vs_anno_2_dict(os.path.join(anno_root, anno_file_name))
        out.output_pascal_format(mid_data, os.path.join(output_root, output_xml_file_name))
