from xml.dom import minidom


def pascal_anno_2_dict(anno_path):
    org_xml_dom = minidom.parse(anno_path)
    mid_data = dict()
    anno_size = org_xml_dom.getElementsByTagName('size')[0]
    mid_data['width'] = anno_size.getElementsByTagName('width')[0].childNodes[0].data
    mid_data['height'] = anno_size.getElementsByTagName('height')[0].childNodes[0].data
    mid_data['depth'] = anno_size.getElementsByTagName('depth')[0].childNodes[0].data
    mid_data['filename'] = org_xml_dom.getElementsByTagName('filename')[0].childNodes[0].data
    anno_objects = org_xml_dom.getElementsByTagName('object')
    mid_objects = []
    for anno_object in anno_objects:
        mid_object = dict()
        mid_object['xmin'] = anno_object.getElementsByTagName('xmin')[0].childNodes[0].data
        mid_object['ymin'] = anno_object.getElementsByTagName('ymin')[0].childNodes[0].data
        mid_object['xmax'] = anno_object.getElementsByTagName('xmax')[0].childNodes[0].data
        mid_object['ymax'] = anno_object.getElementsByTagName('ymax')[0].childNodes[0].data
        object_label = anno_object.getElementsByTagName('name')[0].childNodes[0].data
        mid_object['name'] = object_label
        mid_object['pose'] = anno_object.getElementsByTagName('pose')[0].childNodes[0].data
        mid_object['truncated'] = anno_object.getElementsByTagName('truncated')[0].childNodes[0].data
        mid_object['difficult'] = anno_object.getElementsByTagName('difficult')[0].childNodes[0].data
        mid_objects.append(mid_object)
    mid_data['objects'] = mid_objects
    return mid_data