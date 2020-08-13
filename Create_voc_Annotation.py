from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import os

def make_xml(xmin_tuple, ymin_tuple, xmax_tuple, ymax_tuple, image_name):

    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'VOC'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_name + '.jpg'

    node_object_num = SubElement(node_root, 'object_num')
    node_object_num.text = str(len(xmin_tuple))

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '512'

    node_height = SubElement(node_size, 'height')
    node_height.text = '384'

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    for i in range(len(xmin_tuple)):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = 'ship'
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'

        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(xmin_tuple[i])
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(ymin_tuple[i])
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(xmax_tuple[i])
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(ymax_tuple[i])


    xml = tostring(node_root, pretty_print = True)
    dom = parseString(xml)
    print (xml) #打印查看结果
    return dom


dom = make_xml([1], [119], [27], [201], '000077.jpg')

xml_name = os.path.join('/home/tianhao.lu/code/CenterNet-45/data/NICE1/', '000077.jpg' + '.xml')
with open(xml_name, 'wb') as f:
    f.write(dom.toprettyxml(indent='\t', encoding='utf-8'))

# dom = make_xml(xmin_tuple, ymin_tuple, xmax_tuple, ymax_tuple, image_name)
#
# xml_name = os.path.join(save_xml_path, image_name + '.xml')
# with open(xml_name, 'w') as f:
#     f.write(dom.toprettyxml(indent='\t', encoding='utf-8'))