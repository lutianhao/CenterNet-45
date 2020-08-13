#我们需要读取图片的名称，尺寸，然后读取相应名称的ini文件，然后把对应的变量按照xml格式存入生成对应的xml文件
import glob
import os
import matplotlib.pyplot as plt
import cv2
import configparser
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString

def make_xml(iris_exist, iris_centerX, iris_centerY, iris_radius, pupil_exist, pupil_centerX, pupil_centerY, pupil_radius, image_name, width, height, depth):

    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'NICE1'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_name

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(width)

    node_height = SubElement(node_size, 'height')
    node_height.text = str(height)

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = str(depth)

    # iris object
    node_object = SubElement(node_root, 'object')
    node_name = SubElement(node_object, 'name')
    node_name.text = 'iris'
    node_difficult = SubElement(node_object, 'difficult')
    node_difficult.text = '0'
    node_iris_exist = SubElement(node_object, 'exist')
    node_iris_exist.text = str(iris_exist)
    if iris_exist == 'true':
        node_bndbox = SubElement(node_object, 'bndbox')
        node_iris_centerX = SubElement(node_bndbox, 'iris_centerX')
        node_iris_centerX.text = str(iris_centerX)
        node_iris_centerY = SubElement(node_bndbox, 'iris_centerY')
        node_iris_centerY.text = str(iris_centerY)
        node_iris_radius = SubElement(node_bndbox, 'iris_radius')
        node_iris_radius.text = str(iris_radius)

    # pupil object
    node_object = SubElement(node_root, 'object')
    node_name = SubElement(node_object, 'name')
    node_name.text = 'pupil'
    node_difficult = SubElement(node_object, 'difficult')
    node_difficult.text = '0'
    node_pupil_exist = SubElement(node_object, 'exist')
    node_pupil_exist.text = str(pupil_exist)
    if pupil_exist == 'true' :
        node_bndbox = SubElement(node_object, 'bndbox')
        node_pupil_centerX = SubElement(node_bndbox, 'pupil_centerX')
        node_pupil_centerX.text = str(pupil_centerX)
        node_pupil_centerY = SubElement(node_bndbox, 'pupil_centerY')
        node_pupil_centerY.text = str(pupil_centerY)
        node_pupil_radius = SubElement(node_bndbox, 'pupil_radius')
        node_pupil_radius.text = str(pupil_radius)

    xml = tostring(node_root, pretty_print = True)
    dom = parseString(xml)
    # print (xml) #打印查看结果
    return dom



Raw_Pic_path = '/home/tianhao.lu/code/CenterNet-45/data/NICE1/train/image/'  #存放train原始图片的路径
# Raw_Pic_path = '/home/tianhao.lu/code/CenterNet-45/data/NICE1/test/image/'  #存放test原始图片的路径
Ini_path = '/home/tianhao.lu/code/CenterNet-45/data/NICE1/train/circle_params/'  #存放tarin的ini文件的路径
# Ini_path = '/home/tianhao.lu/code/CenterNet-45/data/NICE1/test/circle_params/'  #存放test的ini文件的路径
Save_path = '/home/tianhao.lu/code/CenterNet-45/data/NICE1/voc/train/Annotations/' #存放train的xml文件的路径
# Save_path = '/home/tianhao.lu/code/CenterNet-45/data/NICE1/voc/test/Annotations/' #存放的xml文件的路径
# #获取指定目录下的所有图片
paths = glob.glob(os.path.join(Raw_Pic_path,'*.JPEG'))
# # path.sort()  #排序
for path in paths:
    print('当前path路径为： {}'.format(path))
    filepath,tmpfilename= os.path.split(path) #分割文件名和路径
    filename,extension = os.path.splitext(tmpfilename) #分割文件名和后缀
    img = cv2.imread(path)
    tmpwidth = img.shape[1]  #图片宽度
    tmpheight = img.shape[0] #图片高度
    tmpdepth = img.shape[2] #图片深度

    #查找对应ini文件
    tmp_ini_paths = os.listdir(Ini_path)
    for tmp_ini_path in tmp_ini_paths:
        tmp_ini_filename = os.path.split(tmp_ini_path)[1]
        tmp_ini_filename = os.path.splitext(tmp_ini_filename)[0]
        if tmp_ini_filename == filename :    # 读取ini内容
            #  实例化configParser对象
            config = configparser.ConfigParser()
            # -read读取ini文件
            print('当前ini文件是：{}'.format(Ini_path+filename+'.ini'))
            config.read(Ini_path+filename+'.ini')
            print('sections:', ' ', config.sections())
            iris_exist = config.get('iris','exist')
            if iris_exist == 'true':
                iris_centerX = config.get('iris', 'center_x')
                iris_centerY = config.get('iris', 'center_y')
                iris_radius = config.get('iris', 'radius')
            else :
                iris_centerX = []
                iris_centerY = []
                iris_radius = []
            pupil_exist = config.get('pupil','exist')
            if pupil_exist == 'true' :
                pupil_centerX = config.get('pupil', 'center_x')
                pupil_centerY = config.get('pupil', 'center_y')
                pupil_radius = config.get('pupil', 'radius')
            else :
                pupil_centerX = []
                pupil_centerY = []
                pupil_radius = []

    dom = make_xml(iris_exist, iris_centerX, iris_centerY, iris_radius, pupil_exist, pupil_centerX, pupil_centerY, pupil_radius, tmpfilename, tmpwidth, tmpheight, tmpdepth)
    print('当前xml文件路径为： {}'.format(os.path.join(Save_path, filename + '.xml') ))
    xml_name = os.path.join(Save_path, filename + '.xml')  #注意保存路径train和test 要调整一下
    with open(xml_name, 'wb') as f:
        f.write(dom.toprettyxml(indent='\t', encoding='utf-8'))


    #生成对应xml格式文件


    #画图
    # img = cv2.imread('image.jpg')
    # # 第一种转换方法
    # b, g, r = cv2.split(img)
    # img2 = cv2.merge([r, g, b])
    # # 第二种转换方法
    # cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # # 第三种转换方法
    # img = img[..., ::-1]
    #第四种方法：
    # plt.imshow(img,cmap=None ,interpolation='bicubic')
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
    # cv2.waitKey(0)
# print glob.glob(r"E:/Picture/*/*.jpg")
# #获取上级目录的所有.py文件
# print glob.glob(r'../*.py') #相对路径
