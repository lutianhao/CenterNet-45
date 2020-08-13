import configparser

#  实例化configParser对象
config = configparser.ConfigParser()
# -read读取ini文件
config.read('/home/tianhao.lu/code/CenterNet-45/data/NICE1/train/circle_params/0001_01_007_0001.ini')
# -sections得到所有的section，并以列表的形式返回
print('sections:', ' ', config.sections())

print('items:', ' ', config.items('iris'))

exist = config.get('iris','exist')
print('this is exist: {}'.format(exist))
print('get:', ' ', config.get('iris', 'exist'))
print('get:', ' ', config.get('iris', 'center_x'))
print('get:', ' ', config.get('iris', 'center_y'))
print('get:', ' ', config.get('iris', 'radius'))
# # -get(section,option)得到section中option的值，返回为string类型
# print('get:', ' ', config.get('cmd', 'startserver'))
#
# # -getint(section,option)得到section中的option的值，返回为int类型
# print('getint:', ' ', config.getint('cmd', 'id'))
# print('getfloat:', ' ', config.getfloat('cmd', 'weight'))
# print('getboolean:', '  ', config.getboolean('cmd', 'isChoice'))
"""
首先得到配置文件的所有分组，然后根据分组逐一展示所有
"""
for sections in config.sections():
    for items in config.items(sections):
        print(items)