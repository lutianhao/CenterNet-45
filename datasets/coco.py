import os
import cv2
import json
import math
import numpy as np

import torch
import torch.utils.data as data
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval

from utils.image import get_border, get_affine_transform, affine_transform, color_aug
from utils.image import draw_umich_gaussian, gaussian_radius

COCO_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
              'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
              'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
              'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
              'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
              'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
              'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
              'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
              'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
              'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
              'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
              'scissors', 'teddy bear', 'hair drier', 'toothbrush']

COCO_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90]

COCO_MEAN = [0.40789654, 0.44719302, 0.47026115]
COCO_STD = [0.28863828, 0.27408164, 0.27809835]
COCO_EIGEN_VALUES = [0.2141788, 0.01817699, 0.00341571]
COCO_EIGEN_VECTORS = [[-0.58752847, -0.69563484, 0.41340352],
                      [-0.5832747, 0.00994535, -0.81221408],
                      [-0.56089297, 0.71832671, 0.41158938]]


class COCO(data.Dataset):
  def __init__(self, data_dir, split, split_ratio=1.0, img_size=512):
    super(COCO, self).__init__()
    self.num_classes = 80    #类别数目
    self.class_name = COCO_NAMES
    self.valid_ids = COCO_IDS    #类别的id
    self.cat_ids = {v: i for i, v in enumerate(self.valid_ids)}

    self.data_rng = np.random.RandomState(123)
    self.eig_val = np.array(COCO_EIGEN_VALUES, dtype=np.float32)   #特征值
    self.eig_vec = np.array(COCO_EIGEN_VECTORS, dtype=np.float32)  #特征向量
    self.mean = np.array(COCO_MEAN, dtype=np.float32)[None, None, :]   #数据集均值
    self.std = np.array(COCO_STD, dtype=np.float32)[None, None, :]    #数据集标准差

    self.split = split   #翻译是分开，就是到底是test还是train还是valid
    self.data_dir = os.path.join(data_dir, 'coco')   #数据路径
    self.img_dir = os.path.join(self.data_dir, '%s2017' % split)     #这是图片的地址
    if split == 'test':
      self.annot_path = os.path.join(self.data_dir, 'annotations', 'image_info_test-dev2017.json')  #这里定义test的解释文件地址
    else:
      self.annot_path = os.path.join(self.data_dir, 'annotations', 'instances_%s2017.json' % split) #这里定义其他的解释文件地址

    self.max_objs = 128
    self.padding = 127  # 31 for resnet/resdcn
    self.down_ratio = 4   #翻译为下降比率
    self.img_size = {'h': img_size, 'w': img_size}  #图片大小
    self.fmap_size = {'h': img_size // self.down_ratio, 'w': img_size // self.down_ratio} #经过缩放处理后的图片大小？
    self.rand_scales = np.arange(0.6, 1.4, 0.1)
    self.gaussian_iou = 0.7

    print('==> initializing coco 2017 %s data.' % split)
    self.coco = coco.COCO(self.annot_path)    #这里应该是在通过coco官方库导入数据集
    self.images = self.coco.getImgIds()     #这里获得的应该是每张图片的id值

    if 0 < split_ratio < 1:    #这里应该是如果设定不取全部的值那么就选定部分图片id值吧
      split_size = int(np.clip(split_ratio * len(self.images), 1, len(self.images)))
      self.images = self.images[:split_size]

    self.num_samples = len(self.images)   #样本数目

    print('Loaded %d %s samples' % (self.num_samples, split))

  def __getitem__(self, index):  #这个没记错的话应该是个迭代器
    img_id = self.images[index]  #图片id
    img_path = os.path.join(self.img_dir, self.coco.loadImgs(ids=[img_id])[0]['file_name']) #某张图片路径
    ann_ids = self.coco.getAnnIds(imgIds=[img_id]) #对应图片的注释段落
    annotations = self.coco.loadAnns(ids=ann_ids)  #注释文件
    labels = np.array([self.cat_ids[anno['category_id']] for anno in annotations]) #获得对应的标签
    bboxes = np.array([anno['bbox'] for anno in annotations], dtype=np.float32)  #获得对应的bbox
    if len(bboxes) == 0:
      bboxes = np.array([[0., 0., 0., 0.]], dtype=np.float32)
      labels = np.array([[0]])
    bboxes[:, 2:] += bboxes[:, :2]  # xywh to xyxy

    img = cv2.imread(img_path)
    height, width = img.shape[0], img.shape[1]
    center = np.array([width / 2., height / 2.], dtype=np.float32)  # center of image
    scale = max(height, width) * 1.0  #这里应该是把图片变成个正方形的，取的是长宽中长的那一块

    flipped = False
    if self.split == 'train':
      scale = scale * np.random.choice(self.rand_scales)  #随机尺寸？
      w_border = get_border(128, width)
      h_border = get_border(128, height)
      center[0] = np.random.randint(low=w_border, high=width - w_border)
      center[1] = np.random.randint(low=h_border, high=height - h_border)

      if np.random.random() < 0.5:   #相当于一半一半的概率
        flipped = True
        img = img[:, ::-1, :]
        center[0] = width - center[0] - 1

    trans_img = get_affine_transform(center, scale, 0, [self.img_size['w'], self.img_size['h']])  #这应该就是仿射变换了
    img = cv2.warpAffine(img, trans_img, (self.img_size['w'], self.img_size['h'])) #同上面

    # -----------------------------------debug---------------------------------
    # for bbox, label in zip(bboxes, labels):
    #   if flipped:
    #     bbox[[0, 2]] = width - bbox[[2, 0]] - 1
    #   bbox[:2] = affine_transform(bbox[:2], trans_img)
    #   bbox[2:] = affine_transform(bbox[2:], trans_img)
    #   bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.img_size['w'] - 1)
    #   bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.img_size['h'] - 1)
    #   cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
    #   cv2.putText(img, self.class_name[label + 1], (int(bbox[0]), int(bbox[1])),
    #               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # cv2.imshow('img', img)
    # cv2.waitKey()
    # -----------------------------------debug---------------------------------

    img = img.astype(np.float32) / 255.   #归一化

    if self.split == 'train':
      color_aug(self.data_rng, img, self.eig_val, self.eig_vec)    #看名字叫做色彩增强

    img -= self.mean   #高斯处理
    img /= self.std
    img = img.transpose(2, 0, 1)  # from [H, W, C] to [C, H, W]   换一下通道表示

    trans_fmap = get_affine_transform(center, scale, 0, [self.fmap_size['w'], self.fmap_size['h']])  #对替换的图像做仿射变换？

    hmap = np.zeros((self.num_classes, self.fmap_size['h'], self.fmap_size['w']), dtype=np.float32)  # heatmap
    w_h_ = np.zeros((self.max_objs, 2), dtype=np.float32)  # width and height
    regs = np.zeros((self.max_objs, 2), dtype=np.float32)  # regression
    inds = np.zeros((self.max_objs,), dtype=np.int64)
    ind_masks = np.zeros((self.max_objs,), dtype=np.uint8)

    # detections = []
    for k, (bbox, label) in enumerate(zip(bboxes, labels)):
      if flipped:   #flipped 应该是用来判断是否有标签用的，true了才来组织bbox
        bbox[[0, 2]] = width - bbox[[2, 0]] - 1   #设定好bbox
      bbox[:2] = affine_transform(bbox[:2], trans_fmap)
      bbox[2:] = affine_transform(bbox[2:], trans_fmap)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.fmap_size['w'] - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.fmap_size['h'] - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      if h > 0 and w > 0:   #判断得到的是否正确
        obj_c = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        obj_c_int = obj_c.astype(np.int32)  #这个表示的是bbox的中心

        radius = max(0, int(gaussian_radius((math.ceil(h), math.ceil(w)), self.gaussian_iou)))
        draw_umich_gaussian(hmap[label], obj_c_int, radius)
        w_h_[k] = 1. * w, 1. * h
        regs[k] = obj_c - obj_c_int  # discretization error 离散化误差
        inds[k] = obj_c_int[1] * self.fmap_size['w'] + obj_c_int[0]
        ind_masks[k] = 1
        # groundtruth bounding box coordinate with class
        # detections.append([obj_c[0] - w / 2, obj_c[1] - h / 2,
        #                    obj_c[0] + w / 2, obj_c[1] + h / 2, 1, label])

    # detections = np.array(detections, dtype=np.float32) \
    #   if len(detections) > 0 else np.zeros((1, 6), dtype=np.float32)

    return {'image': img,
            'hmap': hmap, 'w_h_': w_h_, 'regs': regs, 'inds': inds, 'ind_masks': ind_masks,
            'c': center, 's': scale, 'img_id': img_id}

  def __len__(self):
    return self.num_samples


class COCO_eval(COCO):
  def __init__(self, data_dir, split, test_scales=(1,), test_flip=False, fix_size=False):
    super(COCO_eval, self).__init__(data_dir, split)
    self.test_flip = test_flip
    self.test_scales = test_scales
    self.fix_size = fix_size  #这个应该是控制是否裁剪图片尺寸的变量

  def __getitem__(self, index):
    img_id = self.images[index]
    img_path = os.path.join(self.img_dir, self.coco.loadImgs(ids=[img_id])[0]['file_name'])
    image = cv2.imread(img_path)
    height, width = image.shape[0:2]

    out = {}
    for scale in self.test_scales:
      new_height = int(height * scale)
      new_width = int(width * scale)

      if self.fix_size:
        img_height, img_width = self.img_size['h'], self.img_size['w']
        center = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
        scaled_size = max(height, width) * 1.0
        scaled_size = np.array([scaled_size, scaled_size], dtype=np.float32)
      else:
        img_height = (new_height | self.padding) + 1
        img_width = (new_width | self.padding) + 1
        center = np.array([new_width // 2, new_height // 2], dtype=np.float32)
        scaled_size = np.array([img_width, img_height], dtype=np.float32)

      img = cv2.resize(image, (new_width, new_height))  #重置图片的大小
      trans_img = get_affine_transform(center, scaled_size, 0, [img_width, img_height])  #这个应该是裁剪了
      img = cv2.warpAffine(img, trans_img, (img_width, img_height))

      img = img.astype(np.float32) / 255.
      img -= self.mean
      img /= self.std
      img = img.transpose(2, 0, 1)[None, :, :, :]  # from [H, W, C] to [1, C, H, W]

      if self.test_flip:
        img = np.concatenate((img, img[:, :, :, ::-1].copy()), axis=0)

      out[scale] = {'image': img,
                    'center': center,
                    'scale': scaled_size,
                    'fmap_h': img_height // self.down_ratio,
                    'fmap_w': img_width // self.down_ratio}

    return img_id, out

  def convert_eval_format(self, all_bboxes):
    # all_bboxes: num_samples x num_classes x 5
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = self.valid_ids[cls_ind - 1]
        for bbox in all_bboxes[image_id][cls_ind]:
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = bbox[4]
          bbox_out = list(map(lambda x: float("{:.2f}".format(x)), bbox[0:4]))

          detection = {"image_id": int(image_id),
                       "category_id": int(category_id),
                       "bbox": bbox_out,
                       "score": float("{:.2f}".format(score))}
          detections.append(detection)
    return detections

  def run_eval(self, results, save_dir=None):
    detections = self.convert_eval_format(results)

    if save_dir is not None:
      result_json = os.path.join(save_dir, "results.json")
      json.dump(detections, open(result_json, "w"))

    coco_dets = self.coco.loadRes(detections)
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")#接下来几行都是cocoeval的使用
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats

  @staticmethod
  def collate_fn(batch):
    out = []
    for img_id, sample in batch:
      out.append((img_id, {s: {k: torch.from_numpy(sample[s][k]).float()
      if k == 'image' else np.array(sample[s][k]) for k in sample[s]} for s in sample}))
    return out


if __name__ == '__main__':
  from tqdm import tqdm
  import pickle

  dataset = COCO('E:\\coco_debug', 'train')
  for d in dataset:
    b1 = d
  #   pass

  pass
  # train_loader = torch.utils.data.DataLoader(dataset, batch_size=2,
  #                                            shuffle=False, num_workers=0,
  #                                            pin_memory=True, drop_last=True)
  #
  # for b in tqdm(train_loader):
  #   pass
