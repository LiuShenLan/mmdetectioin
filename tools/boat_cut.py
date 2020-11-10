import numpy as np
import cv2

import argparse
import os
import json

# 读取参数
print("loading args")
parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str,default='data/boat/')
parser.add_argument('--json_path', type=str,default='data/boat_detection/faster_rcnn_x101_64x4d_fpn_1x_coco/score080_iou050/')
parser.add_argument('--u', type=float, default=0.15)
parser.add_argument('--d', type=float, default=0.05)
parser.add_argument('--l', type=float, default=0.05)
parser.add_argument('--r', type=float, default=0.05)
args = parser.parse_args()

assert os.path.exists(args.img_path), "img path is wrong"
assert os.path.exists(args.json_path), "bbox.json path is wrong"

# 读取图片路径
json_list_all = os.listdir(args.json_path) # 所有json文件list
img_dir_list = []
for i in json_list_all:
    if i[-5:] == '.json':
        img_dir_list.append(i[:-5])
img_dir_num = len(img_dir_list) # img文件夹数目

# 处理输出路径
output_path = '/'.join(args.json_path.split('/')[:-4]) + '/' + "boat_cut" + '/' + '/'.join(args.json_path.split('/')[-3:]) + 'u{}d{}l{}r{}/'.format(args.u, args.d, args.l, args.r)
if not os.path.exists(output_path):
    os.makedirs(output_path)

for index_img_dir_name, img_dir_name in enumerate(img_dir_list):
    img_input_dir = args.img_path + img_dir_name + '/'
    img_output_dir = output_path + img_dir_name + '/'
    # 检测当前img_dir是否已经处理过
    if os.path.exists(img_output_dir):
        print("\rclass: {}/{}                                                                                             ".format(index_img_dir_name+1, img_dir_num),end='')
        continue
    os.mkdir(img_output_dir)

    # 加载bbox数据
    bbox_input_file = args.json_path + img_dir_name + '.json'
    with open(bbox_input_file, 'r') as f:
        bbox_input_data = json.load(f)

    # 保存bbox数据
    bbox_output_data = {}
    bbox_output_file = output_path + img_dir_name + '.json'

    # 处理img
    img_input_num = len(bbox_input_data)
    for index_img_name, img_name in enumerate(bbox_input_data.keys()):
        img_input_file = img_input_dir + '/' + img_name
        # img_output_file = img_output_dir + img_name
        
        # 读取图片与bbox信息
        img = cv2.imread(img_input_file)
        bbox_list = bbox_input_data[img_name]['bbox']
        bbox_output = []

        # 合并bbox
        while bbox_list:
            bbox = bbox_list.pop()
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            if not bbox_output:
                bbox_output.append(bbox)
            else:
                flag_bbox_output_append = True
                for i in range(len(bbox_output)):
                    x1_, y1_, x2_, y2_ = bbox[0], bbox[1], bbox[2], bbox[3]
                    if not (x2 < x1_ or x1 > x2_ or y2 < y1_ or y1 > y2_):
                        flag_bbox_output_append = False
                        bbox_output[i][0] = min(x1, x1_)
                        bbox_output[i][1] = min(y1, y1_)
                        bbox_output[i][2] = max(x2, x2_)
                        bbox_output[i][3] = max(y2, y2_)
                if flag_bbox_output_append:
                    bbox_output.append(bbox)
        
        # bbox扩展1.1倍
        x_max = img.shape[1]
        y_max = img.shape[0]
        for i in range(len(bbox_output)):
            w = bbox_output[i][2] - bbox_output[i][0]
            h = bbox_output[i][3] - bbox_output[i][1]
            bbox_output[i][0] = max(bbox_output[i][0] - w * args.l, 0)
            bbox_output[i][1] = max(bbox_output[i][1] - h * args.u, 0)
            bbox_output[i][2] = min(bbox_output[i][2] + w * args.r, x_max)
            bbox_output[i][3] = min(bbox_output[i][3] + h * args.d, y_max)
        
        # 保存bbox信息
        data = {}
        data['bbox'] = bbox_output
        data['shape'] = img.shape[:2]
        bbox_output_data[img_name] = data

        # 保存裁剪后的图片
        img_output_file_name_list = img_name.split('.')
        for i in range(len(bbox_output)):
            # 图片保存路径
            img_output_file = img_output_dir + img_output_file_name_list[0] + '_' + str(i) + '.' + img_output_file_name_list[1]
            # 图片裁剪范围
            x1, y1, x2, y2 = int(bbox_output[i][0]), int(bbox_output[i][1]), int(bbox_output[i][2]), int(bbox_output[i][3])
            img_cropped = img[y1:y2, x1:x2]
            # 保存
            cv2.imwrite(img_output_file, img_cropped)

        # 输出进度
        print("\rclass: {}/{} | img: {}/{} | img name: {}".format(index_img_dir_name+1, img_dir_num, index_img_name+1, img_input_num, img_input_file),end='')
    
    # 保存json文件
    with open(bbox_output_file,'w') as f:
        json.dump(bbox_output_data,f)

print("\rDone                                                        ")