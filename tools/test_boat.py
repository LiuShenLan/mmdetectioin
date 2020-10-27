from mmdet.apis import init_detector, inference_detector
import mmcv

import numpy as np
import argparse
import os
import json

# 读取参数
print("loading args")
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/faster_rcnn/faster_rcnn_boat_x101_64x4d_fpn_1x_coco.py')
parser.add_argument('--checkpoint', type=str, default='checkpoints/faster_rcnn_x101_64x4d_fpn_1x_coco_20200204-833ee192.pth')
parser.add_argument('--data_path', type=str, default='data/boat/')
parser.add_argument('--output_path', type=str, default='data/result_boat/faster_rcnn_x101_64x4d_fpn_1x_coco/score080_iou050/')
args = parser.parse_args()

# 读取图片路径
print("loading img path")
img_dir_list = os.listdir(args.data_path) # img文件夹名称list
img_dir_num = len(img_dir_list) # img文件夹数目
# 处理输出路径
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

# 忽略coco其他79类数据
print("appending other result")
other_result = np.array([],dtype=np.float32).reshape(0,5)
result = []
for _ in range(80):
    result.append(other_result)

# 从config与checkpoint建立模型
print("Creating model",end='')
config_file = args.config
checkpoint_file = args.checkpoint
model = init_detector(config_file, checkpoint_file, device='cuda:0')
print(" | done")

for index_img_dir_name, img_dir_name in enumerate(img_dir_list):
    img_input_dir = args.data_path + img_dir_name + '/'
    img_output_dir = args.output_path + img_dir_name + '/'
    # 检测当前img_dir是否已经处理过
    if os.path.exists(img_output_dir):
        print("\rclass: {}/{}                     ".format(index_img_dir_name+1, img_dir_num),end='')
        continue
    os.mkdir(img_output_dir)

    # bbox数据
    bbox_data = {}
    bbox_output_file = args.output_path + img_dir_name + 'bbox.json'

    # 处理img
    img_name_list = os.listdir(img_input_dir)
    img_input_num = len(img_name_list)

    for index_img_name, img_name in enumerate(img_name_list):
        img_input_file = img_input_dir + '/' + img_name
        img_output_file = img_output_dir + img_name
        
        img = mmcv.imread(img_input_file)   # 仅加载图像一次
        result[8] = inference_detector(model, img)[8]   # 仅读取boat类别

        # 保存bbox信息
        if result[8].size == 0:
            continue
        data = {}
        data['bbox'] = result[8].tolist()
        data['shape'] = img.shape[:2]
        bbox_data[img_name] = data

        # 保存结果
        model.show_result(img, result, out_file=img_output_file)    # 保存可视化结果到img文件

        # 输出进度
        print("\rclass: {}/{} | img: {}/{}".format(index_img_dir_name+1, img_dir_num, index_img_name+1, img_input_num),end='')
    
    # 保存json文件
    with open(bbox_output_file,'w') as f:
        json.dump(bbox_data,f)

print("\rDone                                           ")