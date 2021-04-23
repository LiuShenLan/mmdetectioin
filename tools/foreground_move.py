from mmdet.apis import init_detector, inference_detector
import mmcv

import numpy as np
import argparse
import os
import json
import time
import shutil

# 读取参数
print("loading args")
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/faster_rcnn/faster_rcnn_boat_x101_64x4d_fpn_1x_coco.py')
parser.add_argument('--checkpoint', type=str, default='checkpoints/faster_rcnn_x101_64x4d_fpn_1x_coco_20200204-833ee192.pth')
parser.add_argument('--data_path', type=str, default='data/foreground/')
parser.add_argument('--move_path', type=str, default='data/foreground_delete/')
parser.add_argument('--threshold_move', type=float, default=0.05)
args = parser.parse_args()

assert os.path.exists(args.config), "config file not exist"
assert os.path.exists(args.checkpoint), "checkpoint file not exist"
assert os.path.exists(args.data_path), "data path is wrong"

# 读取图片路径
print("loading img path")
img_dir_list = os.listdir(args.data_path) # img文件夹名称list
img_dir_num = len(img_dir_list) # img文件夹数目
# 处理输出路径
if not os.path.exists(args.move_path):
    os.makedirs(args.move_path)

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

# 开始时间
time_start = time.time()
img_all_num = 0

img_move_num = 0    # 移动的图片数量

for index_img_dir_name, img_dir_name in enumerate(img_dir_list):
    img_input_dir = os.path.join(args.data_path, img_dir_name)
    img_move_dir = os.path.join(args.move_path, img_dir_name)
    # 检测当前img_dir是否已经处理过
    print("")
    if not os.path.exists(img_move_dir):
        os.mkdir(img_move_dir)

    # 处理img
    img_name_list = os.listdir(img_input_dir)
    img_input_num = len(img_name_list)
    img_all_num += img_input_num
    for index_img_name, img_name in enumerate(img_name_list):
        img_input_file = os.path.join(img_input_dir, img_name)
        img_move_file = os.path.join(img_move_dir, img_name)
        
        # 输出进度
        print("\rclass: {}/{} | img: {}/{} | img name: {}   ".format(index_img_dir_name+1, img_dir_num, index_img_name+1, img_input_num, img_input_file),end='')

        # 判断img大小，若img<2K，则删除图片
        if os.path.getsize(img_input_file) < 5120:
            try:
                os.remove(img_input_file)
            except BaseException:
                continue
        
        # detection
        try:
            img = mmcv.imread(img_input_file)   # 仅加载图像一次
            result[8] = inference_detector(model, mmcv.imread(img_input_file))[8]   # 仅读取boat类别
        except BaseException:
            print("\r{} raise error                                                 ".format(img_input_file), end='')
            continue

        # 移动图像
        if result[8].size == 0:
            img_move_num += 1
            shutil.move(img_input_file, img_move_file)
        else:
            data_detection = result[8].tolist()
            score_max = 0
            for x1, y1, x2, y2, s in data_detection:
                score_max = max(score_max, s)
            if score_max < args.threshold_move:
                img_move_num += 1
                shutil.move(img_input_file, img_move_file)

# 结束时间
time_end = time.time()
time_run = int(time_end - time_start)

print("\nDone! {} images, {} seconds， {} img moved, {} img remain".format(img_all_num, time_run, img_move_num, img_all_num - img_move_num))