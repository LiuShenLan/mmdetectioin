import argparse
import os
import json

# 读取参数
print("loading args")
parser = argparse.ArgumentParser()
parser.add_argument('--json_path', type=str,default='data/boat_detection/faster_rcnn_x101_64x4d_fpn_1x_coco/score080_iou050/')
parser.add_argument('--img_path', type=str,default='data/boat/')
args = parser.parse_args()

assert os.path.exists(args.json_path), "bbox.json path is wrong"

det_img_num = 0
img_num = 0

# 读取json路径
json_list_all = os.listdir(args.json_path) # 所有json文件list
json_list = []
for i in json_list_all:
    if i[-5:] == '.json':
        json_list.append(i)
json_num = len(json_list) # json文件数目

# 读取img文件路径
img_list = os.listdir(args.img_path)

for index_json, json_name in enumerate(json_list):
    json_file = args.json_path + json_name
    img_dir_class = args.img_path + json_name[:-5]
    
    with open(json_file, 'r') as f:
        bbox_data = json.load(f)
    
    det_img_num_class = len(bbox_data)
    det_img_num += det_img_num_class

    img_num_class = len(os.listdir(img_dir_class))
    img_num += img_num_class

    print("{:3d}/{:3d} | img_num:{:7d} | dec_num:{:7d} | miss_num:{:6d} | dec_per:{:6.2f}% | json: {}".format(
        index_json+1, json_num, img_num_class, det_img_num_class, img_num_class - det_img_num_class, det_img_num_class / img_num_class * 100.0, json_name[:-5]))
print("Finish! | img_num:{:7d} | dec_num:{:7d} | miss_num:{:6d} | dec_per:{:6.2f}%".format(
    img_num, det_img_num, img_num - det_img_num, det_img_num / img_num * 100.0))
