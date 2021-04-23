import json
import random
import argparse

def loadArgs():
    print("loading args")
    parser = argparse.ArgumentParser()
    parser.add_argument('--u', type=float, default=0.05)
    parser.add_argument('--d', type=float, default=0.05)
    parser.add_argument('--l', type=float, default=0.05)
    parser.add_argument('--r', type=float, default=0.05)
    parser.add_argument('--input_file', type=str, default='/home/lsl/data/boat/ships_generate/bbox_gt.json')
    parser.add_argument('--output_file', type=str, default='/home/lsl/data/boat/ships_generate/bbox_random.json')
    args = parser.parse_args()
    return args

def loadJson(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def saveJson(json_file, data):
    with open(json_file, 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    args = loadArgs()
    data_input = loadJson(args.input_file)
    data_output = {}
    gt_images = data_input['images']
    gt_anno = data_input['annotations']

    for img_label in gt_images:
        img_id = img_label[0]["id"]
        img_name = img_label[0]["file_name"]
        width_max = img_label[0]["width"]
        height_max = img_label[0]["height"]

        data_bbox = []

        for img_anno in gt_anno:
            if img_anno[0]["image_id"] == img_id:
                x1_gt, y1_gt, w_gt, h_gt = img_anno[0]["bbox"]
                x1, y1, x2, y2 = x1_gt, y1_gt, x1_gt + w_gt, y1_gt + h_gt

                x1 -= int(w_gt * random.uniform(0, args.l))
                x2 += int(w_gt * random.uniform(0, args.r))
                y1 -= int(h_gt * random.uniform(0, args.u))
                y2 += int(h_gt * random.uniform(0, args.d))

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width_max, x2)
                y2 = min(height_max, y2)

                cat_id = img_anno[0]["category_id"]
                data_bbox.append([x1, y1, x2, y2, cat_id])
        data_output[img_name] = data_bbox
    saveJson(args.output_file, data_output)