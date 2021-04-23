import os
import json
import argparse
import cv2

def loadArgs():
    print("loading args")
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, default='/home/lsl/data/boat/ships_generate/bbox_random.json')
    parser.add_argument('--img_path', type=str, default='/home/lsl/data/boat/ships_generate/')
    parser.add_argument('--output_path', type=str, default='/home/lsl/data/boat/ships_generate/detection/')
    parser.add_argument('--line_width', type=int, default=2)
    parser.add_argument('--R', type=int, default=0)
    parser.add_argument('--G', type=int, default=255)
    parser.add_argument('--B', type=int, default=0)
    args = parser.parse_args()
    return args

def loadJson(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    args = loadArgs()
    data_input = loadJson(args.json_file)

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)


    bbox_color = (args.B, args.G, args.R)

    for img_name, img_bbox in data_input.items():
        img = cv2.imread(os.path.join(args.img_path, img_name))
        print("\r{} ".format(img_name), end='')
        for x1, y1, x2, y2, cat_id in img_bbox:
            if cat_id == 4:
                continue
            cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, args.line_width, 8)
        cv2.imwrite(os.path.join(args.output_path, img_name), img)
    print("")
