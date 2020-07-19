import argparse
from mmdet.apis import init_detector, inference_detector
import mmcv


parser = argparse.ArgumentParser(description='show img')
parser.add_argument('--cfg', default=None, help='config file path')
parser.add_argument('--img', help='img path')
parser.add_argument('--ckpt', default=None, help='checkpoint file path')
args = parser.parse_args()

if args.cfg==None:
    config_file = 'configs/reppoints/reppoints_moment_r50_fpn_1x_citypersons.py'
else:
    config_file=args.cfg
if args.ckpt==None:
    checkpoint_file = 'work_dirs/reppoints_moment_r50_fpn_1x_coco_20200330-b73db8d1.pth'
else:
    checkpoint_file=args.ckpt


# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = args.img  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
# model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='data/result.jpg')
