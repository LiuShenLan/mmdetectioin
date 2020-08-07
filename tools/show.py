import argparse
from mmdet.apis import init_detector, inference_detector
import mmcv

# 读取参数
parser = argparse.ArgumentParser(description='show img')
parser.add_argument('--cfg', default=None, help='config file path')
parser.add_argument('--img', help='img path')
parser.add_argument('--ckpt', default=None, help='checkpoint file path')
parser.add_argument('--out', default=None, help='output img path')
parser.add_argument('--save', default=True, help='save or show img')
args = parser.parse_args()

if args.cfg==None:
    config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
else:
    config_file=args.cfg

if args.ckpt==None:
    checkpoint_file = 'data/model/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
else:
    checkpoint_file=args.ckpt

if args.out==None:
    output_file = 'data/result.jpg'
else:
    output_file=args.out

img = args.img  # or img = mmcv.imread(img), which will only load it once

# 建立模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 测试一张图片并显示结果
result = inference_detector(model, img)
# result为list，包含CLASS个元素，每个元素为n*5的np.array，n为对应class所预测到的bbox数目，5为bbox与score

if args.save == True:
    model.show_result(img, result, out_file=output_file)    # 保存可视化结果到文件
else:
    model.show_result(img, result)  # 在新窗口中可视化结果