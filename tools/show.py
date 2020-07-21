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
    config_file = 'configs/reppoints/reppoints_moment_r50_fpn_1x_citypersons.py'
else:
    config_file=args.cfg

if args.ckpt==None:
    checkpoint_file = 'work_dirs/reppoints_moment_r50_fpn_1x_coco_20200330-b73db8d1.pth'
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

ptirn("\n\nshow.py")
ptirn("result={}".format(result))
ptirn("type(result)={}".format(type(result)))
ptirn("type(result[0])={}".format(type(result[0]))
ptirn("len(result)={}".format(len(result)))
ptirn("len(result[0])={}".format(len(result[0]))

if args.save == True:
    model.show_result(img, result, out_file=output_file)    # 保存可视化结果到文件
else:
    model.show_result(img, result)  # 在新窗口中可视化结果


