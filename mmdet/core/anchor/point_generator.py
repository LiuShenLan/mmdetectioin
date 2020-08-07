import torch

from .builder import ANCHOR_GENERATORS


@ANCHOR_GENERATORS.register_module()
class PointGenerator(object):

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))   # [0,s,2s,...,0,s,2s,...]
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)   # [0,0,...,s,s,...,2s,2s,...]
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_points(self, featmap_size, stride=16, device='cuda'):
        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0., feat_w, device=device) * stride
        shift_y = torch.arange(0., feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        stride = shift_x.new_full((shift_xx.shape[0], ), stride)    # 与xx相同维度的由s组成的一维张量
        shifts = torch.stack([shift_xx, shift_yy, stride], dim=-1)  # [h*w,3],先遍历x再遍历y
        all_points = shifts.to(device)
        return all_points

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        return valid
