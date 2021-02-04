import os
import logging
import datetime
import matplotlib
import matplotlib.cm
import numpy as np

# def DepthNorm(depth, maxDepth=1000.0):
#     return maxDepth / depth

# def DepthNorm(depth):
#     """
#     input: tensor, (B,C,H,W) normalize to [0,1]
#     """
#     B,C,H,W = depth.shape
#     depth_ = depth.view(B,C,-1)
#     depth_max = depth_.max(dim=2)[0].unsqueeze(2).unsqueeze(2)
#     depth_min = depth_.min(dim=2)[0].unsqueeze(2).unsqueeze(2)
#     depth_norm = ((depth-depth_min)/(depth_max-depth_min)).view(*depth.shape)
#     return depth_norm

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
      self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

# def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
#     value = value.cpu().numpy()[0,:,:]

#     # normalize
#     vmin = value.min() if vmin is None else vmin
#     vmax = value.max() if vmax is None else vmax
#     if vmin!=vmax:
#         value = (value - vmin) / (vmax - vmin) # vmin..vmax
#     else:
#         # Avoid 0-division
#         value = value*0.
#     # squeeze last dim if it exists
#     #value = value.squeeze(axis=0)

#     cmapper = matplotlib.cm.get_cmap(cmap)
#     value = cmapper(value,bytes=True) # (nxmx4)

#     img = value[:,:,:3]

#     return img.transpose((2,0,1))

# def get_logger(logdir):
#     logger = logging.getLogger("DehazeNet")
#     # '2020-02-11 10:54:25.188198' ->-> '2020-02-11_10:54:25'
#     ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
#     # '2020_02_11_10_54_25'
#     ts = ts.replace(":", "_").replace("-", "_")
#     file_path = os.path.join(logdir, "run_{}.log".format(ts))
#     hdlr = logging.FileHandler(file_path)
#     formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
#     hdlr.setFormatter(formatter)
#     logger.addHandler(hdlr)
#     logger.setLevel(logging.INFO)
#     return logger

