import torch
import torch.nn as nn
from math import exp
import torch.nn.functional as F
import PerceptualSimilarity as ps
from torchvision.models.vgg import vgg16
import numpy as np


#################### Convert RGB to HSV ####################

def RGBtoHSV(img, eps=1e-8):
    '''
    img: tensor, [B,C,H,W], value 0~1, float, unnormalize
    return: img_hsv [B,C,H,W] C for hsv channel
    '''
    # hue:(B,H,W)
    hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)
    # 360 degree divide 60 degree
    #  (r-g) / (max-min) if max == b
    hue[ img[:,2]==img.max(1)[0] ] = 4.0 + ( (img[:,0]-img[:,1]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,2]==img.max(1)[0] ]
    # (b-r) / (max-min) if max == g
    hue[ img[:,1]==img.max(1)[0] ] = 2.0 + ( (img[:,2]-img[:,0]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,1]==img.max(1)[0] ]
    # (g-b) / (max-min) if max == r
    hue[ img[:,0]==img.max(1)[0] ] = (0.0 + ( (img[:,1]-img[:,2]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,0]==img.max(1)[0] ]) % 6
    # max == min
    hue[img.min(1)[0]==img.max(1)[0]] = 0.0
    # divide 6, value from [0,1]
    hue = hue/6
    # saturation
    # (max-min)/max
    saturation = ( img.max(1)[0] - img.min(1)[0] ) / ( img.max(1)[0] + eps )
    # max == 0
    saturation[ img.max(1)[0]==0 ] = 0
    # value: max
    value = img.max(1)[0]

    img_hsv = torch.cat([hue.unsqueeze(1), saturation.unsqueeze(1), value.unsqueeze(1)], dim=1)

    return img_hsv

#################### SSIM function ####################
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1,sigma=-1.):
    if sigma == -1.:
        sigma = 0.3*( (window_size-1)/2.0 - 1 )+0.8

    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
    L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel, sigma=1.5).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs

    return ret

#################### X,Y Gradient ####################

def gradient(x):
    h_x = x.size()[-2]
    w_x = x.size()[-1]
    r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
    l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
    t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
    b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
    return torch.abs(r-l), torch.abs(t-b)

#################### PerceptualLoss ####################

class PerceptualLossVGG16(nn.Module):
    def __init__(self):
        super(PerceptualLossVGG16, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        if torch.cuda.is_available():
            loss_network = loss_network.cuda()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, x, y):
        features_x=[x]
        features_y=[y]
        # obtain features for each layers
        for i in range(len(self.loss_network)):
            layer = self.loss_network[i]
            features_x.append(layer(features_x[-1]))
            features_y.append(layer(features_y[-1]))
        # features_x, features_y: input and all the features
        # extract specific layers : relu1_1, relu2_2, relu3_3, relu4_3
        relu_x = [features_x[4],features_x[9], features_x[16],features_x[23]]
        relu_y = [features_y[4],features_y[9], features_y[16],features_y[23]]
        # for i in range(4):
            # print(relu_x[i].size())
            # print(relu_y[i].size())
        # consider relu1_1, relu2_2, relu3_3
        loss = 0
        # if torch.cuda.is_available():
            # loss = loss.cuda()
        # print(loss)
        for i in range(3):
            loss += self.mse_loss(relu_x[i], relu_y[i])

        return loss

#################### Weighted MSE ####################

def weighted_mse_loss(predict, target, weight):
    return torch.mean(weight*(predict-target)**2)

################### Total Loss ###################
# version 1, date:0611, trans, atmos, nonhaze_rec, dehaze
class GeneratorLoss(nn.Module):
    def __init__(self, w_trans =1., w_atmos = 1., w_nonhaze_rec = 1., w_dehaze=1.,use_weighted_mse = True):
        super(GeneratorLoss, self).__init__()
        self.pixel_loss = nn.MSELoss()
        if torch.cuda.is_available():
            self.pixel_loss = self.pixel_loss.cuda()

        self.vgg_loss = PerceptualLossVGG16()

        self.w_trans = w_trans
        self.w_atmos = w_atmos
        self.w_nonhaze_rec = w_nonhaze_rec
        self.w_dehaze = w_dehaze

        self.last_trans_loss = 0
        self.last_atmos_loss = 0
        self.last_nonhaze_rec_loss = 0
        self.last_dehaze_loss = 0

        self.last_trans_l2 = 0
        self.last_trans_grad = 0
        self.last_trans_ssim = 0

        self.last_atmos_l2 = 0
        # self.last_atmos_grad = 0

        self.last_nonhaze_rec_l2 = 0
        self.last_nonhaze_rec_ssim = 0
        self.last_nonhaze_rec_per = 0

        self.last_dehaze_l2 = 0
        self.last_dehaze_ssim = 0
        self.last_dehaze_per = 0

        self.use_weighted_mse = use_weighted_mse

    def forward(self, nonhaze_rec, dehaze, nonhaze_gt, trans, trans_gt, atmos, atmos_gt):
        atmos = atmos.unsqueeze(-1).unsqueeze(-1)
        atmos = atmos.repeat(1,1,dehaze.size()[-2],dehaze.size()[-1])

        atmos_gt = atmos_gt.unsqueeze(-1).unsqueeze(-1)
        atmos_gt = atmos_gt.repeat(1,1,dehaze.size()[-2],dehaze.size()[-1])

        if self.use_weighted_mse:
            # use trans as weight: normalize to [0,1] and reverse in order to focus on deep scene
            weight = (trans_gt - trans_gt.min())/(trans_gt.max()-trans_gt.min())
            weight = 1+ (1-weight)
            self.last_trans_l2 = weighted_mse_loss(trans, trans_gt, weight)
            weight = weight.repeat(1,3,1,1)
            self.last_atmos_l2 = weighted_mse_loss(atmos, atmos_gt, weight)
            self.last_nonhaze_rec_l2 = weighted_mse_loss(nonhaze_rec, nonhaze_gt, weight)
            self.last_dehaze_l2 = weighted_mse_loss(dehaze, nonhaze_gt, weight)
        else:
            self.last_trans_l2 = self.color_loss(trans, trans_gt)
            self.last_atmos_l2 = self.color_loss(atmos, atmos_gt)
            self.last_nonhaze_rec_l2 = self.color_loss(nonhaze_rec, nonhaze_gt)
            self.last_dehaze_l2 = self.color_loss(dehaze, nonhaze_gt)

        self.last_trans_grad = self.gradient_loss(trans, trans_gt)
        # self.last_atmos_grad = self.gradient_loss(atmos, atmos_gt)

        self.last_trans_ssim = self.ssim_loss(trans, trans_gt)
        self.last_nonhaze_rec_ssim = self.ssim_loss(nonhaze_rec, nonhaze_gt)
        self.last_dehaze_ssim = self.ssim_loss(dehaze, nonhaze_gt)

        self.last_nonhaze_rec_per = self.perceptual_loss(nonhaze_rec, nonhaze_gt)
        self.last_dehaze_per = self.perceptual_loss(dehaze, nonhaze_gt)

        # date: 0708, parameter for uniform syn haze (nyu + ots)
        self.last_trans_loss = 16.16*self.last_trans_l2 + 46.0*self.last_trans_grad + 7.56*self.last_trans_ssim
        self.last_atmos_loss = 116.64*self.last_atmos_l2
        self.last_nonhaze_rec_loss = 9.5*self.last_nonhaze_rec_l2 + 7.07*self.last_nonhaze_rec_ssim + 1.01*self.last_nonhaze_rec_per
        self.last_dehaze_loss = 11.84*self.last_dehaze_l2 + 4.75*self.last_dehaze_ssim + 0.37*self.last_dehaze_per
        # print("Trans:", self.last_trans_loss)
        # print("Nonhaze Rec:", self.last_nonhaze_rec_loss)
        # print("Dehaze:", self.last_dehaze_loss)

        loss = self.w_trans*self.last_trans_loss + self.w_atmos*self.last_atmos_loss
        # print("LOSS:", loss.item())
        loss += self.w_nonhaze_rec*self.last_nonhaze_rec_loss + self.w_dehaze*self.last_dehaze_loss

        return loss

    def color_loss(self,predict, gt):
        return self.pixel_loss(predict, gt)

    def gradient_loss(self, predict, gt):
        dx_pred, dy_pred = gradient(predict)
        dx_true, dy_true = gradient(gt)
        l_edges = torch.mean(torch.abs(dy_pred - dy_true) + torch.abs(dx_pred - dx_true))
        return l_edges

    def ssim_loss(self, predict, gt, DepthValRange=1.0):
        l_ssim = torch.clamp((1 - ssim( predict, gt, val_range = DepthValRange)) * 0.5, 0, 1)
        return l_ssim

    def perceptual_loss(self, predict, gt):
        return self.vgg_loss(predict, gt)


# version 2, date:0712, add hsv loss,trans, atmos, nonhaze_rec, dehaze
class GeneratorLoss_hsv(nn.Module):
    def __init__(self, w_trans =1., w_atmos = 1., w_nonhaze_rec = 1., w_dehaze=1.,w_nonhaze_rec_hsv=0.3,w_dehaze_hsv=0.5,use_weighted_mse = True):
        super(GeneratorLoss_hsv, self).__init__()
        self.pixel_loss = nn.MSELoss()
        if torch.cuda.is_available():
            self.pixel_loss = self.pixel_loss.cuda()

        self.vgg_loss = PerceptualLossVGG16()

        self.w_trans = w_trans
        self.w_atmos = w_atmos
        self.w_nonhaze_rec = w_nonhaze_rec
        self.w_dehaze = w_dehaze
        self.w_nonhaze_rec_hsv = w_nonhaze_rec_hsv
        self.w_dehaze_hsv = w_dehaze_hsv

        self.last_trans_loss = 0
        self.last_atmos_loss = 0
        self.last_nonhaze_rec_loss = 0
        self.last_dehaze_loss = 0

        self.last_trans_l2 = 0
        self.last_trans_grad = 0
        self.last_trans_ssim = 0

        self.last_atmos_l2 = 0
        # self.last_atmos_grad = 0

        self.last_nonhaze_rec_l2 = 0
        self.last_nonhaze_rec_ssim = 0
        self.last_nonhaze_rec_per = 0

        self.last_dehaze_l2 = 0
        self.last_dehaze_ssim = 0
        self.last_dehaze_per = 0

        self.last_nonhaze_rec_hsv_l2 = 0
        self.last_nonhaze_rec_hsv_grad = 0
        self.last_nonhaze_rec_hsv_ssim = 0

        self.last_dehaze_hsv_l2 = 0
        self.last_dehaze_hsv_grad = 0
        self.last_dehaze_hsv_ssim = 0

        self.use_weighted_mse = use_weighted_mse

    def forward(self, nonhaze_rec, dehaze, nonhaze_gt, trans, trans_gt, atmos, atmos_gt):
        atmos = atmos.unsqueeze(-1).unsqueeze(-1)
        atmos = atmos.repeat(1,1,dehaze.size()[-2],dehaze.size()[-1])

        atmos_gt = atmos_gt.unsqueeze(-1).unsqueeze(-1)
        atmos_gt = atmos_gt.repeat(1,1,dehaze.size()[-2],dehaze.size()[-1])

        nonhaze_rec_hsv = RGBtoHSV(nonhaze_rec)
        dehaze_hsv = RGBtoHSV(dehaze)
        nonhaze_gt_hsv = RGBtoHSV(nonhaze_gt)

        if self.use_weighted_mse:
            # use trans as weight: normalize to [0,1] and reverse in order to focus on deep scene
            weight = (trans_gt - trans_gt.min())/(trans_gt.max()-trans_gt.min())
            weight = 1+ (1-weight)
            self.last_trans_l2 = weighted_mse_loss(trans, trans_gt, weight)
            self.last_atmos_l2 = weighted_mse_loss(atmos, atmos_gt, weight)
            self.last_nonhaze_rec_l2 = weighted_mse_loss(nonhaze_rec, nonhaze_gt, weight)
            self.last_dehaze_l2 = weighted_mse_loss(dehaze, nonhaze_gt, weight)
            self.last_nonhaze_rec_hsv_l2 = weighted_mse_loss(nonhaze_rec_hsv,nonhaze_gt_hsv, weight)
            self.last_dehaze_hsv_l2 = weighted_mse_loss(dehaze_hsv,nonhaze_gt_hsv, weight)
        else:
            self.last_trans_l2 = self.color_loss(trans, trans_gt)
            self.last_atmos_l2 = self.color_loss(atmos, atmos_gt)
            self.last_nonhaze_rec_l2 = self.color_loss(nonhaze_rec, nonhaze_gt)
            self.last_dehaze_l2 = self.color_loss(dehaze, nonhaze_gt)
            self.last_nonhaze_rec_hsv_l2 = self.color_loss(nonhaze_rec_hsv,nonhaze_gt_hsv)
            self.last_dehaze_hsv_l2 = self.color_loss(dehaze_hsv,nonhaze_gt_hsv)

        self.last_trans_grad = self.gradient_loss(trans, trans_gt)
        # s,v gradient
        self.last_nonhaze_rec_hsv_grad = self.gradient_loss(nonhaze_rec_hsv[:,1:],nonhaze_gt_hsv[:,1:])
        self.last_dehaze_hsv_grad = self.gradient_loss(dehaze_hsv[:,1:],nonhaze_gt_hsv[:,1:])

        # self.last_atmos_grad = self.gradient_loss(atmos, atmos_gt)

        self.last_trans_ssim = self.ssim_loss(trans, trans_gt)
        self.last_nonhaze_rec_ssim = self.ssim_loss(nonhaze_rec, nonhaze_gt)
        self.last_dehaze_ssim = self.ssim_loss(dehaze, nonhaze_gt)
        # s,v ssim
        self.last_nonhaze_rec_hsv_ssim = self.ssim_loss(nonhaze_rec_hsv[:,1:],nonhaze_gt_hsv[:,1:])
        self.last_dehaze_hsv_ssim = self.ssim_loss(dehaze_hsv[:,1:],nonhaze_gt_hsv[:,1:])

        self.last_nonhaze_rec_per = self.perceptual_loss(nonhaze_rec, nonhaze_gt)
        self.last_dehaze_per = self.perceptual_loss(dehaze, nonhaze_gt)

        # date: 0712, parameter for uniform syn haze (nyu + ots)
        self.last_trans_loss = 18.43*self.last_trans_l2 + 45.54*self.last_trans_grad + 7.88*self.last_trans_ssim
        self.last_atmos_loss = 115.20*self.last_atmos_l2
        self.last_nonhaze_rec_loss = 8.93*self.last_nonhaze_rec_l2 + 6.73*self.last_nonhaze_rec_ssim + 0.99*self.last_nonhaze_rec_per
        self.last_dehaze_loss = 10.89*self.last_dehaze_l2 + 4.63*self.last_dehaze_ssim + 0.36*self.last_dehaze_per
        self.last_nonhaze_rec_hsv_loss = 7.3*self.last_nonhaze_rec_hsv_l2 + 13.71*self.last_nonhaze_rec_hsv_grad+4.99*self.last_nonhaze_rec_hsv_ssim
        self.last_dehaze_hsv_loss = 7.09*self.last_dehaze_hsv_l2 + 9.68*self.last_dehaze_hsv_grad+3.58*self.last_dehaze_hsv_ssim

        # print("Trans:", self.last_trans_loss)
        # print("Nonhaze Rec:", self.last_nonhaze_rec_loss)
        # print("Dehaze:", self.last_dehaze_loss)

        loss = self.w_trans*self.last_trans_loss + self.w_atmos*self.last_atmos_loss
        # print("LOSS:", loss.item())
        loss += self.w_nonhaze_rec*self.last_nonhaze_rec_loss + self.w_dehaze*self.last_dehaze_loss
        loss += self.w_nonhaze_rec_hsv*self.last_nonhaze_rec_hsv_loss + self.w_dehaze_hsv*self.last_dehaze_hsv_loss

        return loss

    def color_loss(self,predict, gt):
        return self.pixel_loss(predict, gt)

    def gradient_loss(self, predict, gt):
        dx_pred, dy_pred = gradient(predict)
        dx_true, dy_true = gradient(gt)
        l_edges = torch.mean(torch.abs(dy_pred - dy_true) + torch.abs(dx_pred - dx_true))
        return l_edges

    def ssim_loss(self, predict, gt, DepthValRange=1.0):
        l_ssim = torch.clamp((1 - ssim( predict, gt, val_range = DepthValRange)) * 0.5, 0, 1)
        return l_ssim

    def perceptual_loss(self, predict, gt):
        return self.vgg_loss(predict, gt)

# version 2, date:0625, trans, atmos, dehaze
# class GeneratorLoss(nn.Module):
#     def __init__(self, w_trans =1., w_atmos = 1.,w_dehaze=1.,w_dehaze_hsv=1.,use_weighted_mse = True):
#         super(GeneratorLoss, self).__init__()
#         self.pixel_loss = nn.MSELoss()
#         if torch.cuda.is_available():
#             self.pixel_loss = self.pixel_loss.cuda()

#         self.vgg_loss = PerceptualLossVGG16()

#         self.w_trans = w_trans
#         self.w_atmos = w_atmos
#         self.w_dehaze = w_dehaze
#         self.w_dehaze_hsv = w_dehaze_hsv

#         self.last_trans_loss = 0
#         self.last_atmos_loss = 0
#         self.last_dehaze_loss = 0
#         self.last_dehaze_hsv_loss = 0

#         self.last_trans_l2 = 0
#         self.last_trans_grad = 0
#         self.last_trans_ssim = 0

#         self.last_atmos_l2 = 0

#         self.last_dehaze_l2 = 0
#         self.last_dehaze_ssim = 0
#         self.last_dehaze_per = 0

#         self.last_dehaze_hsv_l2 = 0
#         self.last_dehaze_hsv_grad = 0
#         self.last_dehaze_hsv_ssim = 0

#         self.use_weighted_mse = use_weighted_mse

#     def forward(self, dehaze, nonhaze_gt, trans, trans_gt, atmos, atmos_gt):
#         dehaze_hsv = RGBtoHSV(dehaze)
#         nonhaze_gt_hsv = RGBtoHSV(nonhaze_gt)

#         if self.use_weighted_mse:
#             # use trans as weight: normalize to [0,1] and reverse in order to focus on deep scene
#             weight = (trans_gt - trans_gt.min())/(trans_gt.max()-trans_gt.min())
#             weight = 1+ (1-weight)
#             self.last_trans_l2 = weighted_mse_loss(trans, trans_gt, weight)
#             self.last_atmos_l2 = weighted_mse_loss(atmos, atmos_gt, weight)
#             self.last_dehaze_l2 = weighted_mse_loss(dehaze, nonhaze_gt, weight)
#             self.last_dehaze_hsv_l2 = weighted_mse_loss(dehaze_hsv,nonhaze_gt_hsv, weight)
#         else:
#             self.last_trans_l2 = self.color_loss(trans, trans_gt)
#             self.last_atmos_l2 = self.color_loss(atmos, atmos_gt)
#             self.last_dehaze_l2 = self.color_loss(dehaze, nonhaze_gt)
#             self.last_dehaze_hsv_l2 = self.color_loss(dehaze_hsv,nonhaze_gt_hsv)

#         self.last_trans_grad = self.gradient_loss(trans, trans_gt)
#         # s,v gradient
#         self.last_dehaze_hsv_grad = self.gradient_loss(dehaze_hsv[:,1:],nonhaze_gt_hsv[:,1:])
#         # self.last_atmos_grad = self.gradient_loss(atmos, atmos_gt)

#         self.last_trans_ssim = self.ssim_loss(trans, trans_gt)
#         self.last_dehaze_ssim = self.ssim_loss(dehaze, nonhaze_gt)
#         # s,v ssim
#         self.last_dehaze_hsv_ssim = self.ssim_loss(dehaze_hsv[:,1:],nonhaze_gt_hsv[:,1:])

#         self.last_dehaze_per = self.perceptual_loss(dehaze, nonhaze_gt)

#         # parameter for uniform syn haze (nyu + ots)
#         self.last_trans_loss = 9.268*self.last_trans_l2 + 49.334*self.last_trans_grad + 6.873*self.last_trans_ssim
#         self.last_atmos_loss = 132.39*self.last_atmos_l2
#         self.last_dehaze_loss = 11.357*self.last_dehaze_l2 + 7.94*self.last_dehaze_ssim + 1.18*self.last_dehaze_per
#         self.last_dehaze_hsv_loss = 8.834*self.last_dehaze_hsv_l2 + 16.134*self.last_dehaze_hsv_grad+5.543*self.last_dehaze_hsv_ssim

#         # print("Trans:", self.last_trans_loss)
#         # print("Nonhaze Rec:", self.last_nonhaze_rec_loss)
#         # print("Dehaze:", self.last_dehaze_loss)

#         loss = self.w_trans*self.last_trans_loss + self.w_atmos*self.last_atmos_loss
#         # print("LOSS:", loss.item())
#         loss += self.w_dehaze*self.last_dehaze_loss + self.w_dehaze_hsv*self.last_dehaze_hsv_loss

#         return loss

#     def color_loss(self,predict, gt):
#         return self.pixel_loss(predict, gt)

#     def gradient_loss(self, predict, gt):
#         dx_pred, dy_pred = gradient(predict)
#         dx_true, dy_true = gradient(gt)
#         l_edges = torch.mean(torch.abs(dy_pred - dy_true) + torch.abs(dx_pred - dx_true))
#         return l_edges

#     def ssim_loss(self, predict, gt, ValRange=1.0):
#         l_ssim = torch.clamp((1 - ssim( predict, gt, val_range = ValRange)) * 0.5, 0, 1)
#         return l_ssim

#     def perceptual_loss(self, predict, gt):
#         return self.vgg_loss(predict, gt)




