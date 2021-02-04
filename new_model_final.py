import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from data_final import UnNormalize
from skimage.util.shape import view_as_blocks
import numpy as np

# This sets the batch norm layers in pytorch as if {'is_training': False, 'scale': True} in tensorflow
def bn_init_as_tf(m):
    if isinstance(m, nn.BatchNorm2d):
        m.track_running_stats = True  # These two lines enable using stats (moving mean and var) loaded from pretrained model
        m.eval()                      # or zero mean and variance of one if the batch norm layer has no pretrained values
        m.affine = True
        m.requires_grad = True


def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes+inter_planes)
        self.conv2 = nn.Conv2d(in_planes+inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False)
        out = torch.cat([x, out], dim=1)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False)
        return F.interpolate(out, scale_factor=2, mode='bilinear',align_corners=True)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        return x + residual

class TransitionLayer(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(TransitionLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1,
                               bias=False)
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = F.avg_pool2d(out, 2)
        return out

class FeatureExtraction(nn.Module):
    def __init__(self, in_planes, out_planes, k_size=1):
        super(FeatureExtraction, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.padding = (k_size-1)//2
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=k_size,stride=1,
                               padding=self.padding, bias=False)
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        return out

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        import torchvision.models as models
        # select model without linear classifier
        self.base_model = models.densenet121(pretrained=True).features
        self.feat_names = ['relu0', 'pool0', 'transition1', 'transition2', 'norm5']
        self.feat_out_channels = [64, 64, 128, 256, 1024]

    def forward(self, x):
        features = [x]
        skip_feat = [x]
        for k, v in self.base_model._modules.items():
            feature = v(features[-1])
            features.append(v(features[-1]))
            if any(x in k for x in self.feat_names):
                skip_feat.append(feature)

        return skip_feat

class decoder_T(nn.Module):
    def __init__(self, out_channel,feat_out_channels, num_features=256):
        super(decoder_T, self).__init__()
        # norm5 (B,1024,H/32,W/32) -> (B,256,H/16,W/16)
        self.dense_block1 = BottleneckBlock(feat_out_channels[4], num_features)
        self.trans_block1 = TransitionBlock(feat_out_channels[4]+num_features,num_features//2)
        self.residual_block11 = ResidualBlock(num_features//2)
        self.residual_block12 = ResidualBlock(num_features//2)

        # (B,128,H/16,W/16) + trans2 (B,256,H/16,W/16) -> (B,164,H/8,W/8)
        self.dense_block2 = BottleneckBlock( num_features//2 + feat_out_channels[3], num_features//2)
        self.trans_block2 = TransitionBlock(num_features + feat_out_channels[3],num_features//4)
        self.residual_block21 = ResidualBlock(num_features//4)
        self.residual_block22 = ResidualBlock(num_features//4)

        # (B,64,H/8,W/8) + trans1 (B,128,H/8,W/8)  -> (B,32,H/4,W/4)
        self.dense_block3 = BottleneckBlock(num_features//4 + feat_out_channels[2], num_features//4)
        self.trans_block3 = TransitionBlock(num_features//2 + feat_out_channels[2], num_features//8)
        self.residual_block31 = ResidualBlock(num_features//8)
        self.residual_block32 = ResidualBlock(num_features//8)

        # (B,32,H/4,W/4) + pool0 (B,64,H/4,W/4) -> (B,32,H/2,W/2)
        self.dense_block4 = BottleneckBlock(num_features//8 + feat_out_channels[1], num_features//8)
        self.trans_block4 = TransitionBlock(num_features//4 + feat_out_channels[1],num_features//8)
        self.residual_block41 = ResidualBlock(num_features//8)
        self.residual_block42 = ResidualBlock(num_features//8)

        # (B,32,H/2,W/2) + relu0 (B,64,H/2,W/2) -> (B,16,H,W)
        self.dense_block5 = BottleneckBlock(num_features//8 + feat_out_channels[0], num_features//8)
        self.trans_block5 = TransitionBlock(num_features//4 + feat_out_channels[0],num_features//16)
        self.residual_block51 = ResidualBlock(num_features//16)
        self.residual_block52 = ResidualBlock(num_features//16)

#         self.conv_refine = nn.Conv2d(num_features//16, 20, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv1010 = nn.Conv2d(num_features//16, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(num_features//16, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(num_features//16, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(num_features//16, 1, kernel_size=1, stride=1, padding=0)  # 1mm

        self.refine1= torch.nn.Sequential(nn.Conv2d(num_features//16 + 4, 20, kernel_size=3, stride=1, padding=1),nn.ELU(inplace=True))
        self.refine2= torch.nn.Sequential(nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1),nn.ReLU(inplace=True))
        self.refine3= torch.nn.Sequential(nn.Conv2d(20, 20, kernel_size=7, stride=1, padding=3),nn.ReLU(inplace=True))
        self.get_transmission_map = torch.nn.Sequential(nn.Conv2d(20, out_channel, kernel_size=7, stride=1, padding=3),nn.Sigmoid())

    def forward(self, features):
        # relu0, pool0, trans1, trans2, norm5: encoder output
        skip0, skip1, skip2, skip3, dense_features = features[1], features[2], features[3], features[4], features[5]
        x1 = self.trans_block1(self.dense_block1(dense_features))
        x1 = self.residual_block11(x1)
        x1 = self.residual_block12(x1)
        x2 = torch.cat([x1, skip3], dim=1)

        x2 = self.trans_block2(self.dense_block2(x2))
        x2 = self.residual_block21(x2)
        x2 = self.residual_block22(x2)
        x3 = torch.cat([x2, skip2], dim=1)

        x3 = self.trans_block3(self.dense_block3(x3))
        x3 = self.residual_block31(x3)
        x3 = self.residual_block32(x3)
        x4 = torch.cat([x3, skip1], dim=1)

        x4 = self.trans_block4(self.dense_block4(x4))
        x4 = self.residual_block41(x4)
        x4 = self.residual_block42(x4)
        x5 = torch.cat([x4, skip0], dim=1)

        x5 = self.trans_block5(self.dense_block5(x5))
        x5 = self.residual_block51(x5)
        x5 = self.residual_block52(x5)
        # x5: original image size with channel number 16
        shape_out = x5.size()
        shape_out = shape_out[2:4]
        x101 = F.avg_pool2d(x5, 32)
        x102 = F.avg_pool2d(x5, 16)
        x103 = F.avg_pool2d(x5, 8)
        x104 = F.avg_pool2d(x5, 4)
        x1010 = F.interpolate(self.relu(self.conv1010(x101)), size=shape_out, mode='bilinear',align_corners=True)
        x1020 = F.interpolate(self.relu(self.conv1020(x102)), size=shape_out, mode='bilinear',align_corners=True)
        x1030 = F.interpolate(self.relu(self.conv1030(x103)), size=shape_out, mode='bilinear',align_corners=True)
        x1040 = F.interpolate(self.relu(self.conv1040(x104)), size=shape_out, mode='bilinear',align_corners=True)
        dehaze = torch.cat((x1010, x1020, x1030, x1040, x5), dim=1)

        dehaze = self.refine1(dehaze)
        dehaze = self.refine2(dehaze)
        dehaze = self.refine3(dehaze)
        trans_map = self.get_transmission_map(dehaze)

        return trans_map
# air light decoder, linear
class decoder_A(nn.Module):
    def __init__(self, out_channel,feat_out_channels,return_value = False):
        super(decoder_A, self).__init__()
        self.atmos_return_value = return_value
        # relu0 (B,64,H/2,W/2) -> (B,64,H/4,W/4)
        self.trans_block1 = TransitionLayer(feat_out_channels[0],feat_out_channels[1])
        # (B,64,H/4,W/4) +  pool0 (B,64,H/4,W/4) -> (B,128,H/8,W/8)
        self.trans_block2 = TransitionLayer(2*feat_out_channels[1],feat_out_channels[2])
        # (B,128,H/8,W/8) + trans1 (B,128,H/8,W/8) -> (B,256,H/16,W/16)
        self.trans_block3 = TransitionLayer(2*feat_out_channels[2], feat_out_channels[3])
        # (B,256,H/16,W/16) + trans2 (B,256,H/16,W/16) -> (B, 512, H/32,W/32)
        self.trans_block4 = TransitionLayer(2*feat_out_channels[3],feat_out_channels[4]//2)
        # self.dense_block5 = BottleneckBlock(feat_out_channels[4]//2 + feat_out_channels[4], feat_out_channels[4]//2)
        self.trans_block5 = TransitionLayer(feat_out_channels[4] + feat_out_channels[4]//2, feat_out_channels[4])

        self.conv1 = nn.Conv2d(feat_out_channels[4], feat_out_channels[4]//4, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(feat_out_channels[4]//4, feat_out_channels[4]//8, kernel_size=1, stride=1, padding=0)
        self.fc1 = nn.Linear(feat_out_channels[4]//8*5*5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.get_atmosphere_map = torch.nn.Sequential(nn.Linear(64, out_channel),nn.Sigmoid())

    def forward(self, features):
        # relu0, pool0, trans1, trans2, norm5: encoder output
        skip0, skip1, skip2, skip3, dense_features = features[1], features[2], features[3], features[4], features[5]
        x1 = self.trans_block1(skip0)
        x2 = torch.cat([x1, skip1], dim=1)

        x2 = self.trans_block2(x2)
        x3 = torch.cat([x2, skip2], dim=1)

        x3 = self.trans_block3(x3)
        x4 = torch.cat([x3, skip3], dim=1)

        x4 = self.trans_block4(x4)
        x5 = torch.cat([x4, dense_features], dim=1)

        x5 = self.trans_block5(x5)

        x6 = F.relu(self.conv1(x5), inplace=True)
        x7 = F.relu(self.conv2(x6), inplace=True)
        x7 = x7.view(-1, 128*5*5)
        x8 = F.relu(self.fc1(x7), inplace=True)
        x9 = F.relu(self.fc2(x8), inplace=True)
        atmos_map = self.get_atmosphere_map(x9)
        # print("airlight:", atmos_map.size())
        if self.atmos_return_value == False:
            atmos_map = atmos_map.unsqueeze(-1).unsqueeze(-1)
            atmos_map = atmos_map.repeat(1,1,features[0].size()[-2],features[0].size()[-1])

        return atmos_map
# version1 (encoder + 2 decoder + refinement) date:0708
# input: dehaze(3), haze_input(3), trans(1), airlight(3)
class refinement(nn.Module):
    def __init__(self, num_features=64):
        super(refinement, self).__init__()
        self.conv1 = torch.nn.Sequential(nn.Conv2d(3+3, num_features-1-3, kernel_size=3, stride=1, padding=1), nn.PReLU())
        self.bn1 = nn.BatchNorm2d(num_features, momentum=0.01, affine=True, eps=1.1e-5)

        self.dense_block1 = BottleneckBlock(num_features, num_features//2)
        self.conv2 = torch.nn.Sequential(nn.Conv2d(num_features+ num_features//2, num_features//2, kernel_size=3, stride=1, padding=1), nn.PReLU())
        self.residual_block11 = ResidualBlock(num_features//2)
        self.residual_block12 = ResidualBlock(num_features//2)

        self.dense_block2 = BottleneckBlock(num_features//2, num_features//4)
        self.conv3 = torch.nn.Sequential(nn.Conv2d(num_features//2+ num_features//4, num_features//4, kernel_size=3, stride=1, padding=1), nn.PReLU())
        self.residual_block21 = ResidualBlock(num_features//4)
        self.residual_block22 = ResidualBlock(num_features//4)

        self.get_dehaze_img = torch.nn.Sequential(nn.Conv2d( num_features // 4, 3, kernel_size=3, stride=1, padding=1),nn.Sigmoid())

    def forward(self, dehaze, haze_input, trans, airlight):
        input1 = self.conv1(torch.cat([dehaze, haze_input], dim=1))
        block = self.bn1(torch.cat([input1, trans, airlight], dim=1))

        block = self.dense_block1(block)
        block = self.conv2(block)
        block = self.residual_block11(block)
        block = self. residual_block12(block)

        block = self.dense_block2(block)
        block = self.conv3(block)
        block = self.residual_block21(block)
        block = self.residual_block22(block)

        dehaze = self.get_dehaze_img(block)

        return dehaze
# version2 (encoder + 2 decoder + refinement) date:0712
# input: dehaze(3), haze_input(3), trans(3), airlight(3)
class refinement_final(nn.Module):
    def __init__(self, num_features=64):
        super(refinement_final, self).__init__()
        self.conv1 = torch.nn.Sequential(nn.Conv2d(3+3, num_features-3-3, kernel_size=3, stride=1, padding=1), nn.PReLU())
        self.bn1 = nn.BatchNorm2d(num_features, momentum=0.01, affine=True, eps=1.1e-5)

        self.dense_block1 = BottleneckBlock(num_features, num_features//2)
        self.conv2 = torch.nn.Sequential(nn.Conv2d(num_features+ num_features//2, num_features//2, kernel_size=3, stride=1, padding=1), nn.PReLU())
        self.residual_block11 = ResidualBlock(num_features//2)
        self.residual_block12 = ResidualBlock(num_features//2)

        self.dense_block2 = BottleneckBlock(num_features//2, num_features//4)
        self.conv3 = torch.nn.Sequential(nn.Conv2d(num_features//2+ num_features//4, num_features//4, kernel_size=3, stride=1, padding=1), nn.PReLU())
        self.residual_block21 = ResidualBlock(num_features//4)
        self.residual_block22 = ResidualBlock(num_features//4)

        self.get_dehaze_img = torch.nn.Sequential(nn.Conv2d( num_features // 4, 3, kernel_size=3, stride=1, padding=1),nn.Sigmoid())

    def forward(self, dehaze, haze_input, trans, airlight):
        input1 = self.conv1(torch.cat([dehaze, haze_input], dim=1))
        block = self.bn1(torch.cat([input1, trans, airlight], dim=1))

        block = self.dense_block1(block)
        block = self.conv2(block)
        block = self.residual_block11(block)
        block = self. residual_block12(block)

        block = self.dense_block2(block)
        block = self.conv3(block)
        block = self.residual_block21(block)
        block = self.residual_block22(block)

        dehaze = self.get_dehaze_img(block)

        return dehaze

# version1 (encoder + 2 decoder + refinement) date:0708
# trans: 1 channel, atmos: (r,g,b)
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.encoder = encoder()
#         self.decoder_T = decoder_T(1,self.encoder.feat_out_channels)
#         self.decoder_A = decoder_A(3,self.encoder.feat_out_channels, return_value=True)
#         self.generate_dehaze = refinement()
#         self.unnormalize_fun = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

#     def forward(self, x):
#         skip_feat = self.encoder(x)
#         # (B,C,H,W), 1 channel
#         trans_map = self.decoder_T(skip_feat)
#         # (B,3), 3 values, each for RGB
#         atmos_light = self.decoder_A(skip_feat)

#         trans = trans_map.repeat(1,3,1,1)
#         atmos_map = atmos_light.unsqueeze(-1).unsqueeze(-1)
#         atmos_map = atmos_map.repeat(1,1,x.size()[-2],x.size()[-1])

#         # Unnormalize haze images
#         hazes_unnormalized = self.unnormalize_fun(x)
#         # Reconstruct clean images
#         nonhaze_rec = (hazes_unnormalized-atmos_map*(1-trans))
#         # nonhaze_rec = torch.clamp(nonhaze_rec, 0.0, 1.0)
#         nonhaze_rec = nonhaze_rec/trans
#         nonhaze_rec = torch.clamp(nonhaze_rec, 0.0, 1.0)
#         # Refinement Module
#         nonhaze_refinement = self.generate_dehaze(nonhaze_rec, hazes_unnormalized, trans_map, atmos_map)

#         return trans_map, atmos_light, nonhaze_rec, nonhaze_refinement

# version2 (encoder + 2 decoder + refinement) date:0712
# trans: 3 channel. atmos: (r,g,b)
# class Model_final(nn.Module):
#     def __init__(self):
#         super(Model_final, self).__init__()
#         self.encoder = encoder()
#         self.decoder_T = decoder_T(3,self.encoder.feat_out_channels)
#         self.decoder_A = decoder_A(3,self.encoder.feat_out_channels, return_value=True)
#         self.generate_dehaze = refinement_final()
#         self.unnormalize_fun = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

#     def forward(self, x):
#         skip_feat = self.encoder(x)
#         # (B,C,H,W), 3 channel
#         trans_map = self.decoder_T(skip_feat)
#         # (B,3), 3 values, each for RGB
#         atmos_light = self.decoder_A(skip_feat)

#         atmos_map = atmos_light.unsqueeze(-1).unsqueeze(-1)
#         atmos_map = atmos_map.repeat(1,1,x.size()[-2],x.size()[-1])

#         # Unnormalize haze images
#         hazes_unnormalized = self.unnormalize_fun(x)
#         # Reconstruct clean images
#         nonhaze_rec = (hazes_unnormalized-atmos_map*(1-trans_map))
#         # nonhaze_rec = torch.clamp(nonhaze_rec, 0.0, 1.0)
#         nonhaze_rec = nonhaze_rec/trans_map
#         nonhaze_rec = torch.clamp(nonhaze_rec, 0.0, 1.0)
#         # Refinement Module
#         nonhaze_refinement = self.generate_dehaze(nonhaze_rec, hazes_unnormalized, trans_map, atmos_map)

#         return trans_map, atmos_light, nonhaze_rec, nonhaze_refinement


# in thesis
# (encoder + 2 decoder + refinement)
# trans: 1 channel, atmos: (r,g,b)
class Model_exp3(nn.Module):
    def __init__(self):
        super(Model_exp3, self).__init__()
        self.encoder = encoder()
        self.decoder_T = decoder_T(1,self.encoder.feat_out_channels)
        self.decoder_A = decoder_A(3,self.encoder.feat_out_channels, return_value=True)
        self.generate_dehaze = refinement_final()
        self.unnormalize_fun = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def forward(self, x):
        skip_feat = self.encoder(x)
        # (B,C,H,W), 1 channel
        trans_map = self.decoder_T(skip_feat)
        trans_map = trans_map.repeat(1,3,1,1)
        # (B,3), 3 values, each for RGB
        atmos_light = self.decoder_A(skip_feat)

        atmos_map = atmos_light.unsqueeze(-1).unsqueeze(-1)
        atmos_map = atmos_map.repeat(1,1,x.size()[-2],x.size()[-1])

        # Unnormalize haze images
        hazes_unnormalized = self.unnormalize_fun(x)
        # Reconstruct clean images
        nonhaze_rec = (hazes_unnormalized-atmos_map*(1-trans_map))
        # nonhaze_rec = torch.clamp(nonhaze_rec, 0.0, 1.0)
        nonhaze_rec = nonhaze_rec/trans_map
        nonhaze_rec = torch.clamp(nonhaze_rec, 0.0, 1.0)
        # Refinement Module
        nonhaze_refinement = self.generate_dehaze(nonhaze_rec, hazes_unnormalized, trans_map, atmos_map)

        return trans_map, atmos_light, nonhaze_rec, nonhaze_refinement


def normalize_to_0_1(x):
    """
    normalize input tensor x to [0,1]
    x: tensor, (B,C,H,W)
    """
    b,c,h,w = x.shape
    x = x.view(b,-1)
    x -= x.min(1, keepdim=True)[0]
    x /= x.max(1, keepdim=True)[0]
    if torch.any(torch.isnan(x)):
        # avoid NaNs caused by dividing 0, should not happen
        x[torch.isnan(x)]=1.0
        print("divide by 0")
    x = x.view(b,c,h,w)
    return x


# class Model_test(nn.Module):
#     """
#     global trans. map, 320x320 input for atmos. map
#     """
#     def __init__(self):
#         super(Model_test, self).__init__()
#         self.encoder = encoder()
#         self.decoder_T = decoder_T(3,self.encoder.feat_out_channels)
#         self.decoder_A = decoder_A(3,self.encoder.feat_out_channels, return_value=True)
#         self.generate_dehaze = refinement()
#         self.unnormalize_fun = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

#     def forward(self, x):
#         skip_feat = self.encoder(x)
#         trans_map = self.decoder_T(skip_feat)

#         skip_feat_airlight = self.encoder(F.interpolate(x, size=(320,320), mode='bilinear', align_corners=True))
#         atmos_light = self.decoder_A(skip_feat_airlight)
#         atmos_light = atmos_light.unsqueeze(-1).unsqueeze(-1)
#         atmos_light = atmos_light.repeat(1,1,x.size()[-2],x.size()[-1])

#         # Unnormalize haze images
#         hazes_unnormalized = self.unnormalize_fun(x)
#         # Reconstruct clean images
#         nonhaze_rec = (hazes_unnormalized-atmos_light*(1-trans_map))
#         # nonhaze_rec = torch.clamp(nonhaze_rec, 0.0, 1.0)
#         nonhaze_rec = nonhaze_rec/trans_map
#         nonhaze_rec = torch.clamp(nonhaze_rec, 0.0, 1.0)
#         # Refinement Module
#         nonhaze_refinement = self.generate_dehaze(nonhaze_rec, hazes_unnormalized, trans_map, atmos_light)

#         return trans_map, atmos_light, nonhaze_rec, nonhaze_refinement

# class Model_atmos(nn.Module):
#     def __init__(self):
#         super(Model_atmos, self).__init__()
#         self.encoder = encoder()
#         self.decoder_A = decoder_A(3,self.encoder.feat_out_channels, return_value=False)

#     def forward(self, x):
#         skip_feat = self.encoder(x)
#         atmos_light = self.decoder_A(skip_feat)
#         return atmos_light

# class Model_trans(nn.Module):
#     def __init__(self):
#         super(Model_trans, self).__init__()
#         self.encoder = encoder()
#         self.decoder_T = decoder_T(3,self.encoder.feat_out_channels)

#     def forward(self, x):
#         skip_feat = self.encoder(x)
#         trans_map = self.decoder_T(skip_feat)

#         return trans_map

#used for testing (only refinement blocks)
# class Model_dehaze(nn.Module):
#     def __init__(self):
#         super(Model_dehaze, self).__init__()
#         self.generate_dehaze = refinement()
#         self.unnormalize_fun = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

#     def forward(self, x, trans_map, atmos_light):
#         # Unnormalize haze images
#         hazes_unnormalized = self.unnormalize_fun(x)
#         # Reconstruct clean images
#         nonhaze_rec = (hazes_unnormalized-atmos_light*(1-trans_map))
#         # nonhaze_rec = torch.clamp(nonhaze_rec, 0.0, 1.0)
#         nonhaze_rec = nonhaze_rec/trans_map
#         nonhaze_rec = torch.clamp(nonhaze_rec, 0.0, 1.0)
#         # Refinement Module
#         nonhaze_refinement = self.generate_dehaze(nonhaze_rec, hazes_unnormalized, trans_map, atmos_light)

#         return nonhaze_rec, nonhaze_refinement

