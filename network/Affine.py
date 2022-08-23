import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torchvision import models
from torch.autograd import Variable


class AffineGridGen(Module):
    def __init__(self, out_h=240, out_w=240, out_ch=3):
        super(AffineGridGen, self).__init__()
        self.out_h = out_h
        self.out_w = out_w
        self.out_ch = out_ch

    def forward(self, theta):
        theta = theta.contiguous()
        batch_size = theta.size()[0]
        out_size = torch.Size((batch_size, self.out_ch, self.out_h, self.out_w))
        return F.affine_grid(theta, out_size, align_corners=False)


class Res_backbone(nn.Module):
    def __init__(self, device='cuda:0', feature_extraction_cnn='resnet34', last_layer='', imgsize=512):
        super(Res_backbone, self).__init__()
        self.device = device
        self.imgsize = imgsize
        output_channel_dict = dict()
        down_factor = dict()
        if feature_extraction_cnn == 'resnet34':
            for i in range(1,5):
                output_channel_dict[f'layer{i}'] = (2 ** (i-1)) * 64
                down_factor[f'layer{i}'] = 2 * (2 ** i)
            self.model = models.resnet34(pretrained=True)

        if feature_extraction_cnn == 'resnet50':
            for i in range(1,5):
                output_channel_dict[f'layer{i}'] = (2 ** (i-1)) * 256
                down_factor[f'layer{i}'] = 2 * (2 ** i)
            self.model = models.resnet50(pretrained=True)

        if feature_extraction_cnn == 'resnet101':
            for i in range(1,5):
                output_channel_dict[f'layer{i}'] = (2 ** (i-1)) * 256
                down_factor[f'layer{i}'] = 2 * (2 ** i)
            self.model = models.resnet101(pretrained=True)

        self.output_channel = output_channel_dict[last_layer]
        self.out_h = int(self.imgsize / down_factor[last_layer])

        # freeze parameters
        for param in self.model.parameters():
        # save lots of memory
            # param.requires_grad = False
            param.requires_grad = True
        # move to GPU
        if True:
            self.model.to(self.device)

    def forward(self, image_batch):
        x = self.model.conv1(image_batch)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x_1 = self.model.layer1(x)
        x_2 = self.model.layer2(x_1)
        x_3 = self.model.layer3(x_2)
        x_4 = self.model.layer4(x_3)
        return x_4

    def forward_features(self, image_batch):
        x = self.model.conv1(image_batch)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x_1 = self.model.layer1(x)
        x_2 = self.model.layer2(x_1)
        x_3 = self.model.layer3(x_2)
        x_4 = self.model.layer4(x_3)
        return x_1, x_2, x_3, x_4

class FeatureL2Norm(torch.nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), dim=1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)


class FeatureCorrelation(torch.nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()

    def forward(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()
        # reshape features for matrix multiplication
        feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w)
        feature_B = feature_B.view(b, c, h * w).transpose(1, 2)  # [b, h*w, c]
        # perform matrix mult.
        feature_mul = torch.bmm(feature_B, feature_A)  # batch matrix multiplication->(b, h*w, h*w)
        correlation_tensor = feature_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)
        return correlation_tensor


class FeatureRegression(nn.Module):
    def __init__(self, out_h, output_dim=6, device='cuda:0'):
        super(FeatureRegression, self).__init__()
        self.device = device
        output_channel = out_h * out_h
        self.conv = nn.Sequential(
            nn.Conv2d(output_channel, int(output_channel / 2), kernel_size=7, padding=0),
            nn.BatchNorm2d(int(output_channel / 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(output_channel / 2), int(output_channel / 4), kernel_size=5, padding=0),
            nn.BatchNorm2d(int(output_channel / 4)),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(int(output_channel / 4) * (out_h - 10) * (out_h - 10), output_dim)

        if True:
            self.conv.to(self.device)
            self.linear.to(self.device)

    def forward(self, x):
        x = self.conv(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.linear(x)
        return x

class Co_Attention(torch.nn.Module):
    def __init__(self):
        super(Co_Attention, self).__init__()
        self.softmax = torch.nn.Softmax(dim=-1)

    def Correlation_layer(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()
        feature_A = feature_A.reshape(b, c, h * w)
        feature_B = feature_B.reshape(b, c, h * w).transpose(1, 2).contiguous()
        corr = torch.bmm(feature_B, feature_A)  # [batch,idx_B=row_B+h*col_B,idx_A=row_A+h*col_A]
        corr = corr.transpose(1, 2)  # [b,h*w,h*w]
        return corr

    def forward(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()
        corr = self.Correlation_layer(feature_A, feature_B)
        atten_A = torch.bmm(corr, feature_B.reshape(b, c, h * w).transpose(1, 2))  # [b,h*w,c]
        atten_B = torch.bmm(corr.transpose(1, 2), feature_A.reshape(b, c, h * w).transpose(1, 2))
        atten_A = self.softmax(atten_A.transpose(1, 2)).reshape(b, c, h, w)
        atten_B = self.softmax(atten_B.transpose(1, 2)).reshape(b, c, h, w)
        return atten_A, atten_B

class FeatureRegression_affine(nn.Module):
    def __init__(self, out_h, output_dim=6, device='cuda:0'):
        super(FeatureRegression_affine, self).__init__()
        self.device = device
        output_channel = out_h * out_h
        self.conv = nn.Sequential(
            nn.Conv2d(output_channel, int(output_channel / 2), kernel_size=7, padding=0),
            nn.BatchNorm2d(int(output_channel / 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(output_channel / 2), int(output_channel / 4), kernel_size=5, padding=0),
            nn.BatchNorm2d(int(output_channel / 4)),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(int(output_channel / 4) * (out_h - 10) * (out_h - 10), output_dim)

        if True:
            self.conv.to(self.device)
            self.linear.to(self.device)

    def forward(self, x):
        x = self.conv(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.linear(x)
        return x


class Affine(nn.Module):
    def __init__(self, imgsize=512, device='cuda:0', feature_extraction_cnn='resnet34',
                 backbone_layer='layer4',
                 normalize_features=True, normalize_matches=True,theta_out=6):
        super(Affine, self).__init__()
        self.imgsize = imgsize
        self.device = device
        self.normalize_features = normalize_features
        self.normalize_matches = normalize_matches
        self.FeatureExtraction = Res_backbone(device=self.device, feature_extraction_cnn=feature_extraction_cnn,
                                              last_layer=backbone_layer, imgsize=self.imgsize)


        out_h = 16
        self.adpPool = nn.AdaptiveAvgPool2d((out_h, out_h))  # 为适应不同尺寸的图像，添加自适应池化层（ProsRegNet中没有）
        self.FeatureL2Norm = FeatureL2Norm()
        self.Co_Attention = Co_Attention()
        self.co_attn = True
        self.FeatureCorrelation = FeatureCorrelation()
        self.gridGen = AffineGridGen(imgsize, imgsize)
        self.FeatureRegression = FeatureRegression_affine(16, theta_out, device=self.device)  # 输入的维度数应该是最后特征图的h * w
        self.ReLU = nn.ReLU(inplace=True)
        # 特征对齐
        feature_out_h = self.FeatureExtraction.output_channel
        self.attn = True
        self.adpPool_2 = nn.AdaptiveAvgPool2d((1, 1))
        self.generator_signal = False

    def forward(self, source_img, target_img):
        # do feature extraction
        feature_A = self.FeatureExtraction.forward(source_img)
        feature_B = self.FeatureExtraction.forward(target_img)
        # normalize
        feature_A = self.adpPool(feature_A)
        feature_B = self.adpPool(feature_B)
        if self.normalize_features:
            feature_A = self.FeatureL2Norm(feature_A)
            feature_B = self.FeatureL2Norm(feature_B)
        if self.co_attn:
            atten_A, atten_B = self.Co_Attention(feature_A, feature_B)
            feature_A = feature_A + feature_A * atten_A
            feature_B = feature_B + feature_B * atten_B

        # do feature correlation
        correlation = self.FeatureCorrelation(feature_A, feature_B)
        # normalize
        if self.normalize_matches:
            correlation = self.FeatureL2Norm(self.ReLU(correlation))
        # print(correlation.shape)
        # do regression to tnf parameters theta
        theta = self.FeatureRegression(correlation)


        temp = torch.tensor([1.0, 0, 0, 0, 1.0, 0])
        adjust = temp.repeat(theta.shape[0], 1)
        adjust = adjust.to(self.device)
        theta = 0.1 * theta + adjust
        theta = theta.reshape(theta.size()[0], 2, 3)
        theta = theta.to(self.device)

        sampling_grid = self.gridGen(theta)
        warped_source = F.grid_sample(source_img, sampling_grid, padding_mode='border', align_corners=False)
        return warped_source,sampling_grid,theta
