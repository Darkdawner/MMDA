"""
The code and network structure are based on https://github.com/megvii-detection/MSPN.
"""

import re
from turtle import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from lib.utils.loss_h import JointsL2Loss, JointsSumLoss


class conv_bn_relu(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, 
            has_bn=False, has_relu=True, efficient=False):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                stride=stride, padding=padding)
        self.has_bn = has_bn
        self.has_relu = has_relu
        self.efficient = efficient
        self.bn = nn.GroupNorm(8,out_planes)
        # self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        def _func_factory(conv, bn, relu, has_bn, has_relu):
            def func(x):
                x = conv(x)
                if has_bn:
                    x = bn(x)
                if has_relu:
                    x = relu(x)
                return x
            return func 

        func = _func_factory(
                self.conv, self.bn, self.relu, self.has_bn, self.has_relu)

        if self.efficient:
            x = checkpoint(func, x)
        else:
            x = func(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None,
            efficient=False):
        super(Bottleneck, self).__init__()
        self.conv_bn_relu1 = conv_bn_relu(in_planes, planes, kernel_size=1,
                stride=1, padding=0, has_bn=True, has_relu=True,
                efficient=efficient) 
        self.conv_bn_relu2 = conv_bn_relu(planes, planes, kernel_size=3,
                stride=stride, padding=1, has_bn=True, has_relu=True,
                efficient=efficient) 
        self.conv_bn_relu3 = conv_bn_relu(planes, planes * self.expansion,
                kernel_size=1, stride=1, padding=0, has_bn=True,
                has_relu=False, efficient=efficient) 
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        out = self.conv_bn_relu1(x)
        out = self.conv_bn_relu2(out)
        out = self.conv_bn_relu3(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x 
        out = self.relu(out)

        return out


class ResNet_top(nn.Module):

    def __init__(self):
        super(ResNet_top, self).__init__()
        self.conv = conv_bn_relu(3, 64, kernel_size=7, stride=2, padding=3,
                has_bn=True, has_relu=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)

        return x


class ResNet_downsample_module(nn.Module):

    def __init__(self, block, layers, has_skip=False, efficient=False,
            zero_init_residual=False):
        super(ResNet_downsample_module, self).__init__()
        self.has_skip = has_skip 
        self.in_planes = 64
        self.layer1 = self._make_layer(block, 64, layers[0],
                efficient=efficient)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                efficient=efficient)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                efficient=efficient)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                efficient=efficient)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, efficient=False):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = conv_bn_relu(self.in_planes, planes * block.expansion,
                    kernel_size=1, stride=stride, padding=0, has_bn=True,
                    has_relu=False, efficient=efficient)

        layers = list() 
        layers.append(block(self.in_planes, planes, stride, downsample,
            efficient=efficient))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, efficient=efficient))

        return nn.Sequential(*layers)

    def forward(self, x, skip1):
        x1 = self.layer1(x)
        if self.has_skip:
            x1 = x1 + skip1[0]
        x2 = self.layer2(x1)
        if self.has_skip:
            x2 = x2 + skip1[1]
        x3 = self.layer3(x2)
        if self.has_skip:
            x3 = x3 + skip1[2]
        x4 = self.layer4(x3)
        if self.has_skip:
            x4 = x4 + skip1[3]

        return x4, x3, x2, x1


class Upsample_unit(nn.Module): 

    def __init__(self, ind, in_planes, out_planes, up_size, output_chl_num, output_shape,
            chl_num=256, gen_skip=False, gen_cross_conv=False, efficient=False):
        super(Upsample_unit, self).__init__()
        self.output_shape = output_shape
        self.xz_output_shape = (512, 208)
        self.yz_output_shape = (128, 512)

        self.u_skip = conv_bn_relu(in_planes, out_planes, kernel_size=1, stride=1,
                padding=0, has_bn=True, has_relu=False, efficient=efficient)
        self.relu = nn.ReLU(inplace=True)

        self.ind = ind
        if self.ind > 0:
            self.up_size = up_size
            self.up_conv = conv_bn_relu(out_planes*2, out_planes, kernel_size=1,
                    stride=1, padding=0, has_bn=True, has_relu=False,
                    efficient=efficient)

        self.gen_skip = gen_skip
        if self.gen_skip:
            self.skip1 = conv_bn_relu(in_planes, in_planes, kernel_size=1,
                    stride=1, padding=0, has_bn=True, has_relu=True,
                    efficient=efficient)
        self.gen_cross_conv = gen_cross_conv
        if self.ind == 3 and self.gen_cross_conv:
            self.cross_conv = conv_bn_relu(out_planes, 64, kernel_size=1,
                    stride=1, padding=0, has_bn=True, has_relu=True,
                    efficient=efficient)

        self.res_conv1 = conv_bn_relu(out_planes, chl_num, kernel_size=1,
                stride=1, padding=0, has_bn=True, has_relu=True,
                efficient=efficient)
        self.res_conv2 = conv_bn_relu(chl_num, output_chl_num[0], kernel_size=3,
                stride=1, padding=1, has_bn=False, has_relu=False,
                efficient=efficient)
        self.res_yz_conv1 = conv_bn_relu(chl_num, chl_num, kernel_size=1,
                                      stride=1, padding=0, has_bn=True, has_relu=True,
                                      efficient=efficient)
        self.res_yz_conv2 = conv_bn_relu(chl_num, output_chl_num[1], kernel_size=3,
                                      stride=1, padding=1, has_bn=False, has_relu=False,
                                      efficient=efficient)
        self.res_xz_conv1 = conv_bn_relu(chl_num, chl_num, kernel_size=1,
                                      stride=1, padding=0, has_bn=True, has_relu=True,
                                      efficient=efficient)
        self.res_xz_conv2 = conv_bn_relu(chl_num, output_chl_num[2], kernel_size=3,
                                      stride=1, padding=1, has_bn=False, has_relu=False,
                                      efficient=efficient)
        self.conv_yz = conv_bn_relu(15*26*(2**self.ind), chl_num, kernel_size=1,
                                      stride=1, padding=0, has_bn=True, has_relu=True,
                                      efficient=efficient)
        self.conv_xz = conv_bn_relu(15*16*(2**self.ind), chl_num, kernel_size=1,
                                      stride=1, padding=0, has_bn=True, has_relu=True,
                                      efficient=efficient)

    def forward(self, x, up_xy, up_yz, up_xz):
        out = self.u_skip(x)

        if self.ind > 0:
            up_xy = F.interpolate(up_xy, size=self.up_size, mode='bilinear', align_corners=True)
            up_xy = self.up_conv(up_xy)
            out += up_xy
            del up_xy
        out = self.relu(out)

        k = int(128/(2**self.ind))
        out_xz = torch.cat([out[:,k*i:k*i+k,:,:].permute(0,2,1,3) for i in range(15)], dim=1)
        out_yz = torch.cat([out[:,k*i:k*i+k,:,:].permute(0,3,2,1) for i in range(15)], dim=1)
        out_xz = self.conv_xz(out_xz)
        out_yz = self.conv_yz(out_yz)

        if self.ind > 0:
            out_yz = F.interpolate(out_yz, size=(self.up_size[0], 512), mode='bilinear', align_corners=True)
            up_yz = F.interpolate(up_yz, size=(self.up_size[0], 512), mode='bilinear', align_corners=True)
            out_xz = F.interpolate(out_xz, size=(512, self.up_size[1]), mode='bilinear', align_corners=True)
            up_xz = F.interpolate(up_xz, size=(512, self.up_size[1]), mode='bilinear', align_corners=True)
            out_yz += up_yz
            out_xz += up_xz
            del up_yz
            del up_xz


        res = self.res_conv1(out)
        res = self.res_conv2(res)
        res = F.interpolate(res, size=self.output_shape, mode='bilinear', align_corners=True)

        res_xz = self.res_xz_conv1(out_xz)
        res_xz = self.res_xz_conv2(res_xz)
        res_xz = F.interpolate(res_xz, size=self.xz_output_shape, mode='bilinear', align_corners=True)

        res_yz = self.res_yz_conv1(out_yz)
        res_yz = self.res_yz_conv2(res_yz)
        res_yz = F.interpolate(res_yz, size=self.yz_output_shape, mode='bilinear', align_corners=True)

        skip1 = None
        if self.gen_skip:
            skip1 = self.skip1(x)

        cross_conv = None
        if self.ind == 3 and self.gen_cross_conv:
            cross_conv = self.cross_conv(out)


        return out, out_yz, out_xz, res, res_xz, res_yz, skip1, cross_conv


class Upsample_module(nn.Module):

    def __init__(self, output_chl_num, output_shape, chl_num=256,
            gen_skip=False, gen_cross_conv=False, efficient=False):
        super(Upsample_module, self).__init__()
        self.in_planes = [2048, 1024, 512, 256] 
        self.out_planes = [1920, 960, 480, 240] 
        h, w = output_shape
        self.up_sizes = [
                (h // 8, w // 8), (h // 4, w // 4), (h // 2, w // 2), (h, w)]
        self.gen_skip = gen_skip
        self.gen_cross_conv = gen_cross_conv

        self.up_1 = Upsample_unit(0, self.in_planes[0], self.out_planes[0], self.up_sizes[0],
                output_chl_num=output_chl_num, output_shape=output_shape,
                chl_num=chl_num, gen_skip=self.gen_skip,
                gen_cross_conv=self.gen_cross_conv, efficient=efficient)
        self.up_2 = Upsample_unit(1, self.in_planes[1], self.out_planes[1], self.up_sizes[1],
                output_chl_num=output_chl_num, output_shape=output_shape,
                chl_num=chl_num, gen_skip=self.gen_skip,
                gen_cross_conv=self.gen_cross_conv, efficient=efficient)
        self.up_3 = Upsample_unit(2, self.in_planes[2], self.out_planes[2], self.up_sizes[2],
                output_chl_num=output_chl_num, output_shape=output_shape,
                chl_num=chl_num, gen_skip=self.gen_skip,
                gen_cross_conv=self.gen_cross_conv, efficient=efficient)
        self.up_4 = Upsample_unit(3, self.in_planes[3], self.out_planes[3], self.up_sizes[3],
                output_chl_num=output_chl_num, output_shape=output_shape,
                chl_num=chl_num, gen_skip=self.gen_skip,
                gen_cross_conv=self.gen_cross_conv, efficient=efficient)

    def forward(self, x4, x3, x2, x1):
        out_xy1, out_yz1, out_xz1, res1, res_xz1, res_yz1, skip1_1, _= self.up_1(x4, None, None, None)
        
        out_xy2, out_yz2, out_xz2, res2, res_xz2, res_yz2, skip1_2, _ = self.up_2(x3, out_xy1, out_yz1, out_xz1)
        
        out_xy3, out_yz3, out_xz3, res3, res_xz3, res_yz3, skip1_3, _ = self.up_3(x2, out_xy2, out_yz2, out_xz2)
        
        out_xy4, out_yz4, out_xz4, res4, res_xz4, res_yz4, skip1_4, cross_conv = self.up_4(x1, out_xy3, out_yz3, out_xz3)

        # 'res' starts from small size
        res = [res1, res2, res3, res4]
        res_xz = [res_xz1, res_xz2, res_xz3, res_xz4]
        res_yz = [res_yz1, res_yz2, res_yz3, res_yz4]
        skip1 = [skip1_4, skip1_3, skip1_2, skip1_1]

        return res, res_xz, res_yz, skip1, cross_conv


class Single_stage_module(nn.Module):

    def __init__(self, output_chl_num, output_shape, has_skip=False,
            gen_skip=False, gen_cross_conv=False, chl_num=256, efficient=False,
            zero_init_residual=False,):
        super(Single_stage_module, self).__init__()
        self.has_skip = has_skip
        self.gen_skip = gen_skip
        self.gen_cross_conv = gen_cross_conv
        self.chl_num = chl_num
        self.zero_init_residual = zero_init_residual 
        self.layers = [3, 4, 6, 3]  # resnet 50
        self.downsample = ResNet_downsample_module(Bottleneck, self.layers,
                self.has_skip, efficient, self.zero_init_residual)
        self.upsample = Upsample_module(output_chl_num, output_shape,
                self.chl_num, self.gen_skip, self.gen_cross_conv, efficient)

    def forward(self, x, skip1):
        x4, x3, x2, x1 = self.downsample(x, skip1)
        res, res_xz, res_yz, skip1, cross_conv = self.upsample(x4, x3, x2, x1)
        
        return res, res_xz, res_yz, skip1, cross_conv


class MMDA(nn.Module):
    
    def __init__(self, cfg, run_efficient=False, **kwargs):
        super(MMDA, self).__init__()

        self.stage_num = cfg.MODEL.STAGE_NUM
        
        self.kpt_paf_num = cfg.DATASET.KEYPOINT.NUM + cfg.DATASET.PAF.NUM*2
        self.keypoint_num = cfg.DATASET.KEYPOINT.NUM
        self.paf_num = cfg.DATASET.PAF.NUM
        self.output_shape = cfg.OUTPUT_SHAPE
        self.upsample_chl_num = cfg.MODEL.UPSAMPLE_CHANNEL_NUM

        self.ohkm = cfg.LOSS.OHKM
        self.topk = cfg.LOSS.TOPK
        self.ctf = cfg.LOSS.COARSE_TO_FINE
        
        self.top = ResNet_top()
        self.modules_stages = list() 
        for i in range(self.stage_num):
            if i == 0:
                has_skip = False
            else:
                has_skip = True
            if i != self.stage_num - 1:
                gen_skip = True
                gen_cross_conv = True
            else:
                gen_skip = False 
                gen_cross_conv = False 

            self.modules_stages.append(
                    Single_stage_module(
                        [self.kpt_paf_num, self.keypoint_num, self.keypoint_num], self.output_shape,
                        has_skip=has_skip, gen_skip=gen_skip,
                        gen_cross_conv=gen_cross_conv,
                        chl_num=self.upsample_chl_num,
                        efficient=run_efficient,
                        **kwargs
                        )
                    )
            setattr(self, 'stage%d' % i, self.modules_stages[i])
        # for p in self.parameters():
        #     p.requires_grad = False

    def _calculate_loss(self, outputs, valids, labels, xz_labels, yz_labels):
        # outputs: stg1 -> stg2 -> ... , res1: bottom -> up
        # valids: (B, C, 1), labels: (B, 5, C, H, W)
        loss2d_1 = JointsL2Loss()
        loss3d_1 = JointsL2Loss()
        losshm_1 = JointsSumLoss()
        GAUSSIAN_KERNELS = [(15, 15), (11, 11), (9, 9), (7, 7), (5, 5)]
        if self.ohkm:
            loss2d_2 = JointsL2Loss(has_ohkm=self.ohkm, topk=self.topk, paf_num=self.paf_num)
            loss3d_2 = JointsL2Loss(has_ohkm=self.ohkm, topk=self.topk, paf_num=0)
            losshm_2 = JointsSumLoss(has_ohkm=self.ohkm, topk=self.topk)
        loss, loss_2d, loss_3d, loss_hm = 0., 0., 0., 0.
        for i in range(self.stage_num):
            for j in range(4):  # multi-scale
                ind = j
                if i == self.stage_num - 1 and self.ctf:  # coarse-to-fine
                    ind += 1 
                tmp_labels_2d = labels[:, ind, :, :, :]
                tmp_labels_yz = yz_labels[:, ind, :, :, :]
                tmp_labels_xz = xz_labels[:, ind, :, :, :]


                if j == 3 and self.ohkm:
                    tmp_loss_2d = loss2d_2(outputs['heatmap_2d'][i][j],
                                        valids[:, :self.kpt_paf_num], tmp_labels_2d)
                    tmp_loss_3d_xz = loss3d_2(outputs['heatmap_3d_xz'][i][j],
                                        valids[:, self.kpt_paf_num::2], tmp_labels_xz)
                    tmp_loss_3d_yz = loss3d_2(outputs['heatmap_3d_yz'][i][j],
                                        valids[:, self.kpt_paf_num+1::2], tmp_labels_yz)
                    tmp_loss_hm = losshm_2(outputs['heatmap_2d'][i][j][:,:self.keypoint_num], outputs['heatmap_3d'][i][j],
                                        valids[:,self.kpt_paf_num::2])
                else:
                    tmp_loss_2d = loss2d_1(outputs['heatmap_2d'][i][j],
                                        valids[:, :self.kpt_paf_num], tmp_labels_2d)
                    tmp_loss_3d_xz = loss3d_1(outputs['heatmap_3d_xz'][i][j],
                                        valids[:, self.kpt_paf_num::2], tmp_labels_xz)
                    tmp_loss_3d_yz = loss3d_1(outputs['heatmap_3d_yz'][i][j],
                                        valids[:, self.kpt_paf_num+1::2], tmp_labels_yz)
                    tmp_loss_hm = losshm_1(outputs['heatmap_2d'][i][j][:,:self.keypoint_num], outputs['heatmap_3d'][i][j],
                                        valids[:,self.kpt_paf_num::2])
                
                if j == 3:
                    loss_2d += tmp_loss_2d
                    loss_3d += (tmp_loss_3d_xz + tmp_loss_3d_yz)/2
                    loss_hm += tmp_loss_hm

                tmp_loss = tmp_loss_2d + 2*(tmp_loss_3d_xz + tmp_loss_3d_yz) + tmp_loss_hm

                if j < 3:
                    tmp_loss /= 4

                loss += tmp_loss

        return dict(total_loss=loss, loss_2d=loss_2d, loss_3d=loss_3d, loss_hm=loss_hm)
        
    def forward(self, imgs, valids=None, labels=None, xz_labels=None, yz_labels=None):
        x = self.top(imgs)
        skip1 = None
        skip2 = None
        outputs = dict()
        outputs['heatmap_2d'] = list()
        outputs['heatmap_3d_xz'] = list()
        outputs['heatmap_3d_yz'] = list()
        for i in range(self.stage_num):
            res, res_xz, res_yz, skip1, x= eval('self.stage' + str(i))(x, skip1)
            outputs['heatmap_2d'].append(res)
            outputs['heatmap_3d_xz'].append(res_xz)
            outputs['heatmap_3d_yz'].append(res_yz)

        if valids is None and labels is None:
            outputs_2d = outputs['heatmap_2d'][-1][-1] + outputs['heatmap_2d'][-1][-2] + outputs['heatmap_2d'][-1][-3]
            xz = outputs['heatmap_3d_xz'][-1][-1] + outputs['heatmap_3d_xz'][-1][-2] + outputs['heatmap_3d_xz'][-1][-3]
            yz = outputs['heatmap_3d_yz'][-1][-1] + outputs['heatmap_3d_yz'][-1][-2] + outputs['heatmap_3d_yz'][-1][-3]
            return outputs_2d, xz, yz
        else:
            return self._calculate_loss(outputs, valids, labels, xz_labels, yz_labels)
