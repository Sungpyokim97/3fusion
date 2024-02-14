import torch
import torch.nn as nn
import torchvision
import torch.fft
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
# from PIL import image

import warnings

from mmedit.models.common import (PixelShufflePack, flow_warp)
from mmedit.models.backbones.sr_backbones.basicvsr_net import (ResidualBlocksWithInputConv, SPyNet)

from mmedit.utils import get_root_logger

from mmcv.runner import load_checkpoint
from mmcv.cnn import constant_init
from mmcv.ops import ModulatedDeformConv2d

from davsr.models.common import basicblock as B
import davsr.models.backbones.sr_backbones.slomo as slomo
import torchvision
import PIL
import torchvision.transforms as transforms


"""
# --------------------------------------------
# basic functions
# --------------------------------------------
"""
class BasicVSRPP(nn.Module):
    """BasicVSR++ network structure.

    Support either x4 upsampling or same size output. Since DCN is used in this
    model, it can only be used with CUDA enabled. If CUDA is not enabled,
    feature alignment will be skipped.

    Paper:
        BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation
        and Alignment

    Args:
        mid_channels (int, optional): Channel number of the intermediate
            features. Default: 64.
        num_blocks (int, optional): The number of residual blocks in each
            propagation branch. Default: 7.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
        is_low_res_input (bool, optional): Whether the input is low-resolution
            or not. If False, the output resolution is equal to the input
            resolution. Default: True.
        spynet_pretrained (str, optional): Pre-trained model path of SPyNet.
            Default: None.
        cpu_cache_length (int, optional): When the length of sequence is larger
            than this value, the intermediate features are sent to CPU. This
            saves GPU memory, but slows down the inference speed. You can
            increase this number if you have a GPU with large memory.
            Default: 100.
    """

    def __init__(self,
                 img_channels=3,
                 mid_channels=64,
                 num_blocks=7,
                 max_residue_magnitude=10,
                 is_low_res_input=True,
                 spynet_pretrained=None,
                 vsr_pretrained=None,
                 cpu_cache_length=100):

        super().__init__()
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input
        self.cpu_cache_length = cpu_cache_length
        self.img_channels = img_channels
        
        # optical flow
        self.spynet = SPyNet(pretrained=spynet_pretrained)

        if isinstance(vsr_pretrained, str):
            load_net = torch.load(vsr_pretrained)
            for k, v in load_net['state_dict'].items():
                if k.startswith('generator.'):
                    k = k.replace('generator.', '')
                    load_net[k] = v
                    load_net.pop(k)
            self.load_state_dict(load_net, strict=False)
        elif vsr_pretrained is not None:
            raise TypeError('[vsr_pretrained] should be str or None, '
                            f'but got {type(vsr_pretrained)}.')

        # feature extraction module
        if is_low_res_input:
            self.feat_extract = ResidualBlocksWithInputConv(img_channels, mid_channels, 5)     
        else:
            self.feat_extract = nn.Sequential(
                nn.Conv2d(img_channels, mid_channels, 3, 2, 1),       
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ResidualBlocksWithInputConv(mid_channels, mid_channels, 5))

        # propagation branches
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(modules):
            if torch.cuda.is_available():
                self.deform_align[module] = SecondOrderDeformableAlignment(
                    2 * mid_channels,
                    mid_channels,
                    3,
                    padding=1,
                    deform_groups=16,
                    max_residue_magnitude=max_residue_magnitude)
            self.backbone[module] = ResidualBlocksWithInputConv(
                (2 + i) * mid_channels, mid_channels, num_blocks)

        # upsampling module
        self.reconstruction = ResidualBlocksWithInputConv(
            5 * mid_channels, mid_channels, 5)
        
        self.upsample_last1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample_last2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 1, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 1, upsample_kernel=3)

        self.conv_upsample = nn.Sequential(
                                            nn.Conv2d(3, 64, 3, 1, 1),
                                            nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                            nn.Conv2d(64, 64, 3, 1, 1)
                                            ) 

        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False

        if len(self.deform_align) > 0:
            self.is_with_alignment = True
        else:
            self.is_with_alignment = False
            warnings.warn(
                'Deformable alignment module is not added. '
                'Probably your CUDA is not configured correctly. DCN can only '
                'be used with CUDA enabled. Alignment is skipped now.')
            


    def check_if_mirror_extended(self, lqs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
        """

        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lqs):
        """Compute optical flow using SPyNet for feature alignment.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        if self.cpu_cache:
            flows_backward = flows_backward.cpu()
            flows_forward = flows_forward.cpu()

        return flows_forward, flows_backward

    def propagate(self, feats, flows, module_name):
        """Propagate the latent features throughout the sequence.

        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.

        Return:
            dict(list[tensor]): A dictionary containing all the propagated
                features. Each key in the dictionary corresponds to a
                propagation branch, which is represented by a list of tensors.
        """

        n, t, _, h, w = flows.size()

        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]
            if self.cpu_cache:
                feat_current = feat_current.cuda()
                feat_prop = feat_prop.cuda()
            # second-order deformable alignment
            if i > 0 and self.is_with_alignment:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                if self.cpu_cache:
                    flow_n1 = flow_n1.cuda()

                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:  # second-order features
                    feat_n2 = feats[module_name][-2]
                    if self.cpu_cache:
                        feat_n2 = feat_n2.cuda()

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    if self.cpu_cache:
                        flow_n2 = flow_n2.cuda()

                    flow_n2 = flow_n1 + flow_warp(flow_n2,
                                                  flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                # flow-guided deformable convolution
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond,
                                                           flow_n1, flow_n2)

            # concatenate and residual blocks
            feat = [feat_current] + [
                feats[k][idx]
                for k in feats if k not in ['spatial', module_name]
            ] + [feat_prop]
            if self.cpu_cache:
                feat = [f.cuda() for f in feat]

            feat = torch.cat(feat, dim=1)
            feat_prop = feat_prop + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop)

            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def upsample(self, lqs, feats):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propgation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).

        """
        outputs = []
        num_outputs = len(feats['spatial'])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            hr.insert(0, feats['spatial'][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)
            if self.cpu_cache:
                hr = hr.cuda()
            hr = self.reconstruction(hr)
            # print(f'hr : {hr.size()}')
            hr = self.lrelu(self.upsample1(hr))
            hr = self.lrelu(self.upsample2(hr))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            if self.is_low_res_input:
                # hr += self.img_upsample(lqs[:, i, :, :, :])
                hr += lqs[:, i, :, :, :]
            else:
                hr += lqs[:, i, :, :, :]

            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()

            outputs.append(hr)

        return torch.stack(outputs, dim=1).permute(0,2,1,3,4)

    def forward(self, lqs_ab):    #TODO
    # def forward(self, lqs):         #TODO
        """Forward function for BasicVSR++.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        lqs_ab = lqs_ab.permute(0,2,1,3,4).contiguous()

        if lqs_ab.shape[2] == 4:
            lqs = lqs_ab[:,:,:-1,:,:] #TODO
        else:
            lqs = lqs_ab
        
        n, t, c, h, w = lqs.size()

        # whether to cache the features in CPU
        self.cpu_cache = True if t > self.cpu_cache_length else False

        if self.is_low_res_input:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w), scale_factor=0.25,
                mode='bicubic').view(n, t, c, h // 4, w // 4)

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lqs)

        feats = {}
        # compute spatial features
        if self.cpu_cache:
            feats['spatial'] = []
            for i in range(0, t):
                if lqs_ab.shape[2] == 4:
                    feat = self.feat_extract(lqs_ab[:, i, :, :, :]).cpu()  # TODO
                else:
                    feat = self.feat_extract(lqs[:, i, :, :, :]).cpu()   
                feats['spatial'].append(feat)
                torch.cuda.empty_cache()
        else:
            if lqs_ab.shape[2] == 4:
                feats_ = self.feat_extract(lqs_ab.view(n*t, -1, h, w))     # TODO
            else:
                feats_ = self.feat_extract(lqs.view(n*t, -1, h, w))
            h, w = feats_.shape[2:]
            feats_ = feats_.view(n, t, -1, h, w)
            feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]

        # compute optical flow using the low-res inputs
        # assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
        #     'The height and width of low-res inputs must be at least 64, '
        #     f'but got {h} and {w}.')
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        # feature propgation
        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'

                feats[module] = []

                if direction == 'backward':
                    flows = flows_backward
                elif flows_forward is not None:
                    flows = flows_forward
                else:
                    flows = flows_backward.flip(1)

                feats = self.propagate(feats, flows, module)
                if self.cpu_cache:
                    del flows
                    torch.cuda.empty_cache()

        return self.upsample(lqs, feats)


class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.

    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(
            torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1,
                                                    offset_1.size(1) // 2, 1,
                                                    1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1,
                                                    offset_2.size(1) // 2, 1,
                                                    1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding, self.dilation, mask)


def splits3D(a, sf):
    '''split a into sfxsf distinct blocks

    Args:
        a: NxCxTxWxH
        sf: 3x1 split factor

    Returns:
        b: NxCxWxHx(sf0*sf1*sf2)
    '''
    
    b = torch.stack(torch.chunk(a, sf[0], dim=2), dim=5)
    b = torch.cat(torch.chunk(b, sf[1], dim=3), dim=5)
    b = torch.cat(torch.chunk(b, sf[2], dim=4), dim=5)

    return b


def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.

    Args:
        psf: NxTxCxHxW
        shape: [H, W]

    Returns:
        otf: NxCxHxWx2
    '''
    otf = torch.zeros(psf.shape[:2] + shape).type_as(psf)   # [1, 1, 100, 1, 1]
    otf[:, :, :psf.shape[2], ...].copy_(psf)                # [1, 1, 100, 1, 1]
    # for axis, axis_size in enumerate(psf.shape[2:]):
    otf = torch.roll(otf, -int(psf.shape[2]/2), dims=2)     # [1, 1, 100, 1, 1]
    otf = torch.fft.fftn(otf, dim=(2))                      # [1, 1, 100, 1, 1]
    # n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    # otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    return otf


def ps2ot(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = ps2ot(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.

    Args:
        psf: NxCxhxw
        shape: [H, W]

    Returns:
        otf: NxCxHxWx2
    '''
    # print(f'psf : {psf.shape}, shape:{shape}')

    otf = torch.zeros(psf.shape[:2] + shape).type_as(psf)               # [1, 1, 100, 256, 256]
    # print(f'otf : {otf.shape}')
    otf[:, :, :psf.shape[2], :psf.shape[3], :psf.shape[4]].copy_(psf)   # [1, 1, 100, 256, 256]
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)         # [1, 1, 100, 256, 256]
    otf = torch.fft.fftn(otf, dim=(-3, -2, -1))                         # [1, 1, 100, 256, 256]
    # n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    # otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    return otf


def upsample3D(x, sf=(5,4,4)):
    '''s-fold upsampler

    Upsampling the spatial size by filling the new entries with zeros

    x: tensor image, NxCxWxH
    '''
    st = 0
    b, c, t, h, w = x.shape
    z = torch.zeros((b, c, t*sf[0], h*sf[1], w*sf[2])).type_as(x)
    z[:, :, st::sf[0], st::sf[1], st::sf[2]].copy_(x)                  # 
    return z


def downsample(x, sf=3):
    '''s-fold downsampler

    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

    x: tensor image, NxCxWxH
    '''
    st = 0
    return x[..., st::sf, st::sf]


def downsample3D(x, sf=4):
    '''s-fold downsampler

    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

    x: tensor image, NxCxWxH
    '''
    st = 0
    return x[:, :, :, st::sf, st::sf]


def compute_flow(lqs, spynet_pretrained, cpu_cache):
    """Compute optical flow using SPyNet for feature alignment.

    Note that if the input is an mirror-extended sequence, 'flows_forward'
    is not needed, since it is equal to 'flows_backward.flip(1)'.

    Args:
        lqs (tensor): Input low quality (LQ) sequence with
            shape (n, t, c, h, w).

    Return:
        tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
            flows used for forward-time propagation (current to previous).
            'flows_backward' corresponds to the flows used for
            backward-time propagation (current to next).
    """

    n, t, c, h, w = lqs.size()
    lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
    lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)
    
    spynet = SPyNet(pretrained=spynet_pretrained)

    flows_backward = spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)
    flows_forward = spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

    if cpu_cache:
        flows_backward = flows_backward.cpu()
        flows_forward = flows_forward.cpu()

    return flows_forward, flows_backward

"""
# --------------------------------------------
# (1) Prior module; ResUNet: act as a non-blind denoiser
# x_k = P(z_k, beta_k)
# --------------------------------------------
"""
class ResUNet(nn.Module):
    def __init__(self, in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv',
                 upsample_mode='convtranspose'):
        super(ResUNet, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(
            *[B.ResBlock(nc[0], nc[0], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(
            *[B.ResBlock(nc[1], nc[1], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(
            *[B.ResBlock(nc[2], nc[2], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body = B.sequential(
            *[B.ResBlock(nc[3], nc[3], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'),
                                  *[B.ResBlock(nc[2], nc[2], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'),
                                  *[B.ResBlock(nc[1], nc[1], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'),
                                  *[B.ResBlock(nc[0], nc[0], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x):

        b, c, t, h, w = x.shape                     # [B, 4, 5, 256, 256]
        x = x.permute(0,2,1,3,4).view(-1, c, w, h)  # [B*5, 4, 256, 256]

        paddingBottom = int(np.ceil(h / 8) * 8 - h)
        paddingRight = int(np.ceil(w / 8) * 8 - w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)

        x = x[..., :h, :w]
        x = x.view(b, t, -1, h, w).permute(0,2,1,3,4)

        return x


"""
# --------------------------------------------
# (2) Data module, closed-form solution
# It is a trainable-parameter-free module  ^_^
# z_k = D(x_{k-1}, s, k, y, alpha_k)
# some can be pre-calculated
# --------------------------------------------
"""
class DataNet3D(nn.Module):
    def __init__(self):
        super(DataNet3D, self).__init__()

    def forward(self, x, FB, FBC, F2B, FBFy, alpha, sf):
        
        # print(x.device, FB.device, FBC.device, F2B.device, FBFy.device, alpha.device)
        FR = FBFy + torch.fft.fftn(alpha * x, dim=(2,3,4))              # [1, 3, 100, 256, 256]
        x1 = FB.mul(FR)                                                 # [1, 3, 100, 256, 256]
        if sf == (1, 1, 1):
            FBR = splits3D(x1, sf).squeeze(-1)
            invW = splits3D(F2B, sf).squeeze(-1)
        else:
            FBR = torch.mean(splits3D(x1, sf), dim=-1, keepdim=False)       # [1, 3, 20, 256, 256]
            invW = torch.mean(splits3D(F2B, sf), dim=-1, keepdim=False)     # [1, 1, 20, 1, 1]
        invWBR = FBR.div(invW + alpha)                                  # [1, 3, 20, 256, 256]
        FCBinvWBR = FBC * invWBR.repeat(1, 1, sf[0], sf[1], sf[2])      # [1, 3, 100, 256, 256]
        FX = (FR - FCBinvWBR) / alpha                                   # [1, 3, 100, 256, 256]
        Xest = torch.real(torch.fft.ifftn(FX, dim=(2,3,4)))             # [1, 3, 100, 256, 256]

        return Xest

"""
# --------------------------------------------
# (3) Hyper-parameter module
# --------------------------------------------
"""
class HyPaNet(nn.Module):
    def __init__(self, in_nc=2, out_nc=8, channel=64):
        super(HyPaNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv3d(in_nc, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, out_nc, 1, padding=0, bias=True),
            nn.Softplus())
        
    def forward(self, x):
        x = self.mlp(x) + 1e-6
        return x


"""
# --------------------------------------------
# main network
# --------------------------------------------
"""
class DAVSRNet(nn.Module):
    def __init__(self, n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R',
                 downsample_mode='strideconv', upsample_mode='convtranspose',
                 img_channels=3, mid_channels=64, num_blocks=7, max_residue_magnitude=10, is_low_res_input=True,
                 spynet_pretrained=None, vsr_pretrained=None, cpu_cache_length=100,
                 interpolation_mode='nearest', sigma_max=0, noise_level=10, sf=(2,4,4), fix_ab=0,
                 slomo_pretrained=None, pre_denoise_iters=0, use_cuda=True,
                 ):
        super(DAVSRNet, self).__init__()
        self.use_cuda = use_cuda
        self.d = DataNet3D()
        self.p = ResUNet(in_nc=in_nc, out_nc=out_nc, nc=nc, nb=nb, act_mode=act_mode, downsample_mode=downsample_mode, upsample_mode=upsample_mode)

        if fix_ab == 0:
            self.h = HyPaNet(in_nc=3, out_nc=n_iter * 2, channel=h_nc)
        # self.n = n_iter
        self.n = n_iter
        self.pre_denoise_iters = pre_denoise_iters
        if self.pre_denoise_iters:
            self.pre_vsr = BasicVSRPP(img_channels=img_channels, mid_channels=mid_channels, num_blocks=num_blocks, max_residue_magnitude=max_residue_magnitude,
                 is_low_res_input=is_low_res_input, spynet_pretrained=spynet_pretrained, vsr_pretrained=vsr_pretrained, cpu_cache_length=cpu_cache_length)

        self.vsr1 = BasicVSRPP(img_channels=img_channels, mid_channels=mid_channels, num_blocks=num_blocks, max_residue_magnitude=max_residue_magnitude,
                 is_low_res_input=True, spynet_pretrained=spynet_pretrained, vsr_pretrained=vsr_pretrained, cpu_cache_length=cpu_cache_length)
        self.vsr2 = BasicVSRPP(img_channels=img_channels, mid_channels=mid_channels, num_blocks=num_blocks, max_residue_magnitude=max_residue_magnitude,
                 is_low_res_input=True, spynet_pretrained=spynet_pretrained, vsr_pretrained=vsr_pretrained, cpu_cache_length=cpu_cache_length)

        self.upsample_last1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample_last2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)

        self.conv_upsample = nn.Sequential(
                                            nn.Conv2d(3, 64, 3, 1, 1),
                                            nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                            nn.Conv2d(64, 64, 3, 1, 1)
                                            ) 

        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # self.pixel_unshuffle = nn.PixelUnshuffle(3)
        self.interpolation_mode = interpolation_mode
        self.cpu_cache_length = cpu_cache_length

        if interpolation_mode == 'flow':
            self.spynet_pretrained = spynet_pretrained
            # optical flow
            self.spynet = SPyNet(pretrained=spynet_pretrained)
            # check if the sequence is augmented by flipping
            self.is_mirror_extended = False

        self.sigma_max = sigma_max
        self.noise_level = noise_level

        self.img_channels = img_channels

        self.sf = sf
        self.fix_ab = fix_ab

        # data and slomo
        mean = [0.429, 0.431, 0.397]
        mea0 = [-m for m in mean]
        std = [1] * 3
        self.trans_forward = transforms.Compose([transforms.Normalize(mean=mean, std=std)])
        self.trans_backward = transforms.Compose([transforms.Normalize(mean=mea0, std=std)])
        self.slomo_pretrained = slomo_pretrained

    def compute_flow(self, lqs):
        """Compute optical flow using SPyNet for feature alignment.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)
        if self.cpu_cache:
            flows_backward = flows_backward.cpu()
            flows_forward = flows_forward.cpu()

        return flows_forward, flows_backward

    def interpolate_batch(self, frame0, frame1, factor, flow, interp, back_warp):

        # frame0 = torch.stack(frames[:-1])
        # frame1 = torch.stack(frames[1:])

        i0 = frame0
        i1 = frame1
        ix = torch.cat([i0, i1], dim=1)
        # ix = ix.cuda()
        flow_out = flow(ix)
        f01 = flow_out[:, :2, :, :]
        f10 = flow_out[:, 2:, :, :]

        frame_buffer = []
        for i in range(1, factor):
            t = i / factor
            temp = -t * (1 - t)
            co_eff = [temp, t * t, (1 - t) * (1 - t), temp]

            ft0 = co_eff[0] * f01 + co_eff[1] * f10
            ft1 = co_eff[2] * f01 + co_eff[3] * f10

            gi0ft0 = back_warp(i0, ft0)
            gi1ft1 = back_warp(i1, ft1)

            iy = torch.cat((i0, i1, f01, f10, ft1, ft0, gi1ft1, gi0ft0), dim=1)
            io = interp(iy)

            ft0f = io[:, :2, :, :] + ft0
            ft1f = io[:, 2:4, :, :] + ft1
            vt0 = F.sigmoid(io[:, 4:5, :, :])
            vt1 = 1 - vt0

            gi0ft0f = back_warp(i0, ft0f)
            gi1ft1f = back_warp(i1, ft1f)

            co_eff = [1 - t, t]

            ft_p = (co_eff[0] * vt0 * gi0ft0f + co_eff[1] * vt1 * gi1ft1f) / \
                (co_eff[0] * vt0 + co_eff[1] * vt1)

            frame_buffer.append(ft_p)

        return frame_buffer

    def forward(self, x, k=None, sf=(5,4,4), sigma=None):
        '''
        x: tensor, NxCxTxWxH
        k: tensor, Nx(1,3)Txwxh
        sf: integer, 3
        sigma: tensor, Nx1x1x1x1
        '''

        # reset sf by config
        sf = self.sf
        # initialization & pre-calculation

        b, t, c, h, w = x.shape  # [B, 20, 3, 64, 64]


        # for interence REDS4 testdataset

        # if h == 180:
        #     x = x.permute(0,2,1,3,4)
        #     x = x.view(-1,c,h,w)
        #     pad_h = 12
        #     reflect = nn.ZeroPad2d((0,0,0,pad_h))
        #     x = reflect(x)
        #     x= x.view(b,t,c,h+12,w).permute(0,2,1,3,4)
        # print(f'x : {x.device}')
        pad_h = 0
        pad_w = 0
        if not h % 32 == 0:
            x = x.view(-1,c,h,w)
            pad_h = 32 - (h % 32)
            # pad_h = 12
            reflect = nn.ZeroPad2d((0,0,0,pad_h))
            x = reflect(x)
            x= x.view(b,t,c,h + pad_h,w)
        b, t, c, h, w = x.shape  # [B, 20, 3, 64, 64]
        if not w % 32 == 0:
            x = x.view(-1,c,h,w)
            pad_w = 32 - (w % 32)
            reflect = nn.ZeroPad2d((0,pad_w,0,0))
            x = reflect(x)
            x= x.view(b,t,c,h,w+pad_w)

        # whether to cache the features in CPU
        self.cpu_cache = True if t > self.cpu_cache_length else False
        b, t, c, w, h = x.shape  # [B, 20, 3, 64, 64]
        x = x.permute(0,2,1,3,4)
        # unfolding        
        for i in range(self.n):
            if self.use_cuda:
                x = x.cuda()

            x = self.vsr1(x)
            if not self.training:
                if self.use_cuda:
                    x = x.cuda()
        b,c,t,w,h = x.shape
        if self.interpolation_mode == 'nearest':
            x = nn.functional.interpolate(x, scale_factor=sf, mode='nearest')   # [1, 3, 100, 256, 256]
        elif self.interpolation_mode == 'slomo':
            # model
            flow = slomo.UNet(6, 4)
            if self.use_cuda:
                flow = flow.cuda()
            interp = slomo.UNet(20, 5)
            if self.use_cuda:
                interp = interp.cuda()
            states = torch.load(self.slomo_pretrained, map_location='cpu')
            with torch.set_grad_enabled(False):
                flow.load_state_dict(states['state_dictFC'])
                interp.load_state_dict(states['state_dictAT'])
                if self.use_cuda:
                    back_warp = slomo.backWarp(h, w, "cuda")
                else:
                    back_warp = slomo.backWarp(h, w, "cpu")
            x = x.cuda()
            x0 = x.permute(0,2,1,3,4).view(-1,c,w,h)
            # print(f'x0 : {x0.device}')
            x0 = self.trans_forward(x0).view(b,t,c,w,h)
            # print(f'x0 : {x0.device}')

            frame0 = x0[:,:-1,:,:,:].reshape(-1,c,w,h)
            frame1 = x0[:,1:,:,:,:].reshape(-1,c,w,h)
            x_inter = self.interpolate_batch(frame0, frame1, sf[0], flow, interp, back_warp)
            x_inter = torch.stack(x_inter, dim=1).view(-1,c,w,h)  # [20, 3, 64, 64]
            x_inter = self.trans_backward(x_inter).view(b, t-1, sf[0]-1, c, w,h) # [b, 4, 5, 3, 64, 64]
            x0 = self.trans_backward(x0.view(-1,c,w,h)).view(b, t, c, w,h)

            out_x = []
            out_x.append(x0[:,0,:,:,:].unsqueeze(1).repeat(1,2,1,1,1))
            for i in range(t-1):
                out_x.append(x0[:,i,:,:,:].unsqueeze(1))
                out_x.append(x_inter[:,i,...])
                # x_inter = self.interpolate_batch(x0[:,i,:,:,:], x0[:,i+1,:,:,:], sf[0]+1, flow, interp, back_warp)
                # out_x.append(torch.stack(x_inter, dim=1))
            out_x.append(x0[:,-1,:,:,:].unsqueeze(1).repeat(1,3,1,1,1))
            x = torch.cat(out_x, dim=1)  # [b, 25, 3, 64, 64]

            # x = self.trans_backward(out_x).view(-1,c,w,h)

            x = x.view(-1,c,w,h)
            x = x.view(b,t * sf[0],c,w,h)
        x = x.permute(0,2,1,3,4)
        for i in range(1):

            if self.use_cuda:
                x = x.cuda()

            x = self.vsr2(x)

        x = x.permute(0,2,1,3,4)

        outputs = []
        for i in range(0, x.size(1)):
            if self.use_cuda:
                x = x.cuda()
            lqs = x[:,i,:,:,:]
            hr = self.conv_upsample(lqs)
            hr = self.lrelu(self.upsample_last1(hr))
            hr = self.lrelu(self.upsample_last2(hr))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            hr += self.img_upsample(lqs)
            # hr += lqs[:, i, :, :, :]
            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()
            outputs.append(hr)
        x = torch.stack(outputs, dim = 1)
        if not pad_h == 0:
            x = x[:,:,:,:-4*pad_h,:]
        if not pad_w == 0:
            x = x[:,:,:,:,:-4*pad_w]
        x = x.cuda()
        return x

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
