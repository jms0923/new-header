import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init, MaxPool2d

from mmdet.core import multi_apply
from ..builder import HEADS
from .anchor_head import AnchorHead


@HEADS.register_module()
class WanBaseHead(AnchorHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        
        super(WanBaseHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.p_cls_convs = nn.ModuleList()
        self.n_cls_convs = nn.ModuleList()
        self.w_cls_convs = nn.ModuleList()
        self.p_reg_convs = nn.ModuleList()
        self.n_reg_convs = nn.ModuleList()
        self.w_reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.n_cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.n_reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            
            self.w_cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    dilation=3,
                    padding=3,
                    conv_cfg=self.conv_cfg,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='BN', requires_grad=True)))
            self.w_reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    dilation=3,
                    padding=3,
                    conv_cfg=self.conv_cfg,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='BN', requires_grad=True)))

            self.p_cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    1,
                    stride=1,
                    padding=0,
                    conv_cfg=self.conv_cfg,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='BN', requires_grad=True)))
            self.p_reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    1,
                    stride=1,
                    padding=0,
                    conv_cfg=self.conv_cfg,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='BN', requires_grad=True)))

        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.n_cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.p_cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.w_cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.n_reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.p_reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.w_reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * 4.
        """
        return multi_apply(self.forward_single, feats)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_feat = x
        reg_feat = x
        for p_cls_conv, n_cls_conv, w_cls_conv in zip(self.p_cls_convs, self.n_cls_convs, self.w_cls_convs):
            identity = cls_feat
            p_cls_feat = cls_feat
            n_cls_feat = cls_feat
            w_cls_feat = cls_feat
            
            p_cls_feat = p_cls_conv(p_cls_feat)
            n_cls_feat = n_cls_conv(n_cls_feat)
            w_cls_feat = w_cls_conv(w_cls_feat)
            
            cls_feat = p_cls_feat + n_cls_feat + w_cls_feat
            cls_feat += identity
            cls_feat = self.relu(cls_feat)
            
        for p_reg_conv, n_reg_conv, w_reg_conv in zip(self.p_reg_convs, self.n_reg_convs, self.w_reg_convs):
            identity = reg_feat
            p_reg_feat = reg_feat
            n_reg_feat = reg_feat
            w_reg_feat = reg_feat
            
            p_reg_feat = p_reg_conv(p_reg_feat)
            n_reg_feat = n_reg_conv(n_reg_feat)
            w_reg_feat = w_reg_conv(w_reg_feat)
            
            reg_feat = p_reg_feat + n_reg_feat + w_reg_feat
            reg_feat += identity
            reg_feat = self.relu(reg_feat)
            
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred
