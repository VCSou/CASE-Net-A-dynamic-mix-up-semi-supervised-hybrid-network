'''
author: @Toby
function: Segmentation models using PFMs (pathology foundation models)
'''

# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import os
import sys
from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.layers import SwiGLUPacked
import timm
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def resolve_conch_builders():
    try:
        from conch_v1_5_config import ConchConfig
        from build_conch_v1_5 import build_conch_v1_5
        return ConchConfig, build_conch_v1_5
    except ModuleNotFoundError:
        pass

    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_dirs = [
        current_dir,
        os.path.dirname(current_dir),
        os.path.join(current_dir, "conch"),
        os.path.join(os.path.dirname(current_dir), "conch"),
    ]
    env_dir = os.environ.get("CONCH_CODE_DIR", "").strip()
    if env_dir:
        candidate_dirs.insert(0, env_dir)

    for candidate_dir in candidate_dirs:
        config_path = os.path.join(candidate_dir, "conch_v1_5_config.py")
        build_path = os.path.join(candidate_dir, "build_conch_v1_5.py")
        if not (os.path.exists(config_path) and os.path.exists(build_path)):
            continue
        if candidate_dir not in sys.path:
            sys.path.insert(0, candidate_dir)
        from conch_v1_5_config import ConchConfig
        from build_conch_v1_5 import build_conch_v1_5
        return ConchConfig, build_conch_v1_5

    raise ModuleNotFoundError(
        "Conch dependencies were not found. Please make sure 'conch_v1_5_config.py' and "
        "'build_conch_v1_5.py' are available in the current folder, its parent folder, a "
        "'conch' subfolder, or set CONCH_CODE_DIR to the folder containing them."
    )


def get_PFM_model(PFM_name, PFM_weights_path,frozen):
    if PFM_name == 'Gigapath':
        gig_config = {
        "architecture": "vit_giant_patch14_dinov2",
        "num_classes": 0,
        "num_features": 1536,
        "global_pool": "token",
        "model_args": {
        "img_size": 224,
        "in_chans": 3,
        "patch_size": 16,
        "embed_dim": 1536,
        "depth": 40,
        "num_heads": 24,
        "init_values": 1e-05,
        "mlp_ratio": 5.33334,
        "num_classes": 0}} 
        model = timm.create_model("vit_giant_patch14_dinov2", pretrained=False, **gig_config['model_args'])
        state_dict = torch.load(PFM_weights_path , map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        if frozen:
            for param in model.parameters():
                param.requires_grad = False
        return model
    
    elif PFM_name == 'UNI':
        model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
        model.load_state_dict(torch.load(PFM_weights_path, map_location="cpu"), strict=True)
        if frozen:
            for param in model.parameters():
                param.requires_grad = False
        return model
    
    # elif PFM_name == 'Digepath':
    #     Digepath_kwargs = {
    #     'model_name': f'vit_large_patch16_224',
    #     'img_size': 224,
    #     'patch_size': 16,
    #     'init_values': 1e-5,
    #     'num_classes': 0,
    #     'dynamic_img_size': True}
    #     model = timm.create_model(**Digepath_kwargs)
    #     state_dict = torch.load(PFM_weights_path, map_location="cpu")
    #     new_state_dict = OrderedDict({k.replace('backbone.', ''): v for k, v in state_dict['teacher'].items()})
    #     missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    #     if frozen:
    #         for param in model.parameters():
    #             param.requires_grad = False
    #     return model
    elif PFM_name == 'Virchow_v2':
        from timm.layers import SwiGLUPacked
        virchow_v2_config = {
        "img_size": 224,
        "init_values": 1e-5,
        "num_classes": 0,
        "mlp_ratio": 5.3375,
        "reg_tokens": 4,
        "global_pool": "",
        "dynamic_img_size": True}
        model = timm.create_model("vit_huge_patch14_224", pretrained=False,mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU,**virchow_v2_config)
        state_dict = torch.load(PFM_weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        if frozen:
            for param in model.parameters():
                param.requires_grad = False
        return model
    elif PFM_name == 'Conch_v1_5':
        ConchConfig, build_conch_v1_5 = resolve_conch_builders()
        conch_v1_5_config = ConchConfig()
        model = build_conch_v1_5(conch_v1_5_config, PFM_weights_path)
        if frozen:
            for param in model.parameters():
                param.requires_grad = False
        return model

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced_channels = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * attn


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True, dropout=0.0):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.se = SqueezeExcitation(out_channels)
        self.spatial = SpatialAttention()

    def forward(self, x):
        residual = self.proj(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.se(x)
        x = self.spatial(x)
        return x + residual


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=(1, 2, 4, 8), use_batchnorm=True):
        super().__init__()
        branches = []
        for rate in rates:
            if rate == 1:
                branches.append(
                    Conv2dReLU(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        padding=0,
                        use_batchnorm=use_batchnorm,
                    )
                )
            else:
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size=3,
                            padding=rate,
                            dilation=rate,
                            bias=not use_batchnorm,
                        ),
                        nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
                        nn.ReLU(inplace=True),
                    )
                )
        self.branches = nn.ModuleList(branches)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2dReLU(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                use_batchnorm=False,
            ),
        )
        self.project = Conv2dReLU(
            out_channels * (len(rates) + 1),
            out_channels,
            kernel_size=1,
            padding=0,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x):
        size = x.shape[-2:]
        pooled = self.global_pool(x)
        pooled = F.interpolate(pooled, size=size, mode="bilinear", align_corners=False)
        outputs = [branch(x) for branch in self.branches] + [pooled]
        return self.project(torch.cat(outputs, dim=1))


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
            scale=2,
            dropout=0.0,
    ):
        super().__init__()
        self.scale = scale
        self.conv1 = ResidualConvBlock(
            in_channels + skip_channels,
            out_channels,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        if skip is not None:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, emb_dim, decoder_channels, is_14=False, dropout=0.1, head_channels=256):
        super().__init__()
        self.decoder_channels = decoder_channels
        self.conv_more = Conv2dReLU(
            emb_dim,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        self.context = ASPP(head_channels, head_channels, rates=(1, 2, 4, 8), use_batchnorm=True)
        self.context_refine = ResidualConvBlock(
            head_channels,
            head_channels,
            use_batchnorm=True,
            dropout=dropout,
        )
        decoder_channels = self.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch, dropout=dropout) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        if is_14:
            blocks[-1] = DecoderBlock(in_channels[-1], out_channels[-1], skip_channels[-1], scale=1.75, dropout=dropout)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None, return_features=False):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        if h * w != n_patch:
            raise ValueError(f"PFM token count {n_patch} cannot be reshaped to a square feature map.")
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        x = self.context(x)
        x = self.context_refine(x)
        decoder_features = []
        for i, decoder_block in enumerate(self.blocks):
            skip = None
            x = decoder_block(x, skip=skip)
            decoder_features.append(x)
        if return_features:
            return x, decoder_features
        return x


class PFM_Seg_Model(nn.Module):
    def __init__(
            self,
            PFM_name,
            PFM_weights_path,
            emb_dim,
            frozen_PFM=True,
            img_size=224,
            num_classes=2,
            zero_head=False,
            vis=False,
            deep_supervision=False,
            decoder_dropout=0.1,
            decoder_channels=(128, 64, 32, 16),
            decoder_head_channels=256,
            feature_noise_std=0.0,
    ):
        super(PFM_Seg_Model, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.PFM_name = PFM_name
        self.img_size = img_size
        self.deep_supervision = deep_supervision
        self.decoder_channels = tuple(decoder_channels)
        self.feature_noise_std = feature_noise_std
        self.classifier = 'seg'
        # self.transformer = Transformer(config, img_size, vis)
        if PFM_name == 'Virchow_v2':
            self.decoder = DecoderCup(emb_dim, self.decoder_channels, is_14=True, dropout=decoder_dropout, head_channels=decoder_head_channels)
        else:
            self.decoder = DecoderCup(emb_dim, self.decoder_channels, dropout=decoder_dropout, head_channels=decoder_head_channels)
        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder_channels[-1],
            out_channels=num_classes,
            kernel_size=3,
        )
        if deep_supervision:
            self.auxiliary_heads = nn.ModuleList(
                [
                    SegmentationHead(in_channels=channels, out_channels=num_classes, kernel_size=3)
                    for channels in self.decoder_channels[:-1]
                ]
            )
        else:
            self.auxiliary_heads = None
        self.PFM_name = PFM_name
        self.transformer = get_PFM_model(PFM_name, PFM_weights_path, frozen_PFM)

    def forward(self, x):
        input_size = x.shape[-2:]
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        transformer_is_trainable = any(param.requires_grad for param in self.transformer.parameters())
        if transformer_is_trainable:
            if self.PFM_name == 'Virchow_v2':
                x = self.transformer(x)[:,5:,:]
            elif self.PFM_name == 'Conch_v1_5':
                x = self.transformer.trunk.forward_features(x)[:,1:,:]  # (B, n_patch, hidden)
            else:
                x = self.transformer.forward_features(x)[:,1:,:]  # (B, n_patch, hidden)
        else:
            with torch.no_grad():
                if self.PFM_name == 'Virchow_v2':
                    x = self.transformer(x)[:,5:,:]
                elif self.PFM_name == 'Conch_v1_5':
                    x = self.transformer.trunk.forward_features(x)[:,1:,:]  # (B, n_patch, hidden)
                else:
                    x = self.transformer.forward_features(x)[:,1:,:]  # (B, n_patch, hidden)
        if self.training and self.feature_noise_std > 0:
            x = x + torch.randn_like(x) * self.feature_noise_std
        features = None
        return_aux = self.deep_supervision and self.training
        decoder_output = self.decoder(x, features, return_features=return_aux)
        if return_aux:
            x, decoder_features = decoder_output
        else:
            decoder_features = None
            x = decoder_output
        logits = self.segmentation_head(x)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

        if return_aux:
            aux_logits = []
            for feature, head in zip(decoder_features[:-1], self.auxiliary_heads):
                aux = head(feature)
                aux_logits.append(aux)
            return {"out": logits, "aux": aux_logits}
        return logits

    def unfreeze_last_blocks(self, n_blocks=0):
        if n_blocks <= 0:
            return []

        unfreezed_names = []
        candidate_containers = []
        if hasattr(self.transformer, "trunk") and hasattr(self.transformer.trunk, "blocks"):
            candidate_containers.append(("transformer.trunk.blocks", self.transformer.trunk.blocks))
        if hasattr(self.transformer, "blocks"):
            candidate_containers.append(("transformer.blocks", self.transformer.blocks))

        for prefix, blocks in candidate_containers:
            for idx, block in enumerate(list(blocks)[-n_blocks:]):
                for name, param in block.named_parameters():
                    param.requires_grad = True
                    unfreezed_names.append(f"{prefix}.{len(blocks) - n_blocks + idx}.{name}")
            break
        return unfreezed_names

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)



if __name__ == '__main__':
    # 权重路径
    # Gigapath_weights_path = '/path/to/Gigapath_weights'
    UNI_weights_path = '/home/rainyfog/code/Extra_features/ckpt/UNI/pytorch_model.bin'
    # Digepath_weights_path = '/path/to/Digepath_weights'
    # Conch_v1_5_weights_path = '/Data/lingxt/Other_Pathology_Model/Conch_1_5_weights/'
    # Virchow_v2_weights_path = '/Data/lingxt/Other_Pathology_Model/Virchow_2_weights/pytorch_model.bin'
    
    # 创建分割模型输入（1,3,224,224）输出（1，num_classes,224,224）
    # GigaPath_Seg_Model = PFM_Seg_Model(PFM_name='Gigapath', PFM_weights_path=Gigapath_weights_path, emb_dim=1536, frozen_PFM=True,num_classes=20)
    # UNI_Seg_Model = PFM_Seg_Model(PFM_name='UNI', PFM_weights_path=UNI_weights_path, emb_dim=1024, frozen_PFM=True,num_classes=20)
    # DIgepath_Seg_Model = PFM_Seg_Model(PFM_name='Digepath', PFM_weights_path=Digepath_weights_path, emb_dim=1024, frozen_PFM=True,num_classes=20)
    # Virchow_v2_Seg_Model = PFM_Seg_Model(PFM_name='Virchow_v2', PFM_weights_path=Virchow_v2_weights_path, emb_dim=1280, frozen_PFM=True,num_classes=2)
    Conch_v1_5_Seg_Model = PFM_Seg_Model(PFM_name='UNI', PFM_weights_path=UNI_weights_path, emb_dim=1024, frozen_PFM=True,num_classes=2)
    
    # 模拟输入
    input_img = torch.randn(1, 3, 224, 224)
    
    # 进行分割预测
    # Gigapath_Seg_Ans = GigaPath_Seg_Model(input_img)
    # UNI_Seg_Ans = UNI_Seg_Model(input_img)
    # Digepath_Seg_Ans = DIgepath_Seg_Model(input_img)
    # Virchow_v2_Seg_Ans = Virchow_v2_Seg_Model(input_img)
    Conch_v1_5_Seg_Ans = Conch_v1_5_Seg_Model(input_img)
    
    # 打印输出
    # print(Gigapath_Seg_Ans.shape)
    # print(UNI_Seg_Ans.shape)
    # print(Digepath_Seg_Ans.shape)
    # print(Virchow_v2_Seg_Ans.shape)
    print(Conch_v1_5_Seg_Ans.shape)
