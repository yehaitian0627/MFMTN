# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import torch
import torch.nn as nn
from functools import partial

from sympy.codegen.cnodes import sizeof
from torch import Tensor
from typing import Optional
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat

from einops import rearrange
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math
import torch_dct as dct
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from mamba_ssm.modules.mamba_simple import Mamba
from models.csms6s import SelectiveScanMamba, SelectiveScanCore, SelectiveScanOflex
from models.transformer import EfficientTokenTransformer
from models.BranchFusion import BranchFusion

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


MODEL_PATH = 'your_model_path'
_MODELS = {
    "videomamba_t16_in1k": os.path.join(MODEL_PATH, "videomamba_t16_in1k_res224.pth"),
    "videomamba_s16_in1k": os.path.join(MODEL_PATH, "videomamba_s16_in1k_res224.pth"),
    "videomamba_m16_in1k": os.path.join(MODEL_PATH, "videomamba_m16_in1k_res224.pth"),
}


class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Block(nn.Module, mamba_init):
    def __init__(self,
                 scan_type=None,
                 group_type = None,
                 k_group = None,
                 LM=None,
                 dim=None,
                 dt_rank="auto",
                 d_state = None,
                 d_model = None,
                 ssm_ratio = None,
                 bimamba=None,
                 seq=False,
                 force_fp32=True,
                 dropout=0.0,
                 **kwargs):
        super().__init__()
        act_layer = nn.SiLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        self.force_fp32 = force_fp32
        self.seq = seq
        self.k_group = k_group
        self.group_type = group_type
        self.scan_type = scan_type
        d_inner = int(ssm_ratio * d_model)
        d_norm_inner = int(ssm_ratio * d_model/4)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(int(dim/4), d_norm_inner * 2, bias=bias, **kwargs)
        self.act: nn.Module = act_layer()
        self.forward_conv1d = nn.Conv1d(
            in_channels=d_norm_inner, out_channels=d_norm_inner, kernel_size=1
        )
        self.conv2d = nn.Conv2d(
            in_channels=d_inner, out_channels=d_inner, groups=d_inner,
            bias=True, kernel_size=(1, 1), **kwargs,
        )
        self.conv3d = nn.Conv3d(
            in_channels=d_inner, out_channels=d_inner, groups=d_inner,
            bias = True, kernel_size=(1, 1, 1), ** kwargs,
        )

        self.x_proj = [
            nn.Linear(d_norm_inner, (dt_rank + d_state * 2), bias=False, **kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        self.dt_projs = [
            self.dt_init(dt_rank, d_norm_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **kwargs)
            for _ in range(k_group)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A的size来源于这里
        self.A_logs = self.A_log_init(d_state, d_norm_inner, copies=k_group, merge=True)  # (K * D, N)
        self.Ds = self.D_init(d_norm_inner, copies=k_group, merge=True)  # (K * D)

        self.out_norm = nn.LayerNorm(d_norm_inner)
        self.out_proj = nn.Linear(d_norm_inner, int(dim/4), bias=bias, **kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, x: Tensor, xs: Tensor, route=None, SelectiveScan = SelectiveScanMamba):
        x = self.in_proj(x)  # d_inner=192  [10, 64, 96] -> [10, 64, 384]    [10, 8, 8, 96]->[10, 8, 8, 384]
        x, z = x.chunk(2, dim=-1)  # [10, 64, 192]  [10, 8, 8, 192]
        z = self.act(z)  # [10, 64, 192]   [10, 8, 8, 192]

        if self.group_type == 'Interval':
            x1_rearranged = rearrange(x, "b s d -> b d s").contiguous()  # [10, 192, 64]
            x = self.forward_conv1d(x1_rearranged)  # [10, 192, 64]
            x = self.act(x)

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            u = u.contiguous()
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, False)

        if len(x.size()) == 3:
            B, D, L = x.shape
        self.B = B
        self.L = L
        D, N = self.A_logs.shape  # D 768   N 16
        K, D, R = self.dt_projs_weight.shape  # 4   192    6
        self.xs = xs.unsqueeze(dim=1).permute(0, 1, 3, 2)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", self.xs, self.x_proj_weight) #[10, 2, 64, 64] einsum指定输入张量和输出张量之间的维度关系，你可以定义所需的运算操作

        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)  #[10, 2, 32, 64]  [10, 2, 16, 64]  [10, 2, 16, 64]
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)  #[10, 2, 192, 64]

        xs = self.xs.view(B, -1, L) # [10, 384, 64]  [10, 768, 64]
        dts = dts.contiguous().view(B, -1, L) # [10, 768, 64] .contiguous()是一个用于确保张量存储连续性
        Bs = Bs.contiguous()  #[10, 2, 16, 64]   [10, 4, 16, 64]
        Cs = Cs.contiguous()  #[10, 2, 16, 64]   [10, 4, 16, 64]

        As = -torch.exp(self.A_logs.float())   # [384, 16]  [768, 16]
        Ds = self.Ds.float() # (k * d)  [384]  [768]
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # [384]  [768]

        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        if self.force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        if self.seq:
            out_y = []
            for i in range(self.k_group):
                yi = selective_scan(
                    xs.view(B, K, -1, L)[:, i], dts.view(B, K, -1, L)[:, i],  #xs[1, 192, 3136]  dts[1, 192, 3136]
                    As.view(K, -1, N)[i], Bs[:, i].unsqueeze(1), Cs[:, i].unsqueeze(1), Ds.view(K, -1)[i],  #As[192, 16]  Bs/Cs[1, 1, 16, 3136] Ds[192]
                    delta_bias=dt_projs_bias.view(K, -1)[i],   #[192]
                    delta_softplus=True,
                ).view(B, -1, L)  #[1, 192, 3136]
                out_y.append(yi)
            out_y = torch.stack(out_y, dim=1)  #[1, 4, 192, 3136]
        else:
            out_y = selective_scan(
                xs, dts,
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, K, -1, L)  #[10, 384, 64]->[10, 2, 192, 64]    [10, 4, 192, 64]
        assert out_y.dtype == torch.float

        if out_y.size(1) == 2:
            y = out_y[:, 0] + torch.flip(out_y[:, 1], dims=[-1])  #[10, 192, 64]
            y = y.transpose(dim0=1, dim1=2).contiguous() # [10, 64, 192]
            y = self.out_norm(y)  #[10, 64, 192]
        else:
            y = out_y.squeeze(dim=1).permute(0, 2, 1).contiguous()
            y = self.out_norm(y)  # [10, 64, 192]

        y = y * z   #[10, 64, 192]   [10, 8, 8, 192]
        out = self.dropout(self.out_proj(y))  #[10, 64, 96]   [10, 8, 8, 96]

        return out

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

class MLP_Block(nn.Module):
    def __init__(self, in_features, hidden_features, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0. else nn.Identity(),
            nn.Linear(hidden_features, in_features)
        )

    def forward(self, x):
        return self.mlp(x)

class VisionMamba(nn.Module):
    def __init__(
            self,
            model_type=None,
            k_group=None,
            depth=None,
            embed_dim=None,
            d_state: int = None,
            ssm_ratio: int = None,
            num_classes: int = None,
            drop_rate=0.,
            drop_path_rate=0.1,
            fused_add_norm=False,
            residual_in_fp32=True,
            bimamba=True,
            # video
            fc_drop_rate=0.,
            # checkpoint
            use_checkpoint=False,
            checkpoint_num=0,
            Pos_Cls = False,
            pos: str = None,
            cls: str = None,
            conv3D_channel: int = None,
            conv3D_kernel_1: int = None,
            conv3D_kernel_2: int = None,
            conv3D_kernel_3: int = None,
            dim_patch: int = None,
            dim_linear_1: int = None,
            dim_linear_2: int = None,
            dim_linear_3: int = None,
            **kwargs,
        ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        self.Pos_Cls = Pos_Cls
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.k_group = k_group
        self.depth = depth
        self.model_type = model_type

        self.conv3d_features_1 = nn.Sequential(
            nn.Conv3d(1, out_channels=conv3D_channel, kernel_size=conv3D_kernel_1),
            nn.BatchNorm3d(conv3D_channel),
            nn.ReLU(),
        )
        self.conv3d_features_2 = nn.Sequential(
            nn.Conv3d(1, out_channels=conv3D_channel, kernel_size=conv3D_kernel_2),
            nn.BatchNorm3d(conv3D_channel),
            nn.ReLU(),
        )
        self.conv3d_features_3 = nn.Sequential(
            nn.Conv3d(1, out_channels=conv3D_channel, kernel_size=conv3D_kernel_3),
            nn.BatchNorm3d(conv3D_channel),
            nn.ReLU(),
        )
        self.embedding_spatial_1 = nn.Sequential(nn.Linear(conv3D_channel * dim_linear_1, embed_dim))
        self.embedding_spatial_2 = nn.Sequential(nn.Linear(conv3D_channel * dim_linear_2, embed_dim))
        self.embedding_spatial_3 = nn.Sequential(nn.Linear(conv3D_channel * dim_linear_3, embed_dim))

        self.norm = nn.LayerNorm(int(embed_dim/4))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1792 + 1, self.embed_dim))
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, 28, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.head_drop = nn.Dropout(fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.transformer = EfficientTokenTransformer(channels=32)  # PCA

        self.FFN = nn.ModuleList([Residual(
                LayerNormalize(
                embed_dim, MLP_Block(embed_dim, embed_dim, dropout=drop_path_rate)))
                for i in range(depth)])
        self.bf = BranchFusion(fusion_type='gate', channels=embed_dim)
        self.layers = nn.ModuleList([Block(
                scan_type='Interval',
                group_type='Interval',
                k_group=1,
                dim=embed_dim,
                d_state=d_state,
                d_model=embed_dim,
                ssm_ratio=ssm_ratio,
                bimamba=bimamba,
                **kwargs, )
                for i in range(depth)])

    def get_num_layers(self):
        return len(self.layers)

    def scan(self, x, scan_type=None, group_type=None):
        x = rearrange(x, 'b c t h w -> b (c t) h w')  # [10, 896, 8, 8]
        x = rearrange(x, 'b c h w -> b h w c')  # [64, 896, 8, 8]-> [10, 8, 8, 896]
        return x


    def mask_generate(self, img_size, num):
        out = []
        mask_none = torch.ones(img_size, img_size)
        for i in reversed(range(0, num - 1)):
            mask = torch.ones(img_size // (2 ** i), img_size // (2 ** i))
            mask = torch.triu(mask, diagonal=0)
            mask = torch.rot90(mask, k=1, dims=(0, 1))
            out.append(
                torch.nn.functional.pad(mask, (0, img_size - img_size // (2 ** i), 0, img_size - img_size // (2 ** i)),
                                        mode='constant', value=0))
        out.append(mask_none)
        return out

    def diagonal_zigzag_scan(self, x):  # x: [B, C, W, H]
        B, C, W, H = x.shape
        device = x.device
        idx_list = []

        for s in range(W + H - 1):
            for i in range(s + 1):
                j = s - i
                if i < W and j < H:
                    idx_list.append(i * H + j)

        idx = torch.tensor(idx_list, dtype=torch.long, device=device)  # [W*H]
        x_flat = x.view(B, C, -1)  # [B, C, W*H]
        x_zigzag = torch.index_select(x_flat, dim=2, index=idx)  # [B, C, W*H]
        return x_zigzag

    def interleaved_column_scan(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform interleaved column scan on input tensor.

        Args:
            x: Tensor of shape [B, C, W, H]

        Returns:
            Tensor of shape [B, C, W*H], scanned in interleaved column order.
        """
        B, C, W, H = x.shape
        device = x.device

        idx_list = []
        # Step 1: Even columns
        for col in range(0, H, 2):
            for row in range(W):
                idx_list.append(row * H + col)
        # Step 2: Odd columns
        for col in range(1, H, 2):
            for row in range(W):
                idx_list.append(row * H + col)

        idx = torch.tensor(idx_list, dtype=torch.long, device=device)  # [W*H]
        x_flat = x.view(B, C, -1)  # [B, C, W*H]
        x_scan = torch.index_select(x_flat, dim=2, index=idx)  # [B, C, W*H]

        return x_scan

    def interleaved_row_priority_scan(self, x: torch.Tensor, row_stride: int, col_stride: int) -> torch.Tensor:
        B, C, W, H = x.shape
        device = x.device
        idx_list = []

        # Primary row scan: row 0, 0 + row_stride, ...
        primary_rows = list(range(0, W, row_stride))
        secondary_rows = [r for r in range(W) if r not in primary_rows]

        def scan_row(r):
            # First: every col_stride-th pixel in row
            for offset in range(col_stride):
                for c in range(offset, H, col_stride):
                    idx_list.append(r * H + c)

        for r in primary_rows:
            scan_row(r)

        for r in secondary_rows:
            scan_row(r)

        idx = torch.tensor(idx_list, dtype=torch.long, device=device)
        x_flat = x.view(B, C, -1)
        x_scan = torch.index_select(x_flat, dim=2, index=idx)
        return x_scan

    def checkerboard_multi_pass(self, x):
        B, C, W, H = x.shape
        device = x.device

        grid = torch.arange(W * H, device=device).view(W, H)
        even_mask = (torch.arange(W, device=device).view(-1, 1) + torch.arange(H, device=device).view(1, -1)) % 2 == 0
        pass1_idx = grid[even_mask].flatten()
        pass2_idx = grid[~even_mask].flatten()
        idx = torch.cat([pass1_idx, pass2_idx], dim=0)

        x_flat = x.view(B, C, -1)
        x_check = torch.index_select(x_flat, dim=2, index=idx)
        return x_check

    def x_scan(self, x, scan_type='Interval', group_type=None, route=None):
        x = x.permute(0, 3, 1, 2)
        B,C,W,H = x.shape
        self.B = B
        self.L = W * H  # mamba
        if scan_type == 'Interval':
            x1 = x[:, 0::4, :, :]
            x2 = x[:, 1::4, :, :]
            x3 = x[:, 2::4, :, :]
            x4 = x[:, 3::4, :, :]
            xs1 = x1.view(self.B, -1, self.L).view(self.B, 1, -1, self.L)
            xs2 = torch.transpose(x2, dim0=2, dim1=3).contiguous().view(self.B, -1, self.L).view(self.B, 1, -1, self.L)
            xs3 = self.interleaved_row_priority_scan(x3, row_stride=4, col_stride=4).view(self.B, 1, -1, self.L)
            xs4 = self.checkerboard_multi_pass(x4).view(self.B, 1, -1, self.L)

            xs = torch.stack([xs1, xs2, xs3, xs4], dim=1).view(self.B, 4, -1, self.L)
        return xs

    def forward_features(self, x, inference_params=None, xss=None):
        x_1 = self.conv3d_features_1(x)
        x_2 = self.conv3d_features_1(x)
        x_3 = self.conv3d_features_3(x)
        x_1 = self.scan(x_1)
        x_2 = self.scan(x_2)
        x_3 = self.scan(x_3)
        x_1 = self.embedding_spatial_1(x_1)
        x_2 = self.embedding_spatial_1(x_2)
        x_3 = self.embedding_spatial_3(x_3)
        x1 = self.pos_drop(x_1)
        x2 = self.pos_drop(x_2)
        x3 = self.pos_drop(x_3)

        xs1 = self.x_scan(x1)
        xs2 = self.x_scan(x2)
        xs3 = self.x_scan(x3)

        xss = [None] * 4
        for t in range(4):
            xss[t] = torch.cat((xs1[:, t , :, :], xs2[:, t, :, :], xs3[:, t, :, :]), dim=-1)
            xss[t] = xss[t].squeeze(dim=1).permute(0, 2, 1)

        if self.model_type == 'Parallel MT' or self.model_type == 'Interval MT' or self.model_type == 'Series MT' or self.model_type == 'Series TM':

            LG_1 = rearrange(x1, 'b h w c-> b (h w) c')
            LG_2 = rearrange(x2, 'b h w c-> b (h w) c')
            LG_3 = rearrange(x3, 'b h w c-> b (h w) c')

            LG = torch.cat([LG_1, LG_2, LG_3], dim=1)

            LM_1 = xss[0]
            LM_2 = xss[1]
            LM_3 = xss[2]
            LM_4 = xss[3]
            if self.model_type == 'Parallel MT':
                for i in range(self.depth):
                    LG_1 = self.transformer(LG[:, 0:LG_1.shape[1], :])  # [64, 121, 32]
                    LG_2 = self.transformer(LG[:, LG_1.shape[1]:LG_1.shape[1] + LG_2.shape[1], :])
                    LG_3 = self.transformer(LG[:, LG_1.shape[1] + LG_2.shape[1]:, :])
                    LG_T = torch.cat([LG_1, LG_2, LG_3], dim=1)  # [64, 251, 32]

                    LG_T = self.FFN[i](LG_T)

                    LG_M1 = LM_1 + self.drop_path(self.layers[i](self.norm(LM_1),LM_1))  # LM_1:[64, 305, 8]
                    LG_M2 = LM_2 + self.drop_path(self.layers[i](self.norm(LM_2),LM_2))
                    LG_M3 = LM_3 + self.drop_path(self.layers[i](self.norm(LM_3),LM_3))
                    LG_M4 = LM_4 + self.drop_path(self.layers[i](self.norm(LM_4),LM_4))
                    LG_M = torch.cat([LG_M1, LG_M2, LG_M3, LG_M4], dim=2)
                    LG = self.bf(LG_M, LG_T)

        feature = LG.mean(dim=1)

        return feature

    def forward(self, x, inference_params=None):
        feature = self.forward_features(x, inference_params)  ##[10, 192]
        x = self.head(self.head_drop(feature))   #[10, 9]
        return x, feature
