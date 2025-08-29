import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from einops import rearrange

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)),
            nn.Conv3d(1, 1, kernel_size=1),  # 保持轻量，也可移除
            nn.BatchNorm3d(1),
            nn.GELU()
        )
        # 3d卷积后将维度压缩至__维
        self.out_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)  # 最后投影回通道

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        x = x.view(B, C, H, W).permute(0, 2, 3, 1)        # [B, H, W, C]
        x = x.unsqueeze(1).permute(0, 1, 4, 2, 3)         # [B, 1, C, H, W]
        x = self.block(x)                                 # [B, 1, C, H/2, W/2]
        x = x.squeeze(1).permute(0, 2, 3, 1).contiguous() # [B, H/2, W/2, C]
        H_new, W_new, C_new = x.shape[1], x.shape[2], x.shape[3]
        x = x.view(B, -1, C_new).transpose(1, 2)              # [B, C, HW]
        x = self.out_proj(x).transpose(1, 2)              # [B, HW, C']
        return x.permute(0, 2, 1), H_new, W_new


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)

        self.nn1 = nn.Linear(dim, dim)
        self.do1 = nn.Dropout(dropout)
        self.sr = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.head_dim = dim // heads

        # 后加的
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.spe = nn.Conv2d(dim, dim, kernel_size=1, padding=0, groups=self.head_dim, bias=False)
        self.bnc1 = nn.BatchNorm2d(dim)
        self.local = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=self.head_dim,
                               bias=False)

    def forward(self, x, mask=None, use_SR=False):
        x = x.permute(0, 2, 1).contiguous()
        b, n, d, h = *x.shape, self.heads
        patch_size = int(math.sqrt(n))
        s = int((n-1) ** 0.5)
        c = x[:,0,:].reshape(b,1,d)
        f = x[:,1:,:]

        # sr k, v
        if use_SR==True:
            q = x.reshape(b, n, h, d // h).permute(0, 2, 1, 3) # 64, 8, 65, 8
            f_ = f.permute(0, 2, 1).reshape(b, d, s, s)
            f_ = self.sr(f_ )
            f_ = rearrange(f_, 'b h n d -> b h (n d)').permute(0, 2, 1) # .reshape(b, C, -1).permute(0, 2, 1)
            f_ = torch.cat((c, f_), dim=1)
            f_ = self.norm(f_)
            f_ = self.act(f_)
            kv = self.to_kv(f_).chunk(2, dim = -1)
            k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), kv)
        else:
            qkv = self.to_qkv(x).chunk(3, dim = -1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions



        # 消融实验：原始Transformer=================

        # 后加的S
        x = x.contiguous().view(b, self.dim, patch_size, patch_size)
        spe = self.spe(self.act(self.bn(x)))  # 构建S(Conv2d)
        spe = self.avg_pool(spe)
        spe = rearrange(spe, "b (h d) n w -> b h (n w) d", h=self.heads)  # 整理S维度
        # v * s
        v_spe = torch.einsum("b h i j, b h j d -> b h i d", v, spe)
        v_spe = v_spe * self.scale

        # 后加的C
        c = x
        c = self.act(self.bnc1(self.local(c)))
        c = rearrange(c, "b (d1 d2) h w -> b d1 (h w) d2", d1=self.heads)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper
        # 消融实验：使用原始v
        # out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = torch.einsum('bhij,bhjd->bhid', attn, v_spe+c)  # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out = self.nn1(out)
        out = self.do1(out)
        return out

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(in_channels)
        )
        self.expand_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.view(B, C, H, W)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.refine(x)
        x = self.expand_channels(x).contiguous()
        return x.view(B, -1, x.shape[1])

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        x = x.permute(0, 2, 1).contiguous()
        return self.fn(x, **kwargs) + x

class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        # x = x.permute(0, 2, 1)
        return self.fn(self.norm(x).permute(0,2,1), **kwargs)

class EfficientTokenTransformer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # self.down = DownsampleBlock(channels, channels)

        self.transformer = nn.ModuleList(
            [Residual(LayerNormalize(channels, Attention(channels, heads=8, dropout=0.1)))
             for i in range(1)])
        # self.transformer = TransformerBlock(channels)
        # self.up = UpsampleBlock(channels, channels * 2)

    def forward(self, x):
        # x: [B, C, N], hw: (H, W)
        # x, h_new, w_new = self.down(x)
        # x = self.transformer[0](x)
        # x = self.up(x)


        x = x.permute(0, 2, 1).contiguous()
        x = self.transformer[0](x)
        return x
