import torch.nn as nn
import torch
import torch.nn.functional as F

class BranchFusion(nn.Module):
    def __init__(self, fusion_type: str, channels: int):
        super().__init__()
        self.fusion_type = fusion_type
        if self.fusion_type == 'attn':
            self.fusion = nn.Sequential(
                nn.Linear(channels * 2, channels),
                nn.GELU(),
                nn.Linear(channels, channels)
            )
            self.attn = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),  # Pool along L dim
                nn.Conv1d(channels, channels // 4, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(channels // 4, channels, kernel_size=1),
                nn.Sigmoid()
            )

        elif self.fusion_type == 'gate':
             self.gate = nn.Sequential(
                # nn.Linear(channels * 2, channels),
                nn.GELU(),
                nn.Linear(channels, channels),
                nn.Sigmoid()
            )

        elif self.fusion_type == 'cat':
            self.cat = nn.Sequential(
                nn.Linear(channels * 2, channels),
                nn.GELU(),
                nn.Linear(channels, channels),
                nn.Sigmoid()
            )


    def forward(self, x1, x2):
        def interleave_channels_fast(a, b):
            B, L, C = a.shape
            # 扩展维度以便拼接
            a_expanded = a.unsqueeze(-1)  # [B, L, C, 1]
            b_expanded = b.unsqueeze(-1)  # [B, L, C, 1]
            # 拼接并重塑
            cat = torch.cat([a_expanded, b_expanded], dim=-1)  # [B, L, C, 2]
            interleaved = cat.reshape(B, L, 2 * C)
            return interleaved

        if self.fusion_type == 'attn':
            # x1, x2: [B, L, C]
            x = interleave_channels_fast(x1, x2)
            fused = self.fusion(x)  # [B, L, C]

            # Attention: convert to [B, C, L]
            attn = fused.transpose(1, 2)  # [B, C, L]
            attn = self.attn(attn)  # [B, C, 1]
            attn = attn.transpose(1, 2)  # [B, 1, C]

            out = fused * attn + fused  # [B, L, C]
            return out

        elif self.fusion_type == 'gate':
            # x = torch.cat([x1, x2], dim=-1)  # [B, L, 2C]
            # x = interleave_channels_fast(x1, x2)
            g = self.gate(x1)  # [B, L, C], in [0, 1]
            return g * x1 + (1 - g) * x2

        elif self.fusion_type == 'cat':
            # x1:mamba_Output x2:transformer_Output
            # x = x1
            x = x1 + x2
            # x = interleave_channels_fast(x1, x2)
            # x = self.cat(x)
            return x

        return None

