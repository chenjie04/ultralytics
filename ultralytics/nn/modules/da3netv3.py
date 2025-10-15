from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv
from .da3netv2 import ELANBlock

# 效果比我们提出的双轴聚合注意力差太多，可能是数量不够？
class EfficientMultiHeadAttention(nn.Module):
    """更高效的多头注意力实现，使用单个大矩阵"""
    
    def __init__(self, d_model, num_heads, dropout=0.0):
        super(EfficientMultiHeadAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads

        
        # 将Q、K、V的投影合并到一个线性层中
        self.d_k = d_model // 2
        self.d_v = d_model
        hidden_dim = self.d_k * 2 + self.d_v
        self.qkv_proj = nn.Linear(d_model,  hidden_dim)
    
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x: torch.Tensor):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # 确保输入张量与模型参数的数据类型一致
        if x.dtype != self.qkv_proj.weight.dtype:
            x = x.to(self.qkv_proj.weight.dtype)
        
        # 一次性计算Q、K、V
        qkv = self.qkv_proj(x)  # [batch_size, seq_len, hidden_dim]
        q, k, v = qkv[:, :, :self.d_k], qkv[:, :, self.d_k:self.d_k*2], qkv[:, :, -self.d_v:]
        q, k, v = q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2), \
                  k.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2), \
                  v.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        
        
        # 计算注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return output

class SinePositionalEncoding(nn.Module):
    """Position encoding with sine and cosine functions.

    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None
    """

    def __init__(self,
                 num_feats: int,
                 temperature: int = 10000,
                 normalize: bool = False,
                 scale: float = 2 * math.pi,
                 eps: float = 1e-6,
                 offset: float = 0.) -> None:
        super().__init__()
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask: torch.Tensor, input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward function for `SinePositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].
            input (Tensor, optional): Input image/feature Tensor.
                Shape [bs, c, h, w]

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        assert not (mask is None and input is None)

        if mask is not None:
            B, H, W = mask.size()
            device = mask.device
            # For convenience of exporting to ONNX,
            # it's required to convert
            # `masks` from bool to int.
            mask = mask.to(torch.int)
            not_mask = 1 - mask  # logical_not
            y_embed = not_mask.cumsum(1, dtype=torch.float32)
            x_embed = not_mask.cumsum(2, dtype=torch.float32)
        else:
            # single image or batch image with no padding
            B, _, H, W = input.shape
            device = input.device
            x_embed = torch.arange(
                1, W + 1, dtype=torch.float32, device=device)
            x_embed = x_embed.view(1, 1, -1).repeat(B, H, 1)
            y_embed = torch.arange(
                1, H + 1, dtype=torch.float32, device=device)
            y_embed = y_embed.view(1, -1, 1).repeat(B, 1, W)
        if self.normalize:
            y_embed = (y_embed + self.offset) / \
                      (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self) -> str:
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'temperature={self.temperature}, '
        repr_str += f'normalize={self.normalize}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'eps={self.eps})'
        return repr_str

class DualAxisAggAttn_v3(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 4,
        middle_ratio: float = 0.5,
    ):
        super().__init__()
        self.channels = in_channels
        self.groups = groups
        middle_channels = int(in_channels * middle_ratio)
        self.middle_channels = middle_channels

        self.main_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)

        self.positional_encoding = SinePositionalEncoding(num_feats=middle_channels//2)
        self.attn_w = EfficientMultiHeadAttention(
            d_model=middle_channels, num_heads=groups
        )
        self.attn_h = EfficientMultiHeadAttention(
            d_model=middle_channels, num_heads=groups
        )

        self.conv_fusion = nn.ModuleDict(
            {
                "W": Conv(
                    c1=middle_channels,
                    c2=middle_channels,
                    k=3,
                    g=middle_channels,
                    act=True,
                ),
                "H": Conv(
                    c1=middle_channels,
                    c2=middle_channels,
                    k=3,
                    g=middle_channels,
                    act=True,
                ),
            }
        )

        final_channels = int(2 * middle_channels)
        self.out_project = Conv(c1=final_channels, c2=out_channels, k=1, act=True)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [B, C, H, W]

        Returns:
            torch.Tensor: 输出张量，形状为 [B, C, H, W]
        """
        B, C, H, W = x.shape
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)

        # 位置编码
        masks = None
        # [batch_size, embed_dim, h, w]
        pos_embed = self.positional_encoding(masks, input=x_main)
        x_main = x_main + pos_embed

        
        # 宽轴注意力
        x_main = x_main.permute(0, 2, 3, 1).contiguous().view(B*H, W, self.middle_channels)
        x_W = self.attn_w(x_main)
        x_W = x_W.view(B, H, W, self.middle_channels).permute(0, 3, 1, 2).contiguous()
        x_W_fused = self.conv_fusion["W"](x_W) + x_W
        # 高轴注意力
        x_W_fused = x_W_fused.permute(0, 3, 2, 1).contiguous().view(B*W, H, self.middle_channels)
        x_H = self.attn_h(x_W_fused)
        x_H = x_H.view(B, W, H, self.middle_channels).permute(0, 3, 2, 1).contiguous()
        x_H_fused = self.conv_fusion["H"](x_H) + x_H

        x_out = torch.cat([x_H_fused, x_short], dim=1)

        x_out = self.out_project(x_out)

        return x_out

class DA3Block_v3(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        attn_groups: int = 4,
    ) -> None:
        super().__init__()

        self.attn = DualAxisAggAttn_v3(
            in_channels=in_channels, out_channels=out_channels, groups=attn_groups
        )
        self.local_extractor = ELANBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.attn(x) + residual
        residual = x
        x = self.local_extractor(x) + residual

        return x