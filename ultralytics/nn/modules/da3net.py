
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


class MoVE(nn.Module):
    """
    MoVE: Multi-experts Convolutional Neural Network.

    """

    def __init__(
        self,
        channels: int,
        num_experts: int = 9,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.channels = channels
        self.num_experts = num_experts

        assert channels % num_experts == 0, "channels must be divisible by num_experts"

        # 并行化专家计算
        self.experts = Conv(
            channels, channels * num_experts, kernel_size, g=channels, act=True
        )

        # 跨通道融合
        self.fusion = Conv(
                channels * num_experts,
                channels,
                k=1,
                g=num_experts,
                act=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        expert_outputs = self.experts(x)  # (B, C*A, H, W)

        out = self.fusion(expert_outputs)

        return out
    

def channel_shuffle(x, groups):
    """Channel Shuffle operation.

    This function enables cross-group information flow for multiple groups
    convolution layers.

    Args:
        x (Tensor): The input tensor.
        groups (int): The number of groups to divide the input tensor
            in the channel dimension.

    Returns:
        Tensor: The output tensor after channel shuffle operation.
    """

    batch_size, num_channels, height, width = x.size()
    assert num_channels % groups == 0, "num_channels should be " "divisible by groups"
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x


class MoVE_GhostModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        main_kernel_size: int = 3,  # 主分支的卷积核大小
        num_experts: int = 16,  # 轻量分支专家数量
        cheap_kernel_size: int = 3,
    ):
        super().__init__()

        self.middle_channels = int(in_channels // 2)
        self.primary_conv = Conv(
            in_channels, self.middle_channels, k=main_kernel_size, act=True
        )

        self.cheap_operation = MoVE(
            self.middle_channels,
            num_experts,
            kernel_size=cheap_kernel_size,
        )

        self.out_project = Conv(self.middle_channels * 2, out_channels, k=1, act=True)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        out = channel_shuffle(out, 2)
        out = self.out_project(out)
        return out


class MG_ELAN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_experts: int = 16,
        middle_ratio: float = 0.5,
        num_blocks: int = 2,
    ):
        super().__init__()

        middle_channels = int(in_channels * middle_ratio)
        block_channels = int(in_channels * middle_ratio)
        final_channels = int(2 * middle_channels) + int(num_blocks * block_channels)
        self.main_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            internal_block = MoVE_GhostModule(
                in_channels=middle_channels,
                out_channels=block_channels,
                main_kernel_size=kernel_size,
                num_experts=num_experts,
                cheap_kernel_size=3,
            )
            middle_channels = block_channels
            self.blocks.append(internal_block)

        self.out_project = Conv(c1=final_channels, c2=out_channels, k=1, act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        block_outs = []
        x_block = x_main
        for block in self.blocks:
            x_block = block(x_block)
            block_outs.append(x_block)
        x_final = torch.cat((*block_outs[::-1], x_main, x_short), dim=1)
        return self.out_project(x_final)

class light_ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            Conv(c1=in_channels, c2=in_channels, k=1, act=True),
            Conv(c1=in_channels, c2=in_channels, k=3, g=in_channels, act=True),
            Conv(c1=in_channels, c2=out_channels, k=1, act=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)

class DualAxisAggAttn(nn.Module):
    def __init__(
        self,
        channels: int,
        middle_ratio: float = 0.5,
    ):
        super().__init__()
        self.channels = channels
        middle_channels = int(channels * middle_ratio)
        self.middle_channels = middle_channels

        self.main_conv = Conv(c1=channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=channels, c2=middle_channels, k=1, act=True)

        self.qkv = nn.ModuleDict(
            {
                "W": nn.Conv2d(
                    in_channels=middle_channels,
                    out_channels=1 + 2 * middle_channels,
                    kernel_size=1,
                    bias=True,
                ),
                "H": nn.Conv2d(
                    in_channels=middle_channels,
                    out_channels=1 + 2 * middle_channels,
                    kernel_size=1,
                    bias=True,
                ),
            }
        )

        self.conv_fusion = nn.ModuleDict(
            {
                "W": light_ConvBlock(
                    in_channels=middle_channels, out_channels=middle_channels
                ),
                "H": light_ConvBlock(
                    in_channels=middle_channels, out_channels=middle_channels
                ),
            }
        )


        final_channels = int(2 * middle_channels)
        self.out_project = Conv(c1=final_channels, c2=channels, k=1, act=True)

    def _apply_axis_attention(self, x, axis):
        """通用轴注意力计算"""
        qkv = self.qkv[axis](x)
        query, key, value = torch.split(
            qkv, [1, self.middle_channels, self.middle_channels], dim=1
        )

        # 明确指定softmax维度
        dim = -1 if axis == "W" else -2
        context_scores = F.softmax(query, dim=dim)
        context_vector = (key * context_scores).sum(dim=dim, keepdim=True)
        # gate = F.tanh(self.alpha[axis] * value) # 效果不及sigmoid
        # gate = F.silu(value) # 效果最差
        gate = F.sigmoid(value)
        # 将全局上下文向量乘以权重，并广播注入到特征图中
        out = x + gate * context_vector.expand_as(value)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [B, C, H, W]

        Returns:
            torch.Tensor: 输出张量，形状为 [B, C, H, W]
        """
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        # 宽轴注意力
        x_W = self._apply_axis_attention(x_main, "W")
        x_W_fused = self.conv_fusion["W"](x_W) + x_W
        # 高轴注意力
        x_H = self._apply_axis_attention(x_W_fused, "H")
        x_H_fused = self.conv_fusion["H"](x_H) + x_H

        x_out = torch.cat([x_H_fused, x_short], dim=1)

        x_out = self.out_project(x_out)

        return x_out



class DA3Block(nn.Module):
    """采用ELAN结构"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 9,
        kernel_size: int = 3,
    ):
        super().__init__()
        # 注意力子层
        # --------------------------------------------------------------

        self.attn = DualAxisAggAttn(channels=in_channels)
        

        self.norm1 = nn.BatchNorm2d(in_channels)
        # 局部特征提取模块
        #  --------------------------------------------------------------
        self.local_extractor = MG_ELAN(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            num_experts=num_experts,
            middle_ratio=0.5,
            num_blocks=2,
        )
        self.norm2 = nn.BatchNorm2d(in_channels)

        # 输出映射层
        # --------------------------------------------------------------
        if out_channels != in_channels:
            self.out_project = Conv(c1=in_channels, c2=out_channels, k=1, act=True)
        else:
            self.out_project = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # 注意力子层
        residual = x
        x = self.norm1(x)
        x = self.attn(x) + residual

        # 局部特征提取子层
        residual = x
        x = self.norm2(x)
        x = self.local_extractor(x) + residual

        x = self.out_project(x)

        return x
