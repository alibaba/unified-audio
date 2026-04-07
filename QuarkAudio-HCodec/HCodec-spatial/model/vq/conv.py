import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from typing import List


def init_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class Linear(nn.Module):
    """Linear."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(*args, **kwargs)
        
        # init
        # self.apply(init_weights)
    
    def forward(self, x):
        return self.linear(x)


class WNLinear(Linear):
    """Weight norm Conv1d."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = weight_norm(self.linear)


class Conv1d(nn.Module):
    """Conv1d."""
    def __init__(
        self,
        in_channels: int, out_channels: int,
        kernel_size: int, stride: int = 1, dilation: int = 1,
        groups: int = 1, bias: bool = True, causal: bool = False,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        dummy_kernel_size = (kernel_size - 1) * dilation + 1  # 使用dilation则等效kernel_size 变为 (kernel_size - 1) * dilation + 1
        if causal:
            self.pad = nn.ConstantPad1d((dummy_kernel_size - stride, 0), 0.0)
        else:
            self.pad = nn.ConstantPad1d((dummy_kernel_size // 2, dummy_kernel_size // 2), 0.0)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, dilation=dilation, groups=groups, bias=bias)

        # # init
        # self.apply(init_weights)
    
    def forward(self, x):
        x = self.pad(x)
        return self.conv(x)


class ConvTranspose1d(nn.Module):
    """
    内部使用SubPixelUpSampling, 方便流式
    """
    def __init__(
        self,
        in_channels: int, out_channels: int,
        kernel_size: int, stride: int = 1, dilation: int = 1,
        bias: bool = True, causal: bool = False,
    ):
        super().__init__()

        self.stride = stride
        self.out_channels = out_channels
        self.up = nn.Conv1d(in_channels, out_channels * stride, kernel_size=1)
        
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        dummy_kernel_size = (kernel_size - 1) * dilation + 1
        if causal:
            self.pad = nn.ConstantPad1d((dummy_kernel_size - 1, 0), 0.0)
        else:
            self.pad = nn.ConstantPad1d((dummy_kernel_size // 2, dummy_kernel_size // 2), 0.0)

        self.dw = nn.Conv1d(out_channels, out_channels, kernel_size, 1, groups=out_channels, bias=bias)

        # self.apply(init_weights)
    
    def forward(self, x):
        x = self.up(x)  # (B,k*D,T)
        x = x.unflatten(1, (self.stride, self.out_channels)).permute(0, 2, 3, 1)  # (B,D,T,k)
        x = x.flatten(start_dim=-2, end_dim=-1)  # (B,D,T*k)
        x = self.pad(x)
        x = self.dw(x)
        return x


class WNConv1d(Conv1d):
    """Weight norm Conv1d."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = weight_norm(self.conv)


class WNConvTranspose1d(nn.Module):
    """
    Weight norm ConvTranspose1d.
    内部使用SubPixelUpSampling, 方便流式
    """
    def __init__(
        self,
        in_channels: int, out_channels: int,
        kernel_size: int, stride: int = 1, dilation: int = 1,
        bias: bool = True, causal: bool = False,
    ):
        super().__init__()

        self.stride = stride
        self.out_channels = out_channels
        self.up = weight_norm(nn.Conv1d(in_channels, out_channels * stride, kernel_size=1))
        
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        dummy_kernel_size = (kernel_size - 1) * dilation + 1
        if causal:
            self.pad = nn.ConstantPad1d((dummy_kernel_size - 1, 0), 0.0)
        else:
            self.pad = nn.ConstantPad1d((dummy_kernel_size // 2, dummy_kernel_size // 2), 0.0)

        self.dw = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, 1, groups=out_channels, bias=bias))

        self.apply(init_weights)
    
    def forward(self, x):
        x = self.up(x)  # (B,k*D,T)
        x = x.unflatten(1, (self.stride, self.out_channels)).permute(0, 2, 3, 1)  # (B,D,T,k)
        x = x.flatten(start_dim=-2, end_dim=-1)  # (B,D,T*k)
        x = self.pad(x)
        x = self.dw(x)
        return x


class SEANetResnetBlock(nn.Module):
    def __init__(
        self, 
        dim: int, kernel_sizes: List[int] = [3, 1], dilations: List[int] = [1, 1],
        activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
        causal: bool = False, compress: int = 2, true_skip: bool = False,
    ):
        super().__init__()
        assert len(kernel_sizes) == len(dilations), 'Number of kernel sizes should match number of dilations'
        act = getattr(nn, activation)
        hidden = dim // compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                act(**activation_params),
                WNConv1d(in_chs, out_chs, kernel_size=kernel_size, dilation=dilation, causal=causal),
            ]
        self.block = nn.Sequential(*block)
        self.shortcut: nn.Module
        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = WNConv1d(dim, dim, kernel_size=1, causal=causal)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        causal: bool = False,
        dilation: int = 1,
        layer_scale_init_value: float = 0.0,
    ):
        super().__init__()        
        self.dwconv = Conv1d(dim, dim, kernel_size=7, stride=1, dilation=dilation, groups=dim, causal=causal)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x

'''
class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.1, causal=False):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = nn.LayerNorm(in_channels, eps=1e-6)
        self.conv1 = Conv1d(in_channels, out_channels, kernel_size=3, causal=causal)

        self.norm2 = nn.LayerNorm(out_channels, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = Conv1d(out_channels, out_channels, kernel_size=3, causal=causal)
        
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = Conv1d(in_channels, out_channels, kernel_size=3, causal=causal)
            else:
                self.nin_shortcut = Conv1d(in_channels, out_channels, kernel_size=1, causal=causal)
    
    def nonlinearity(self,x):
        # swish
        return x * torch.sigmoid(x)
    
    def forward(self, x):
        h = x
        h = self.norm1(h.transpose(1, 2)).transpose(1, 2)
        h = self.nonlinearity(h)
        h = self.conv1(h)
        
        h = self.norm2(h.transpose(1, 2)).transpose(1, 2)
        h = self.nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h
'''

def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.1, causal=False):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = Conv1d(in_channels, out_channels, kernel_size=3, causal=causal)

        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = Conv1d(out_channels, out_channels, kernel_size=3, causal=causal)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = Conv1d(in_channels, out_channels, kernel_size=3, causal=causal)
            else:
                self.nin_shortcut = Conv1d(in_channels, out_channels, kernel_size=1, causal=causal)

    def forward(self, x):            
        h = x
        h = self.norm1(h)
        h = self.nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        
        return x + h
    
    def nonlinearity(self, x):
        return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels, causal = False):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = Conv1d(in_channels, in_channels, kernel_size=1, causal=causal)
        self.k = Conv1d(in_channels, in_channels, kernel_size=1, causal=causal)
        self.v = Conv1d(in_channels, in_channels, kernel_size=1, causal=causal)
        self.proj_out = Conv1d(in_channels, in_channels, kernel_size=1, causal=causal)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h = q.shape
        q = q.permute(0, 2, 1)  # b,hw,c
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]

        h_ = self.proj_out(h_)

        return x + h_


class SLSTM(nn.Module):
    """
    LSTM without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    """
    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers, batch_first=True)
    
    def forward(self, x):
        # x: (B,C,T)
        res = x
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)
        if self.skip:
            x = x + res
        return x


class Transpose(nn.Module):
    """Transpose layer."""
    def __init__(self, dim1: int, dim2: int):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
    
    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


if __name__ == '__main__':
    x = torch.randn(1, 2, 4)
    # conv = WNConvTranspose1d(1, 1, 5*2+1, 5)
    # print(conv(x).shape)
    # m = SEANetResnetBlock(dim=2)
    m = ConvNeXtBlock(dim=2, intermediate_dim=4)
    print(m(x).shape)

