"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
import math

from x_transformers.x_transformers import RotaryEmbedding, apply_rotary_pos_emb


# RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.native_rms_norm = float(torch.__version__[:3]) >= 2.4

    def forward(self, x):
        if self.native_rms_norm:
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                x = x.to(self.weight.dtype)
            x = F.rms_norm(x, normalized_shape=(x.shape[-1],), weight=self.weight, eps=self.eps)
        else:
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                x = x.to(self.weight.dtype)
            x = x * self.weight

        return x


# Attention with possible joint part
# modified from diffusers/src/diffusers/models/attention_processor.py
class Attention(nn.Module):
    def __init__(
        self,
        processor,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,  # if not None -> joint attention
        context_pre_only: bool = False,
        qk_norm: Optional[str] = None,
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Attention equires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.processor = processor

        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout

        self.context_dim = context_dim
        self.context_pre_only = context_pre_only

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)

        if qk_norm is None:
            self.q_norm = None
            self.k_norm = None
        elif qk_norm == "rms_norm":
            self.q_norm = RMSNorm(dim_head, eps=1e-6)
            self.k_norm = RMSNorm(dim_head, eps=1e-6)
        else:
            raise ValueError(f"Unimplemented qk_norm: {qk_norm}")

        if self.context_dim is not None:
            self.to_q_c = nn.Linear(context_dim, self.inner_dim)
            self.to_k_c = nn.Linear(context_dim, self.inner_dim)
            self.to_v_c = nn.Linear(context_dim, self.inner_dim)
            if qk_norm is None:
                self.c_q_norm = None
                self.c_k_norm = None
            elif qk_norm == "rms_norm":
                self.c_q_norm = RMSNorm(dim_head, eps=1e-6)
                self.c_k_norm = RMSNorm(dim_head, eps=1e-6)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, dim))
        self.to_out.append(nn.Dropout(dropout))

        if self.context_dim is not None and not self.context_pre_only:
            self.to_out_c = nn.Linear(self.inner_dim, context_dim)

    def forward(
        self,
        x: float["b n d"],  # noised input x  # noqa: F722
        c: float["b n d"] = None,  # context c  # noqa: F722
        mask: bool["b n"] | None = None,  # noqa: F722
        rope=None,  # rotary position embedding for x
        c_rope=None,  # rotary position embedding for c
    ) -> torch.Tensor:
        if c is not None:
            return self.processor(self, x, c=c, mask=mask, rope=rope, c_rope=c_rope)
        else:
            return self.processor(self, x, mask=mask, rope=rope)


# Attention Processor
class AttnProcessor:
    def __init__(
        self,
        pe_attn_head: int | None = None,  # number of attention head to apply rope, None for all
    ):
        self.pe_attn_head = pe_attn_head

    def __call__(
        self,
        attn: Attention,
        x: float["b n d"],  # noised input x  # noqa: F722
        mask: bool["b n"] | None = None,  # noqa: F722
        rope=None,  # rotary position embedding
    ) -> torch.FloatTensor:
        batch_size = x.shape[0]

        # `sample` projections
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)  # b,h,n,hd
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # qk norm
        if attn.q_norm is not None:
            query = attn.q_norm(query)
        if attn.k_norm is not None:
            key = attn.k_norm(key)

        # apply rotary position embedding, lib: x_transformers
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)

            if self.pe_attn_head is not None:
                pn = self.pe_attn_head
                query[:, :pn, :, :] = apply_rotary_pos_emb(query[:, :pn, :, :], freqs, q_xpos_scale)
                key[:, :pn, :, :] = apply_rotary_pos_emb(key[:, :pn, :, :], freqs, k_xpos_scale)
            else:
                query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
                key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        # mask. e.g. inference got a batch with different target durations, mask out the padding
        # caution: bool dtype, True for non-paded positions, False for paded positions. Only for padding, not for causal!
        if mask is not None:
            attn_mask = mask
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # 'b n -> b 1 1 n'
            attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None

        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)  # b,h,n,hd
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)  # b,n,d
        x = x.to(query.dtype)

        # linear proj
        x = attn.to_out[0](x)
        # dropout
        x = attn.to_out[1](x)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)

        return x


# Joint Attention processor for MM-DiT
# modified from diffusers/src/diffusers/models/attention_processor.py
class JointAttnProcessor:
    def __init__(self):
        pass

    def __call__(
        self,
        attn: Attention,
        x: float["b n d"],  # noised input x  # noqa: F722
        c: float["b nt d"] = None,  # context c, here text # noqa: F722
        mask: bool["b n"] | None = None,  # noqa: F722
        rope=None,  # rotary position embedding for x
        c_rope=None,  # rotary position embedding for c
    ) -> torch.FloatTensor:
        residual = x

        batch_size = c.shape[0]

        # `sample` projections
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        # `context` projections
        c_query = attn.to_q_c(c)
        c_key = attn.to_k_c(c)
        c_value = attn.to_v_c(c)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        c_query = c_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        c_key = c_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        c_value = c_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # qk norm
        if attn.q_norm is not None:
            query = attn.q_norm(query)
        if attn.k_norm is not None:
            key = attn.k_norm(key)
        if attn.c_q_norm is not None:
            c_query = attn.c_q_norm(c_query)
        if attn.c_k_norm is not None:
            c_key = attn.c_k_norm(c_key)

        # apply rope for context and noised input independently
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)
        if c_rope is not None:
            freqs, xpos_scale = c_rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            c_query = apply_rotary_pos_emb(c_query, freqs, q_xpos_scale)
            c_key = apply_rotary_pos_emb(c_key, freqs, k_xpos_scale)

        # joint attention
        query = torch.cat([query, c_query], dim=2)
        key = torch.cat([key, c_key], dim=2)
        value = torch.cat([value, c_value], dim=2)

        # mask. e.g. inference got a batch with different target durations, mask out the padding
        if mask is not None:
            attn_mask = F.pad(mask, (0, c.shape[1]), value=True)  # no mask for c (text)
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # 'b n -> b 1 1 n'
            attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None

        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        x = x.to(query.dtype)

        # Split the attention outputs.
        x, c = (
            x[:, : residual.shape[1]],
            x[:, residual.shape[1] :],
        )

        # linear proj
        x = attn.to_out[0](x)
        # dropout
        x = attn.to_out[1](x)
        if not attn.context_pre_only:
            c = attn.to_out_c(c)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)
            # c = c.masked_fill(~mask, 0.)  # no mask for c (text)

        return x, c


# FeedForward
class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        self.sequential = torch.nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.sequential(x)


# Conformer convolution module
# modified from torchaudio
class ConvolutionModule(nn.Module):
    r"""Conformer convolution module.

    Args:
        input_dim (int): input dimension.
        num_channels (int): number of depthwise convolution layer input channels.
        depthwise_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        bias (bool, optional): indicates whether to add bias term to each convolution layer. (Default: ``False``)
        use_group_norm (bool, optional): use GroupNorm rather than BatchNorm. (Default: ``False``)
    """

    def __init__(
        self,
        input_dim: int,
        num_channels: int,
        depthwise_kernel_size: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_group_norm: bool = False,
    ) -> None:
        super().__init__()
        if (depthwise_kernel_size - 1) % 2 != 0:
            raise ValueError("depthwise_kernel_size must be odd to achieve 'SAME' padding.")
        self.layer_norm = nn.LayerNorm(input_dim)
        self.sequential = nn.Sequential(
            nn.Conv1d(
                input_dim,
                2 * num_channels,
                1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            nn.GLU(dim=1),
            nn.Conv1d(
                num_channels,
                num_channels,
                depthwise_kernel_size,
                stride=1,
                padding=(depthwise_kernel_size - 1) // 2,
                groups=num_channels,
                bias=bias,
            ),
            nn.GroupNorm(num_groups=1, num_channels=num_channels)
            if use_group_norm
            else nn.BatchNorm1d(num_channels),
            nn.SiLU(),
            nn.Conv1d(
                num_channels,
                input_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, D)`.

        Returns:
            torch.Tensor: output, with shape `(B, T, D)`.
        """
        x = self.layer_norm(input)
        x = x.transpose(1, 2)
        x = self.sequential(x)
        return x.transpose(1, 2)


class ConformerLayer(nn.Module):
    def __init__(
        self, 
        dim, 
        heads, 
        dim_head,
        depthwise_conv_kernel_size = 31,
        ff_mult=4, 
        dropout=0.1, 
        qk_norm=None, 
        pe_attn_head=None,
    ):
        super().__init__()

        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=dropout)

        self.attn_norm = nn.LayerNorm(dim)
        self.attn = Attention(
            processor=AttnProcessor(pe_attn_head=pe_attn_head),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            qk_norm=qk_norm,
        )

        self.conv_module = ConvolutionModule(
            input_dim=dim,
            num_channels=dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            bias=True,
            use_group_norm=False,
        )

        self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=dropout)

        self.final_norm = nn.LayerNorm(dim)
    
    def forward(self, x, mask=None, rope=None):

        residual = x
        x = self.ff1(x)
        x = x * 0.5 + residual

        residual = x
        x = self.attn_norm(x)
        x = self.attn(x, mask=mask, rope=rope)
        x = x + residual

        residual = x
        x = self.conv_module(x)
        x = x + residual

        residual = x
        x = self.ff2(x)
        x = x * 0.5 + residual

        x = self.final_norm(x)

        return x


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers,
        dim, 
        heads, 
        dim_head,
        depthwise_conv_kernel_size = 31,
        ff_mult=4, 
        dropout=0.1, 
        qk_norm=None, 
        pe_attn_head=None,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            ConformerLayer(
                dim=dim, 
                heads=heads, 
                dim_head=dim_head,
                depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                ff_mult=ff_mult, 
                dropout=dropout, 
                qk_norm=qk_norm, 
                pe_attn_head=pe_attn_head,
            )
            for _ in range(num_layers)
        ])

        self.rotary_embedding = RotaryEmbedding(dim_head)
    
    def forward(self, x, mask=None):
        # x: (b,t,d)
        rope = self.rotary_embedding.forward_from_seq_len(x.size(-2))

        for layer in self.layers:
            x = layer(x, mask=mask, rope=rope)
        return x


if __name__ == "__main__":
    model = ConformerEncoder(
        num_layers=2,
        dim=128, 
        heads=4, 
        dim_head=32,
        depthwise_conv_kernel_size=31,
        ff_mult=4, 
        dropout=0.1, 
        qk_norm=None, 
        pe_attn_head=None,
    ).cuda()
    x = torch.randn(2, 100, 128).cuda()
    mask = torch.ones(2, 100).bool().cuda()
    print(model(x, mask=mask).shape)

