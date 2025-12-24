from einops import rearrange
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from typing import List

from .fsq_quantizer import FSQ

class FSQWrapper(nn.Module):
    def __init__(self, input_dim, levels=[8,8,8,8,8], num_codebooks=1, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        
        self.fsq = FSQ(levels=levels, dim=input_dim, num_codebooks=num_codebooks, **kwargs)

    def forward(self, x, n_quantizers=None, possibly_no_quantizer=False):
        # x is (B, D, T) where D is input_dim
        x_transposed = x.transpose(1, 2) # (B, T, D)
        
        zq_transposed, indices = self.fsq(x_transposed)
        
        zq = zq_transposed.transpose(1, 2)
        
        if self.fsq.num_codebooks > 1:
            indices = indices.transpose(1, 2)
        else:
            indices = indices.unsqueeze(1)

        latents = None

        codebook_loss = torch.tensor(0.0, device=x.device)
        loss = torch.tensor(0.0, device=x.device)

        first_layer_quantized = zq
        
        return zq, indices, latents, loss, codebook_loss, first_layer_quantized

    def from_codes(self, codes):
        # codes: (B, n_q, T)
        if self.fsq.num_codebooks > 1:
            indices = codes.transpose(1, 2)
        else:
            indices = codes.squeeze(1)
        
        quantized_transposed = self.fsq.indices_to_codes(indices) 
        
        quantized = quantized_transposed.transpose(1, 2)
        
        return quantized, None 