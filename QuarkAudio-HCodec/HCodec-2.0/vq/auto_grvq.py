from typing import Union

import numpy as np
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))

class AutoGroupVectorQuantize(nn.Module):
    """
    Inspirations:
    The core ideas of our approach are derived from the following papers and concepts:
    1.Grouped Quantization: The concept of grouping from HIFI-CODEC: GROUP-RESIDUAL VECTOR QUANTIZATION FOR HIGH FIDELITY AUDIO CODEC.
    2.Cosine Similarity Search: The technique of performing a codebook search after dimensionality reduction and using L2 normalization to shift the distance metric from Euclidean distance to cosine similarity, as detailed in High-Fidelity Audio Compression with Improved RVQGAN.
    3.Temporal Residual Coding: An idea from traditional codecs, where temporal residual coding is used to reduce the dynamic range of codebook representations for non-initial speech frames.
    Key Steps:
    The core pipeline consists of the following steps:
    1.(Optional) Apply inter-frame residual coding to the input latents along the time dimension.
    2.Simultaneously perform adaptive grouping and dimensionality reduction on the input latents.
    3.Apply intra-frame residual coding to the resulting parallel data after reduction.
    4.Perform the codebook search in parallel across all groups.
    """

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int, frame_residual_vq=False):
        super().__init__()

        self.codebook_size_a = codebook_size
        self.codebook_size_b = codebook_size
        self.codebook_dim = codebook_dim
        self.frame_residual_vq = frame_residual_vq
        self.codebook_dim_a = codebook_dim
        self.codebook_dim_b = codebook_dim

        self.in_proj_a = WNConv1d(input_dim, self.codebook_dim_a, kernel_size=1)
        self.out_proj_a = WNConv1d(self.codebook_dim_a, input_dim // 2, kernel_size=1)

        self.in_proj_b = WNConv1d(input_dim, self.codebook_dim_b, kernel_size=1)
        self.out_proj_b = WNConv1d(self.codebook_dim_b, input_dim // 2, kernel_size=1)

        self.codebook_a = nn.Embedding(self.codebook_size_a, self.codebook_dim_a)
        self.codebook_b = nn.Embedding(self.codebook_size_b, self.codebook_dim_b)

    def forward(self, z):
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """

        # Factorized codes (ViT-VQGAN) Project input into low-dimensional space

        if self.frame_residual_vq:
            for frame in range(z.shape[-1] - 1, 0, -1):
                z[..., frame] = z[..., frame] - z[..., frame - 1]

        z_a = self.in_proj_a(z)  # z_a : (B x D x T)
        z_b = self.in_proj_b(z)  # z_b : (B x D x T)

        # z_a = z_a - z_b
        z_aq, indices_a = self.decode_latents(z_a, self.codebook_a)
        z_bq, indices_b = self.decode_latents(z_b, self.codebook_b)

        commitment_loss = F.mse_loss(z_a, z_aq.detach(), reduction="none").mean([1, 2]) + F.mse_loss(z_b, z_bq.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_aq, z_a.detach(), reduction="none").mean([1, 2]) + F.mse_loss(z_bq, z_b.detach(), reduction="none").mean([1, 2])

        # c
        z_aq = (z_a + (z_aq - z_a).detach())  # noop in forward pass, straight-through gradient estimator in backward pass
        z_bq = (z_b + (z_bq - z_b).detach())  # noop in forward pass, straight-through gradient estimator in backward pass

        z_aq = self.out_proj_a(z_aq)
        z_bq = self.out_proj_b(z_bq)
        z_q = torch.cat((z_aq, z_bq), dim=1)

        if self.frame_residual_vq:
            for frame in range(1, 1, z_q.shape[-1]):
                z_q[..., frame] = z_q[..., frame - 1] + z_q[..., frame]

        indices = indices_a * self.codebook_size_b + indices_b  
        latent = torch.cat((z_a, z_b), dim=1)

        return z_q, commitment_loss, codebook_loss, indices, latent

    def embed_code(self, embed_id, codebook):
        return F.embedding(embed_id, codebook.weight)

    def decode_code(self, embed_id, codebook):
        return self.embed_code(embed_id, codebook).transpose(1, 2)

    def decode_latents(self, latents, codebook_in):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = codebook_in.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (encodings.pow(2).sum(1, keepdim=True) - 2 * encodings @ codebook.t() + codebook.pow(2).sum(1, keepdim=True).t())
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices, codebook_in)
        # z_q shape [B, dim, T]

        return z_q, indices


class AutoGroupResidualVectorQuantize(nn.Module):

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
        frame_residual_vq: bool = False,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers = nn.ModuleList([
            AutoGroupVectorQuantize(input_dim, codebook_size, codebook_dim[i], frame_residual_vq=frame_residual_vq)
            for i in range(n_codebooks)
        ])
        self.quantizer_dropout = quantizer_dropout

    def forward(self, z, n_quantizers: int = None):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            No. of quantizers to use
            (n_quantizers < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.
        Returns
        -------
        dict
            A dictionary with the following keys:

            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
        """
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break

            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(residual)

            # Create mask to apply quantizer dropout
            mask = (torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers)
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            # Sum losses
            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)

        return z_q, codes, latents, commitment_loss, codebook_loss

    def from_codes(self, codes: torch.Tensor):
        """Given the quantized codes, reconstruct the continuous representation
        Parameters
        ----------
        codes : Tensor[B x N x T]
            Quantized discrete representation of input
        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        """

        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            codes_a = codes[:, i, :] // self.quantizers[i].codebook_size_b
            codes_b = codes[:, i, :] - codes_a * self.quantizers[i].codebook_size_b

            z_pa_i = self.quantizers[i].decode_code(codes_a, self.quantizers[i].codebook_a)
            z_pb_i = self.quantizers[i].decode_code(codes_b, self.quantizers[i].codebook_b)

            z_p.append(torch.cat((z_pa_i, z_pb_i), dim=1))

            z_aq = self.quantizers[i].out_proj_a(z_pa_i)
            z_bq = self.quantizers[i].out_proj_b(z_pb_i)
            z_q = z_q + torch.cat((z_aq, z_bq), dim=1)

        return z_q, torch.cat(z_p, dim=1), codes

    def from_latents(self, latents: torch.Tensor):
        """Given the unquantized latents, reconstruct the
        continuous representation after quantization.

        Parameters
        ----------
        latents : Tensor[B x N x T]
            Continuous representation of input after projection

        Returns
        -------
        Tensor[B x D x T]
            Quantized representation of full-projected space
        Tensor[B x D x T]
            Quantized representation of latent space
        """
        z_q = 0
        z_p = []
        codes = []
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[0]
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            z_p_i, codes_i = self.quantizers[i].decode_latents(latents[:, j:k, :])
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)