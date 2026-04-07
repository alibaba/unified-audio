import torch
from torch import nn
from transformers.cache_utils import DynamicCache
import torch.nn.functional as F
from typing import Optional, Tuple


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        max_position_embeddings: int,
        dim: int,
        rope_theta: float = 10000.0,
        partial_rotary_factor: float = 1.0,
        device=None
    ):
        super().__init__()
        self.rope_init_fn = self._compute_default_rope_parameters

        self.max_seq_len_cached = max_position_embeddings
        self.original_max_seq_len = max_position_embeddings

        inv_freq, self.attention_scaling = self.rope_init_fn(
            dim=dim,
            rope_theta=rope_theta,
            partial_rotary_factor=partial_rotary_factor,
            device=device,
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    

    def _compute_default_rope_parameters(
        self,
        dim: int,
        rope_theta: float,
        partial_rotary_factor: float = 1.0,
        device: Optional["torch.device"] = None,
        seq_len: Optional[int] = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PretrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """

        dim = int(dim * partial_rotary_factor)
        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        return inv_freq, attention_factor

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


class Attention(nn.Module):
    def __init__(
        self, 
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int,
        layer_idx: int,
        attention_dropout = 0.0,
    ):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5
        self.layer_idx = layer_idx

        self.rnn = nn.LSTM(hidden_size, hidden_size, 1, batch_first=True)
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)
        self.attention_dropout = attention_dropout
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[DynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        hidden_states, _ = self.rnn(hidden_states)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # (b,h,t,d)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_output, attn_weights = self.attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )  # (b,t,h,d), (b,h,t,t)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()  # (b,t,d)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    
    def attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:  # True for seen, False for unseen
            assert attention_mask.ndim == 3
            attention_mask = attention_mask.unsqueeze(1).expand(*attn_weights.shape)  # (b, h, n, n)
            attention_bias = torch.zeros_like(attn_weights, dtype=attn_weights.dtype, device=attn_weights.device)
            attention_bias.masked_fill_(attention_mask.logical_not(), float("-inf"))
            attn_weights = attn_weights + attention_bias
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights


    def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`, *optional*):
                Deprecated and unused.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed
    
    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))



class MoE(nn.Module):
    def __init__(
        self, 
        dim: int,
        intermediate_size: int,
        n_routed_experts: int = 3,
        n_activated_experts: int = 1,
        n_shared_experts : int = 1,
        n_expert_groups: int = 1,
        n_limited_groups: int = 1,
        score_func: str = "softmax",
        route_scale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.n_local_experts = n_routed_experts 
        self.n_activated_experts = n_activated_experts
        self.experts_start_idx = 0
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        
        self.gate = Gate(
            dim=dim,
            n_routed_experts=n_routed_experts,
            n_activated_experts=n_activated_experts,
            n_expert_groups=n_expert_groups,
            n_limited_groups=n_limited_groups,
            score_func=score_func,
            route_scale=route_scale,
        )

        self.experts = nn.ModuleList([MLP(dim, intermediate_size) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(n_routed_experts)])
        self.shared_experts = MLP(dim, n_shared_experts * intermediate_size)

        # self.norm2 = create_norm_fn("layer_norm", 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x_origin = x
        # x = self.norm2(x) ##from mimi transformer ffn block

        ##from deepseek ffn moe
        shape = x.size()
        # breakpoint()
        x = x.reshape(-1, self.dim)
        weights, indices = self.gate(x)  # (B*T, 1), (B*T, 1)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)
        # if world_size > 1:
        #     dist.all_reduce(y)
        return (y + z).view(shape)


class Gate(nn.Module):
    def __init__(
        self,
        dim : int,
        n_routed_experts: int = 3,
        n_activated_experts: int = 1,
        n_expert_groups: int = 1,
        n_limited_groups: int = 1,
        score_func: str = "softmax",
        route_scale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.topk = n_activated_experts
        self.n_groups = n_expert_groups
        self.topk_groups = n_limited_groups
        self.score_func = score_func
        self.route_scale = route_scale
        # self.weight = nn.Parameter(torch.empty(n_routed_experts, dim))
        self.linear = nn.Linear(dim, n_routed_experts, bias=False)
        self.bias = nn.Parameter(torch.empty(n_routed_experts))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.linear(x)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = torch.zeros_like(scores[..., 0]).scatter_(1, indices, True)
            scores = (scores * mask.unsqueeze(-1)).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(x), indices


class TransformerLayer(nn.Module):
    def __init__(
        self, 
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        head_dim: int,
        layer_idx: int = 0,
        attention_dropout = 0.0,
        use_moe: bool = False,
    ):
        super().__init__()

        self.self_attn = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            layer_idx=layer_idx,
            attention_dropout=attention_dropout,
        )

        # self.mlp = Qwen2MLP(config)
        if use_moe:
            self.mlp = MoE(dim=hidden_size, intermediate_size=intermediate_size, use_rnn=use_rnn)
        else:
            self.mlp = MLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=1e-6)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=1e-6)

    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    ) -> tuple[torch.Tensor]:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Transformer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        head_dim: int = None,
        attention_dropout = 0.0,
        use_moe: bool = False,
        max_position_embeddings: int = 4096,
        causal: bool = False,
        use_sliding_window: bool = False,
        left_context: int = 0,
    ):
        super().__init__()

        head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                head_dim=head_dim,
                layer_idx=i,
                attention_dropout=attention_dropout,
                use_moe=use_moe,
            )
            for i in range(num_hidden_layers)
        ])

        self.rotary_emb = RotaryEmbedding(
            max_position_embeddings=max_position_embeddings,
            dim=head_dim,
        )

        self.causal = causal
        self.use_sliding_window = use_sliding_window
        self.left_context = left_context
    
    @staticmethod
    def create_sliding_window_mask(seq_len, left_context, device):
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)) * torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=-left_context+1)
        mask = mask.bool()  # (n, n)
        return mask
    
    @staticmethod
    def create_causal_mask(seq_len, device):
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        mask = mask.bool()  # (n, n)
        return mask


    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor],
        past_key_values: Optional[DynamicCache] = None,
        use_cache: Optional[bool] = None,
    ) -> Tuple[torch.FloatTensor]:

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, cache_position.unsqueeze(0))

        mask = None
        if self.causal:
            if self.use_sliding_window:
                mask = self.create_sliding_window_mask(hidden_states.shape[1], self.left_context, hidden_states.device)
            else:
                mask = self.create_causal_mask(hidden_states.shape[1], hidden_states.device)
            mask = mask.unsqueeze(0).repeat(hidden_states.size(0), 1, 1)  # (B, T, T)
        
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        
        if use_cache:
            return hidden_states, past_key_values
        else:
            return hidden_states



if __name__ == "__main__":

    # rope = RotaryEmbedding(max_position_embeddings=100, dim=4)
    # position_ids = torch.arange(100).unsqueeze(0)
    # cos, sin = rope(position_ids.float(), position_ids)

    # x = torch.randn(1, 10, 16)
    # moe = MoE(dim=16, n_routed_experts=3, n_activated_experts=1, n_shared_experts=1)
    # y = moe(x)
    # print(y.shape)

    # x = torch.randn(1, 10, 16)
    # m = TransformerLayer(
    #     hidden_size=16,
    #     intermediate_size=64,
    #     num_attention_heads=4,
    #     head_dim=16,
    #     layer_idx=0,
    #     attention_dropout = 0.0,
    #     use_moe = True,
    # )

    # past_seen_tokens = 0
    # cache_position = torch.arange(past_seen_tokens, past_seen_tokens + x.shape[1], device=x.device)
    # past_key_values = None

    # rope = RotaryEmbedding(max_position_embeddings=100, dim=16)
    # position_embeddings = rope(x, cache_position.unsqueeze(0))

    # y = m(
    #     x,
    #     attention_mask=None,
    #     past_key_values=past_key_values, 
    #     cache_position=cache_position, 
    #     position_embeddings=position_embeddings
    # )
    # print(y.shape)

    x = torch.randn(1, 10, 16)
    m = Transformer(
        hidden_size=16,
        intermediate_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        head_dim=None,
        attention_dropout = 0.0,
        use_moe = True,
        max_position_embeddings = 4096,
        causal = False,
        use_sliding_window = False,
        left_context = 5,
    )
    
    y = m(
        x,
    )
    print(y.shape)


    # import pdb; pdb.set_trace()


