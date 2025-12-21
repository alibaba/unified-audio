import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union

from transformers import LlamaModel, LlamaConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from .conformer import ConformerEncoder


class CustomLlamaModel(nn.Module):
    def __init__(
        self,
        cond_dim: int = 80,
        global_size: int = 4096,
        semantic_size: int = 8192,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_attention_heads: int = 8,
        dropout_p: float = 0.1,
        max_position_embeddings = 4096,
        label_smoothing: float = 0.1,
        conformer_params: dict = {
            "num_layers": 2,
            "dim": 256,
            "heads": 8, 
            "dim_head": 32,
            "depthwise_conv_kernel_size": 31,
            'ff_mult': 4,
            'dropout': 0.1,
            'qk_norm': None,
            'pe_attn_head': None,
        },
    ):
        super().__init__()
        
        self.global_sos_token_id = 0
        self.semantic_sos_token_id = 1
        self.semantic_eos_token_id = 2
        self.vocab_size = 3 + global_size + semantic_size
        self.global_offset = 3
        self.semantic_offset = 3 + global_size
        self.global_size = global_size
        self.semantic_size = semantic_size

        # 指示mixture开始的embedding，不需要id，因为模型不需要输出它
        self.mix_sos_embedding = nn.Embedding(1, hidden_size)

        # condition encoder
        self.cond_input_layer = nn.Linear(cond_dim, conformer_params["dim"])
        self.cond_encoder = ConformerEncoder(**conformer_params)
        self.cond_output_layer = nn.Linear(conformer_params["dim"], hidden_size)

        # 自定义输入embedding层
        self.codec_embedding = nn.Embedding(
            self.vocab_size,
            hidden_size,
        )
        
        # 获取Llama的transformer层
        config = LlamaConfig(
            vocab_size=self.vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=hidden_size * 4,
            dropout_rate=dropout_p,
            attention_dropout=dropout_p,
            max_position_embeddings=max_position_embeddings,  # rope的最长序列长度
        )
        self.config = config
        llama_model = LlamaModel(config)
        self.layers = llama_model.layers
        self.rotary_emb = llama_model.rotary_emb
        self.norm = llama_model.norm

        self._update_causal_mask = llama_model._update_causal_mask
        
        # 自定义输出层
        self.output_head = nn.Linear(hidden_size, self.vocab_size, bias=False)  # [hidden_size, vocab_size]

        self.label_smoothing = label_smoothing


    def loss_function(self, logits, target):
        logits = logits.float()
        confidence = 1.0 - self.label_smoothing
        size = logits.size(-1)
        logits = logits.reshape(-1, size)
        target = target.reshape(-1)

        with torch.no_grad():
            true_dist = logits.clone()
            true_dist.fill_(self.label_smoothing / (size - 1))
            true_dist.scatter_(1, target.unsqueeze(1), confidence)
            
        loss = F.kl_div(
            F.log_softmax(logits, dim=-1),
            true_dist,
            reduction='batchmean',
        )
        return loss


    def forward(
        self,
        global_ids: Union[torch.IntTensor, torch.LongTensor],  # (B, 32)
        semantic_ids: Union[torch.IntTensor, torch.LongTensor],  # (B, T)
        cond: Union[torch.FloatTensor, type(None)] = None, # (B, T, n_mels)
    ):
        # (B, 32)
        global_ids = global_ids.long() + self.global_offset
        semantic_ids = semantic_ids.long() + self.semantic_offset

        global_sos_token_ids = torch.full((global_ids.size(0), 1), self.global_sos_token_id, dtype=global_ids.dtype, device=global_ids.device)
        semantic_sos_token_ids = torch.full((semantic_ids.size(0), 1), self.semantic_sos_token_id, dtype=semantic_ids.dtype, device=semantic_ids.device)
        semantic_eos_token_ids = torch.full((semantic_ids.size(0), 1), self.semantic_eos_token_id, dtype=semantic_ids.dtype, device=semantic_ids.device)

        input_ids = torch.cat([global_sos_token_ids, global_ids, semantic_sos_token_ids, semantic_ids], dim=1)  # (B, 1+32+1+T)
        target_ids = torch.cat([global_ids, semantic_sos_token_ids, semantic_ids, semantic_eos_token_ids], dim=1)  # (B, 32+1+T+1)

        # 预训练时防止对semantic_eos_token_ids进行建模，因为可能导致模型倾向于输出终止token
        # 并且训练数据可能是从音频中间截断，此时也不应该终止
        input_ids = input_ids[:, :-1]
        target_ids = target_ids[:, :-1]

        if cond is not None:
            cond = self.cond_input_layer(cond)  # (B, T, hidden_size)
            cond = self.cond_encoder(cond)  # (B, T, hidden_size)
            cond = self.cond_output_layer(cond)  # (B, T, hidden_size)
            mix_sos_embeds = self.mix_sos_embedding(torch.full((cond.size(0), 1), 0, dtype=torch.int64, device=cond.device))  # (B, 1, hidden_size)
            inputs_embeds = torch.cat([mix_sos_embeds, cond, self.codec_embedding(input_ids)], dim=1)
        else:
            inputs_embeds = self.codec_embedding(input_ids)  # (B, 1+32+1+T-1, hidden_size)
        
        # 经过llm
        outputs = self.llm_forward(inputs_embeds)
        hidden_states = outputs.last_hidden_state[:, -target_ids.size(-1):, :]  # 去掉可能的条件部分
        
        logits = self.output_head(hidden_states)  # (B, 1+32+1+T-1, vocab_size)

        loss = self.loss_function(logits, target_ids)
        acc = (logits.argmax(-1) == target_ids).float().mean()

        return loss, acc


    def llm_forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.Tensor] = None,  # SDPA 不需要指定attention_mask，默认为causal
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs,
    ) -> BaseModelOutputWithPast:
        """
        模型前向传播
        
        Returns:
            如果use_cache=False: logits [batch_size, seq_len, vocab_size]
            如果use_cache=True: (logits, past_key_values)
        """
        
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # 此函数仅用作测试QK_Cache
    def test_generate(
        self,
        inputs_embeds: Optional[torch.FloatTensor],
        top_k: int = 50,
        top_p: float = 0.95,
        temperature: float = 0.8,
    ):
        past_key_values = None
        last_hidden_states = []

        for ii in range(inputs_embeds.size(1)):
            current_output = self.llm_forward(
                inputs_embeds[:, ii:ii+1],
                past_key_values=past_key_values,
                use_cache=True,
            )
            last_hidden_states.append(current_output.last_hidden_state)
            past_key_values = current_output.past_key_values
        
        last_hidden_states = torch.cat(last_hidden_states, dim=1)
        return last_hidden_states
    

    def sample_logits(
        self,
        logits: torch.FloatTensor,  # [batch_size, vocab_size]
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
    ):  
        # Top-K 过滤
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Top-p (nucleus) 采样
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 从大到小，默认最后一维
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0  # 使概率刚好超过top_p一个
            
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        assert 0 < temperature <= 1.0
        logits = logits / temperature
        
        # 采样或贪婪搜索
        if do_sample:
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]
        else:
            next_tokens = torch.argmax(logits, dim=-1, keepdim=True)  # [batch_size, 1]
        
        return next_tokens
    

    def generate(
        self,
        cond: Union[torch.FloatTensor, type(None)] = None,
        global_length: int = 32,
        semantic_length: int = 150,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
    ):
        if cond is None:
            past_key_values = None
        else:
            cond = self.cond_input_layer(cond)  # (B, T, hidden_size)
            cond = self.cond_encoder(cond)  # (B, T, hidden_size)
            cond = self.cond_output_layer(cond)  # (B, T, hidden_size)
            mix_sos_embeds = self.mix_sos_embedding(torch.full((cond.size(0), 1), 0, dtype=torch.int64, device=cond.device))  # (B, 1, hidden_size)
            inputs_embeds = torch.cat([mix_sos_embeds, cond], dim=1)
            current_output = self.llm_forward(
                inputs_embeds,
                past_key_values=None,
                use_cache=True,
            )
            past_key_values = current_output.past_key_values
        
        input_ids = torch.full((1, 1), self.global_sos_token_id, dtype=torch.long, device=next(self.parameters()).device)
        output_ids = []
        for _ in range(global_length):
            inputs_embeds = self.codec_embedding(input_ids)
            current_output = self.llm_forward(
                inputs_embeds,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = current_output.past_key_values
            hidden_states = current_output.last_hidden_state  # [batch_size, 1, hidden_size]
            next_token_logits = self.output_head(hidden_states)  # [batch_size, 1, vocab_size]
            
            # 将非global_token的地方设置为 -float('inf')
            mask = torch.zeros_like(next_token_logits, dtype=torch.bool)
            mask[..., self.global_offset: self.global_offset+self.global_size] = True
            next_token_logits[~mask] = float('-inf')

            next_token_id = self.sample_logits(
                next_token_logits.squeeze(1),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )  # [batch_size, 1]
            output_ids.append(next_token_id)
            input_ids = next_token_id
        global_ids = torch.cat(output_ids, dim=-1) - self.global_offset  #  [batch_size, global_length]

        input_ids = torch.full((1, 1), self.semantic_sos_token_id, dtype=torch.long, device=next(self.parameters()).device)
        output_ids = []
        for _ in range(semantic_length):
            inputs_embeds = self.codec_embedding(input_ids)
            current_output = self.llm_forward(
                inputs_embeds,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = current_output.past_key_values
            hidden_states = current_output.last_hidden_state  # [batch_size, 1, hidden_size]
            next_token_logits = self.output_head(hidden_states)  # [batch_size, 1, vocab_size]

            # 将非semantic_token的地方设置为 -float('inf')
            mask = torch.zeros_like(next_token_logits, dtype=torch.bool)
            mask[..., self.semantic_offset: self.semantic_offset+self.semantic_size] = True
            next_token_logits[~mask] = float('-inf')
            
            next_token_id = self.sample_logits(
                next_token_logits.squeeze(1),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )  # [batch_size, 1]
            output_ids.append(next_token_id)
            input_ids = next_token_id
        semantic_ids = torch.cat(output_ids, dim=-1) - self.semantic_offset
        
        return global_ids, semantic_ids


if __name__=='__main__':
    model = CustomLlamaModel(
        hidden_size = 768,
        num_layers = 8,
        num_attention_heads = 8,
    ).eval().cuda()

    # inputs_embeds1 = torch.randn(1, 10, 256)
    # inputs_embeds2 = torch.randn(1, 10, 256)
    # inputs_embeds3 = torch.randn(1, 10, 256)
    # out1 = model.llm_forward(torch.cat([inputs_embeds1, inputs_embeds2], dim=1)).last_hidden_state
    # out2 = model.llm_forward(torch.cat([inputs_embeds1, inputs_embeds3], dim=1)).last_hidden_state

    # print((out1[:, :10] - out2[:, :10]).abs().sum())

    # inputs_embeds = torch.randn(1, 10, 256)
    # out1 = model.llm_forward(inputs_embeds).last_hidden_state
    # out2 = model.test_generate(inputs_embeds)

    # print((out1[:, :10] - out2[:, :10]).abs().sum())

    # global_ids = torch.randint(0, 4096, (1, 32)).cuda()
    # semantic_ids = torch.randint(0, 8192, (1, 100)).cuda()

    # # import ipdb; ipdb.set_trace()
    # loss = model(global_ids, semantic_ids)
    # print(loss)

    # global_ids, semantic_ids = model.generate(input_ids=None)
    print(sum([p.numel() for p in model.parameters()]))

