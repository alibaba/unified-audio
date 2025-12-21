import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .llm import CustomLlamaModel


class LLM_SFT(CustomLlamaModel):
    def __init__(
        self,
        num_tasks: int = 1,  # 任务数量
        task_map: dict = {
            'se': 0,
        },
        feats_dim: int = 768,
        llm_base_config: dict= {},
    ):
        super().__init__(
            **llm_base_config
        )
        self.task_map = task_map

        # 任务专属token
        self.task_embedding = nn.Embedding(num_tasks, llm_base_config['hidden_size'])
        # 注册音频sos
        self.enroll_sos_embedding = nn.Embedding(1, llm_base_config['hidden_size'])

        self.adapter = nn.Linear(feats_dim, llm_base_config['hidden_size'])


    # 重写forward
    def forward(
        self,
        task_name: str,  # se, tse, rtse
        # enroll: torch.Tensor,  # (B, T, n_mels)
        # mixture: torch.Tensor,  # (B, T, n_mels)
        enroll_mel: torch.Tensor,
        enroll_feats: torch.Tensor,
        mix_mel: torch.Tensor,
        mix_feats: torch.Tensor,
        global_ids: Union[torch.IntTensor, torch.LongTensor],  # (B, 32)
        semantic_ids: Union[torch.IntTensor, torch.LongTensor],  # (B, T)
    ):
        global_ids = global_ids.long() + self.global_offset
        semantic_ids = semantic_ids.long() + self.semantic_offset

        global_sos_token_ids = torch.full((global_ids.size(0), 1), self.global_sos_token_id, dtype=global_ids.dtype, device=global_ids.device)
        semantic_sos_token_ids = torch.full((semantic_ids.size(0), 1), self.semantic_sos_token_id, dtype=semantic_ids.dtype, device=semantic_ids.device)
        semantic_eos_token_ids = torch.full((semantic_ids.size(0), 1), self.semantic_eos_token_id, dtype=semantic_ids.dtype, device=semantic_ids.device)

        # 微调时给定mixture，应该让模型预测eos
        input_ids = torch.cat([global_sos_token_ids, global_ids, semantic_sos_token_ids, semantic_ids], dim=1)  # (B, 1+32+1+T)
        target_ids = torch.cat([global_ids, semantic_sos_token_ids, semantic_ids, semantic_eos_token_ids], dim=1)  # (B, 32+1+T+1)

        task_embeds = self.task_embedding(torch.full((mix_mel.size(0), 1), self.task_map[task_name], dtype=torch.int64, device=mix_mel.device))

        # mix_mel = self.cond_input_layer(mix_mel)  # (B, T, hidden_size)
        # mix_mel = self.cond_encoder(mix_mel)  # (B, T, hidden_size)
        # mix_mel = self.cond_output_layer(mix_mel)  # (B, T, hidden_size)
        # mixture = self.adapter(torch.cat([mix_mel, mix_feats], dim=-1))
        mixture = self.adapter(mix_feats)
        mix_sos_embeds = self.mix_sos_embedding(torch.full((mixture.size(0), 1), 0, dtype=torch.int64, device=mixture.device))  # (B, 1, hidden_size)

        if enroll_mel is not None:
            # enroll_mel = self.cond_input_layer(enroll_mel)  # (B, T, hidden_size)
            # enroll_mel = self.cond_encoder(enroll_mel)  # (B, T, hidden_size)
            # enroll_mel = self.cond_output_layer(enroll_mel)  # (B, T, hidden_size)
            # enroll = self.adapter(torch.cat([enroll_mel, enroll_feats], dim=-1))
            enroll = self.adapter(enroll_feats)
            enroll_sos_embeds = self.enroll_sos_embedding(torch.full((enroll.size(0), 1), 0, dtype=torch.int64, device=enroll.device))  # (B, 1, hidden_size)
            inputs_embeds = torch.cat([task_embeds, enroll_sos_embeds, enroll, mix_sos_embeds, mixture, self.codec_embedding(input_ids)], dim=1)
        else:
            inputs_embeds = torch.cat([task_embeds, mix_sos_embeds, mixture, self.codec_embedding(input_ids)], dim=1)
        
        # 经过llm
        outputs = self.llm_forward(inputs_embeds)
        hidden_states = outputs.last_hidden_state[:, -target_ids.size(-1):, :]  # 去掉可能的条件部分

        logits = self.output_head(hidden_states)  # (B, 1+32+1+T-1, vocab_size)

        loss = self.loss_function(logits, target_ids)
        acc = (logits.argmax(-1) == target_ids).float().mean()

        return loss, acc


    # 重写generate
    def generate(
        self,
        task_name: str,  # se, tse, rtse
        # enroll: torch.Tensor,  # (B, T, n_mels)
        # mixture: torch.Tensor,  # (B, T, n_mels)
        enroll_mel: torch.Tensor,
        enroll_feats: torch.Tensor,
        mix_mel: torch.Tensor,
        mix_feats: torch.Tensor,
        global_length: int = 32,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
    ):
        semantic_length = mix_mel.size(1)
        
        task_embeds = self.task_embedding(torch.full((mix_mel.size(0), 1), self.task_map[task_name], dtype=torch.int64, device=mix_mel.device))

        # mix_mel = self.cond_input_layer(mix_mel)  # (B, T, hidden_size)
        # mix_mel = self.cond_encoder(mix_mel)  # (B, T, hidden_size)
        # mix_mel = self.cond_output_layer(mix_mel)  # (B, T, hidden_size)
        # mixture = self.adapter(torch.cat([mix_mel, mix_feats], dim=-1))
        mixture = self.adapter(mix_feats)
        mix_sos_embeds = self.mix_sos_embedding(torch.full((mixture.size(0), 1), 0, dtype=torch.int64, device=mixture.device))  # (B, 1, hidden_size)

        if enroll_mel is not None:
            # enroll_mel = self.cond_input_layer(enroll_mel)  # (B, T, hidden_size)
            # enroll_mel = self.cond_encoder(enroll_mel)  # (B, T, hidden_size)
            # enroll_mel = self.cond_output_layer(enroll_mel)  # (B, T, hidden_size)
            # enroll = self.adapter(torch.cat([enroll_mel, enroll_feats], dim=-1))
            enroll = self.adapter(enroll_feats)
            enroll_sos_embeds = self.enroll_sos_embedding(torch.full((enroll.size(0), 1), 0, dtype=torch.int64, device=enroll.device))  # (B, 1, hidden_size)
            inputs_embeds = torch.cat([task_embeds, enroll_sos_embeds, enroll, mix_sos_embeds, mixture], dim=1)
        else:
            inputs_embeds = torch.cat([task_embeds, mix_sos_embeds, mixture], dim=1)
        
        current_output = self.llm_forward(
            inputs_embeds,
            past_key_values=None,
            use_cache=True,
        )
        past_key_values = current_output.past_key_values

        input_ids = torch.full((mixture.size(0), 1), self.global_sos_token_id, dtype=torch.long, device=mixture.device)
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

        input_ids = torch.full((mixture.size(0), 1), self.semantic_sos_token_id, dtype=torch.long, device=mixture.device)
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
            mask[..., self.semantic_offset: self.semantic_offset + self.semantic_size] = True
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

