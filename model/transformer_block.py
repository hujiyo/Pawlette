import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMSNorm归一化层（Root Mean Square Layer Normalization）"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """预计算RoPE频率"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor):
    """应用RoPE位置编码"""
    # q, k: [batch, num_heads, seq_len, head_dim]
    # cos, sin: [max_seq_len, head_dim//2]
    # position_ids: [batch, seq_len]
    
    # 选择对应位置的cos/sin
    cos = cos[position_ids].unsqueeze(1)  # [batch, 1, seq_len, head_dim//2]
    sin = sin[position_ids].unsqueeze(1)
    
    # 将head_dim拆成两半
    q_embed = (q * cos.repeat(1, 1, 1, 2)) + (rotate_half(q) * sin.repeat(1, 1, 1, 2))
    k_embed = (k * cos.repeat(1, 1, 1, 2)) + (rotate_half(k) * sin.repeat(1, 1, 1, 2))
    return q_embed, k_embed

def rotate_half(x: torch.Tensor):
    """旋转输入张量的一半维度"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class SwiGLUFeedForward(nn.Module):
    """SwiGLU前馈网络，与Mamba的隐藏尺寸保持一致"""
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float, use_bias: bool):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=use_bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=use_bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))

class MultiHeadSelfAttention(nn.Module):
    """带KV缓存和RoPE的自注意力实现"""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        use_bias: bool,
        max_position_embeddings: int = 8192,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size必须能被num_heads整除"

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # 预计算RoPE频率
        freqs_cos, freqs_sin = precompute_freqs_cis(self.head_dim, max_position_embeddings)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, hidden_size = hidden_states.size()

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算position_ids
        if past_key_value is not None:
            past_len = past_key_value[0].shape[2]
            position_ids = torch.arange(past_len, past_len + seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
        else:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
        
        # 应用RoPE
        query, key = apply_rotary_pos_emb(query, key, self.freqs_cos, self.freqs_sin, position_ids)

        # KV缓存
        if past_key_value is not None:
            key = torch.cat([past_key_value[0], key], dim=2)
            value = torch.cat([past_key_value[1], value], dim=2)
        
        present_key_value = (key, value) if use_cache else None
        
        # 计算注意力
        kv_seq_len = key.shape[2]
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        # Causal mask
        causal_mask = torch.full((seq_len, kv_seq_len), float("-inf"), device=hidden_states.device, dtype=attn_scores.dtype)
        if seq_len > 1:
            causal_mask = torch.triu(causal_mask, diagonal=kv_seq_len - seq_len + 1)
        else:
            causal_mask.fill_(0.0)
        attn_scores = attn_scores + causal_mask.unsqueeze(0).unsqueeze(0)

        # Padding mask (如果有)
        if attention_mask is not None:
            # attention_mask: [batch, kv_seq_len] -> [batch, 1, 1, kv_seq_len]
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attention_mask = (1.0 - attention_mask) * torch.finfo(attn_scores.dtype).min
            attn_scores = attn_scores + attention_mask

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)

        output = self.out_proj(context)
        return self.resid_dropout(output), present_key_value

class TransformerBlock(nn.Module):
    """用于替换部分Mamba层的Transformer块，支持KV缓存"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = getattr(config, "transformer_intermediate_size", config.intermediate_size)
        num_heads = config.transformer_num_heads
        dropout = config.transformer_dropout
        attn_dropout = config.transformer_attn_dropout
        use_bias = config.use_bias
        eps = config.rms_norm_eps

        self.layer_idx = layer_idx
        self.input_norm = RMSNorm(hidden_size, eps=eps)
        self.self_attn = MultiHeadSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=attn_dropout,
            use_bias=use_bias,
            max_position_embeddings=8192,
        )
        self.post_attn_dropout = nn.Dropout(dropout)
        self.post_attn_norm = RMSNorm(hidden_size, eps=eps)
        self.mlp = SwiGLUFeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout,
            use_bias=use_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        inference_params=None,
        attention_mask: Optional[torch.Tensor] = None,
        transformer_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            inference_params: Mamba的推理参数（本层不使用，但保持接口兼容）
            attention_mask: [batch, total_seq_len] 注意力掩码
            transformer_kv_cache: 本层的KV缓存 (key, value)
            use_cache: 是否使用缓存
            
        Returns:
            hidden_states: [batch, seq_len, hidden_size]
            present_kv: 更新后的KV缓存
        """
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        attn_output, present_kv = self.self_attn(
            hidden_states, 
            attention_mask=attention_mask,
            past_key_value=transformer_kv_cache,
            use_cache=use_cache,
        )
        hidden_states = residual + self.post_attn_dropout(attn_output)
        residual = hidden_states
        hidden_states = self.post_attn_norm(hidden_states)
        ff_output = self.mlp(hidden_states)
        hidden_states = residual + ff_output

        return hidden_states, present_kv
