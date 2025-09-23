import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from mamba_ssm import Mamba2
from mamba_ssm.utils.generation import InferenceParams

class PawletteConfig(PretrainedConfig):
    """Pawlette模型配置类"""
    model_type = "pawlette"
    
    def __init__(
        self,
        # 基础参数
        vocab_size: int = 6400, 
        hidden_size: int = 672,  # d_model in Mamba
        state_size: int = 128,  # SSM state expansion factor (d_state) 
        conv_size: int = 4,  # Convolution width
        expand_factor: int = 2,  # Block expansion factor (d_inner = expand_factor * d_model)
        
        # 架构参数
        num_hidden_layers: int = 8, 
        dropout: float = 0.1,
        rms_norm_eps: float = 1e-5,
        use_bias: bool = False,
        use_conv_bias: bool = True,
        
        # Mamba2特定参数 - 完整支持最新特性
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        dt_limit: tuple = (0.0, float('inf')),
        
        # A和D矩阵初始化 - Mamba2核心特性
        A_init_range: tuple = (1, 16),  # Mamba2的A初始化范围
        D_has_hdim: bool = False,  # D是否有head维度
        
        # 归一化参数
        rmsnorm: bool = True,  # 是否使用RMSNorm
        norm_before_gate: bool = False,  # 门控前是否归一化
        
        # 分块和头参数 - Mamba2新特性
        headdim: int = 56,  # 每个头的维度，确保d_inner % headdim == 0 (1248*2=2496, 2496%52=0)
        ngroups: int = 1,  # SSM参数的组数
        d_ssm: int = None,  # SSM状态大小，如果为None则使用d_inner
        chunk_size: int = 32,  # 块大小用于分块处理 (优化内存使用)
        use_mem_eff_path: bool = True,  # 是否使用内存高效路径 (启用以优化内存)
        
        # 特殊token (与tokenizer配置保持一致)
        bos_token_id: int = 1,  # [SYS]
        eos_token_id: int = 0,  # [END]
        pad_token_id: int = 0,  # [END]
        
        # 训练参数
        use_cache: bool = True,
        tie_word_embeddings: bool = True,
        
        # LoRA支持
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.1,
        
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.conv_size = conv_size
        self.expand_factor = expand_factor
        self.intermediate_size = int(expand_factor * hidden_size)  # d_inner = expand_factor * d_model
        
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.rms_norm_eps = rms_norm_eps
        self.use_bias = use_bias
        self.use_conv_bias = use_conv_bias
        
        # Mamba2参数 - 完整支持
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init_floor = dt_init_floor
        self.dt_limit = dt_limit
        
        # A和D矩阵初始化参数
        self.A_init_range = A_init_range
        self.D_has_hdim = D_has_hdim
        
        # 归一化参数
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        
        # 分块和头参数
        self.headdim = headdim
        self.ngroups = ngroups
        self.d_ssm = d_ssm
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        
        # 特殊token
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        
        # 其他参数
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout


class RMSNorm(nn.Module):
    """RMSNorm归一化层"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # 计算RMS
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class PawletteMambaBlock(nn.Module):
    """Pawlette Mamba2 块"""
    def __init__(self, config: PawletteConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # 输入归一化
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Mamba2层 - 完整支持所有参数
        self.mamba = Mamba2(
            d_model=config.hidden_size,
            d_state=config.state_size,
            d_conv=config.conv_size,
            expand=config.expand_factor,
            headdim=config.headdim,
            ngroups=config.ngroups,
            A_init_range=config.A_init_range,
            D_has_hdim=config.D_has_hdim,
            rmsnorm=config.rmsnorm,
            norm_before_gate=config.norm_before_gate,
            dt_min=config.dt_min,
            dt_max=config.dt_max,
            dt_init_floor=config.dt_init_floor,
            dt_limit=config.dt_limit,
            bias=config.use_bias,
            conv_bias=config.use_conv_bias,
            chunk_size=config.chunk_size,
            use_mem_eff_path=config.use_mem_eff_path,
            layer_idx=self.layer_idx,
        )

        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # 梯度检查点支持
        self.gradient_checkpointing = False
        
    def gradient_checkpointing_enable(self):
        """启用梯度检查点"""
        self.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """禁用梯度检查点"""
        self.gradient_checkpointing = False
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        inference_params: Optional[InferenceParams] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            inference_params: Mamba2推理参数
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        def mamba_forward(hidden_states):
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
            
            # Mamba2处理 - 使用正确的推理参数
            if inference_params is not None:
                # 推理模式with cache
                hidden_states = self.mamba(
                    hidden_states,
                    inference_params=inference_params
                )
            else:
                # 训练模式
                hidden_states = self.mamba(hidden_states)
            
            # 残差连接
            hidden_states = self.dropout(hidden_states)
            hidden_states = residual + hidden_states
            
            return hidden_states
        
        # 使用梯度检查点（如果启用）
        if self.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(mamba_forward, hidden_states, use_reentrant=False)
        else:
            return mamba_forward(hidden_states)


class PawletteModel(nn.Module):
    """Pawlette基础模型"""
    
    def __init__(self, config: PawletteConfig):
        super().__init__()
        self.config = config
        
        # Token嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
        # Mamba2层堆叠
        self.layers = nn.ModuleList([
            PawletteMambaBlock(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        
        # 最终归一化层
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
                
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs
    ) -> Union[Tuple, dict]:
        """
        前向传播
        
        Args:
            input_ids: 输入token ids [batch_size, seq_len]
            attention_mask: 注意力掩码（Mamba2中主要用于兼容性，实际处理由InferenceParams管理）
            inference_params: Mamba2推理参数
            use_cache: 是否使用缓存
            output_hidden_states: 是否输出所有层的隐藏状态
            return_dict: 是否返回字典
            
        Returns:
            输出
        """
        batch_size, seq_len = input_ids.shape
        
        # 初始化InferenceParams（如果未提供）
        if inference_params is None and use_cache:
            # 为推理模式创建InferenceParams
            inference_params = InferenceParams(
                max_seqlen=seq_len,
                max_batch_size=batch_size,
                seqlen_offset=0,
                batch_size_offset=0,
                key_value_memory_dict={},
                lengths_per_sample=None,  # 训练时通常为None
            )
        
        # Token嵌入
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states)
        
        # Mamba2不需要在嵌入层处理attention_mask
        # Mamba2通过InferenceParams.lengths_per_sample和use_mem_eff_path自动处理变长序列
        # 在嵌入层做mask会破坏状态空间模型的递归状态传播
        
        # 存储所有层的隐藏状态（如果需要）
        all_hidden_states = [] if output_hidden_states else None
        
        # 通过所有Mamba2层
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
                
            hidden_states = layer(
                hidden_states,
                inference_params=inference_params
            )
        
        # 最终归一化
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        
        if return_dict:
            return {
                "last_hidden_state": hidden_states,
                "hidden_states": all_hidden_states,
                "inference_params": inference_params if use_cache else None
            }
        
        return hidden_states, all_hidden_states, inference_params


class PawletteForCausalLM(PreTrainedModel, GenerationMixin):
    """Pawlette因果语言模型"""
    
    config_class = PawletteConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    
    def __init__(self, config: PawletteConfig):
        super().__init__(config)
        self.config = config
        
        # 基础模型
        self.model = PawletteModel(config)
        
        # 语言模型头
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 权重共享
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        
        # 初始化权重
        self.post_init()
        
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def gradient_checkpointing_enable(self):
        """启用梯度检查点"""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        else:
            # 为每个层启用梯度检查点
            for layer in self.model.layers:
                if hasattr(layer, 'gradient_checkpointing_enable'):
                    layer.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        """禁用梯度检查点"""
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()
        else:
            # 为每个层禁用梯度检查点
            for layer in self.model.layers:
                if hasattr(layer, 'gradient_checkpointing_disable'):
                    layer.gradient_checkpointing_disable()
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inference_params: Optional[InferenceParams] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        前向传播
        
        Args:
            input_ids: 输入token ids
            attention_mask: 注意力掩码（Mamba2中主要用于兼容性，实际处理由InferenceParams管理）
            labels: 标签用于计算损失
            inference_params: Mamba2推理参数
            use_cache: 是否使用缓存
            output_hidden_states: 是否输出隐藏状态
            return_dict: 是否返回字典
            
        Returns:
            模型输出
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 初始化InferenceParams（如果未提供且使用缓存）
        if inference_params is None and use_cache:
            batch_size, seq_len = input_ids.shape
            inference_params = InferenceParams(
                max_seqlen=seq_len,
                max_batch_size=batch_size,
                seqlen_offset=0,
                batch_size_offset=0,
                key_value_memory_dict={},
                lengths_per_sample=None,
            )
        
        # 获取模型输出
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inference_params=inference_params,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        
        hidden_states = outputs["last_hidden_state"]
        
        # 计算logits
        logits = self.lm_head(hidden_states)
        
        # 计算损失
        loss = None
        if labels is not None:
            # Shift标签和logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 展平
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.get("inference_params"),
            hidden_states=outputs.get("hidden_states"),
            attentions=None  # Mamba没有注意力权重
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        **kwargs
    ):
        """为生成准备输入"""
        # 如果使用缓存，只需要最后一个token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        
        model_inputs = {
            "input_ids": input_ids,
            "inference_params": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
        
        return model_inputs
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """重排序缓存用于beam search - Mamba2特定实现"""
        if past_key_values is None:
            return None
            
        # Mamba2的cache重排序逻辑
        if isinstance(past_key_values, InferenceParams):
            # 对于InferenceParams，需要重排序其内部状态
            reordered_params = InferenceParams(
                max_seqlen=past_key_values.max_seqlen,
                max_batch_size=past_key_values.max_batch_size,
                seqlen_offset=past_key_values.seqlen_offset,
                batch_size_offset=past_key_values.batch_size_offset,
                key_value_memory_dict=past_key_values.key_value_memory_dict,
                lengths_per_sample=past_key_values.lengths_per_sample,
            )
            
            # 重排序key_value_memory_dict中的张量
            for key, value in reordered_params.key_value_memory_dict.items():
                if isinstance(value, torch.Tensor) and value.dim() > 0:
                    reordered_params.key_value_memory_dict[key] = value.index_select(0, beam_idx.to(value.device))
            
            return reordered_params
        else:
            # 兼容其他类型的cache
            reordered_past = {}
            for key, value in past_key_values.items():
                if isinstance(value, torch.Tensor):
                    reordered_past[key] = value.index_select(0, beam_idx.to(value.device))
                else:
                    reordered_past[key] = value
            return reordered_past


def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "total_M": total / 1e6,
        "trainable_M": trainable / 1e6
    }