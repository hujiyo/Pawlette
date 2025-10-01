import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from mamba_ssm import Mamba2
from mamba_ssm.utils.generation import InferenceParams

class PawletteConfig(PretrainedConfig):
    """Pawlette模型配置类"""
    model_type = "pawlette"
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.vocab_size = 6420    # 词汇表大小
        self.hidden_size = 672    # 隐藏层大小
        self.state_size = 128    # SSM状态大小
        self.conv_size = 4    # 卷积宽度
        self.expand_factor = 2    # 块扩展因子
        self.intermediate_size = self.expand_factor * self.hidden_size    # 中间层大小
        
        self.num_hidden_layers = 8    # 隐藏层数量
        self.rms_norm_eps = 1e-5    # RMSNorm的epsilon
        self.use_bias = False    # 是否使用偏置
        self.use_conv_bias = True    # 是否使用卷积偏置
        
        # Mamba2参数 - 完整支持
        self.dt_min = 0.001    # 最小时间步长
        self.dt_max = 0.1    # 最大时间步长
        self.dt_init_floor = 1e-4    # 初始时间步长
        self.dt_limit = (0.0, float('inf'))    # 时间步长限制
        
        # A和D矩阵初始化参数
        self.A_init_range = (1, 16)    # A初始化范围
        self.D_has_hdim = False    # D是否有head维度
        
        # 归一化参数
        self.rmsnorm = True    # 是否使用RMSNorm
        self.norm_before_gate = False    # 门控前是否归一化
        
        # 分块和头参数
        self.headdim = 56   # 每个头的维度
        self.ngroups = 1    # SSM参数的组数
        self.d_ssm = None    # SSM状态大小，如果为None则使用d_inner
        self.chunk_size = 32    # 块大小用于分块处理 (优化内存使用)
        self.use_mem_eff_path = True    # 是否使用内存高效路径 (启用以优化内存)
        
        # 特殊token
        self.bos_token_id = 0   # [AI]
        self.eos_token_id = 11  # <ed>
        self.pad_token_id = 6   # [SEP]
        
        # 其他默认参数[禁止修改]
        self.use_cache = True   # 是否使用缓存
        self.tie_word_embeddings = False   # 是否共享词嵌入权重

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

class MambaBlock(nn.Module):
    """Pawlette Mamba2 块"""
    def __init__(self, config: PawletteConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx # 层索引

        # 输入归一化
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
        
    def forward(self,hidden_states: torch.Tensor,
        inference_params: Optional[InferenceParams] = None,
        **kwargs # 忽略其他参数，保持兼容性
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            inference_params: Mamba2推理参数
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        residual = hidden_states # 残差连接
        hidden_states = self.norm(hidden_states)
        
        # Mamba2处理 - 使用正确的推理参数
        if inference_params is not None:
            # 推理模式with cache
            hidden_states = self.mamba(hidden_states,inference_params=inference_params)
        else:
            # 训练模式
            hidden_states = self.mamba(hidden_states)
               
        hidden_states = residual + hidden_states # 残差连接
        return hidden_states


class PawletteModelCore(nn.Module):
    """Pawlette模型核心"""    
    def __init__(self, config:PawletteConfig):
        super().__init__()
        self.config = config        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size) # Token嵌入层
        
        
        self.layers = nn.ModuleList([
            MambaBlock(config, layer_idx=i)            
            for i in range(config.num_hidden_layers)
        ])# Mamba2层堆叠
        
        # 最终归一化层
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)        
        self.apply(self._init_weights) # 初始化权重
        
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


class PawletteModelLLM(PreTrainedModel, GenerationMixin):
    """Pawlette因果语言模型"""
    base_model_prefix = "model" 
    
    def __init__(self, config: PawletteConfig):
        super().__init__(config)
        self.config = config        
        self.model = PawletteModelCore(config) # 基础模型
        
        # 语言模型头
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
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
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 初始化InferenceParams（如果未提供）
        if inference_params is None:
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
            use_cache=self.config.use_cache,
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
            
            # 展平 - 使用标准的ignore_index=-100
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
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
        # 将transformers的DynamicCache转换为InferenceParams
        inference_params = None
        if past_key_values is not None:
            # 检查是否是InferenceParams
            if isinstance(past_key_values, InferenceParams):
                inference_params = past_key_values
                input_ids = input_ids[:, -1:]  # 只需要最后一个token
            else:
                # 如果是其他类型的cache（如DynamicCache），忽略并重新初始化
                # 这会导致性能下降，但能保证兼容性
                inference_params = None
        
        model_inputs = {
            "input_ids": input_ids,
            "inference_params": inference_params,
            "use_cache": kwargs.get("use_cache", True),
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