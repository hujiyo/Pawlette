"""
Pawlette优化器模块

包含多种优化器实现：
- Mano: 基于流形优化的LLM训练优化器
- HybridManoAdamW: Mano用于2D矩阵，AdamW用于Embedding和1D参数

两个优化器的区别
1. ManoOptimizer (optimizers/mano.py)
纯粹的Mano优化器实现
只能优化2D矩阵参数
如果传入1D参数会抛出错误
用途：底层实现，遵循论文Algorithm 1
2. HybridManoAdamW (optimizers/hybrid_optimizer.py)
包装器，组合了两个优化器
内部同时维护：
ManoOptimizer 实例（优化2D矩阵）
AdamW 实例（优化Embedding和1D参数）
用途：提供给训练脚本的统一接口
为什么需要混合方案？
根据Mano论文的建议：

参数类型	优化器	原因
embed_tokens.weight (6420×640)	AdamW	稀疏激活，需要自适应学习率
lm_head.weight (6420×640)	AdamW	同上
RMSNorm的weight (640)	AdamW	1D向量
MambaBlock的in_proj.weight (1920×640)	Mano	2D矩阵，流形优化有效
MambaBlock的conv1d.weight (1280×4)	Mano	2D矩阵
实际使用
在train_pretrain.py:370-396中：


if CONFIG['optimizer'] == 'mano':
    # 使用混合优化器（自动参数分组）
    optimizer = HybridManoAdamW(
        mano_params=mano_params,  # 2D矩阵 → 内部用ManoOptimizer
        adamw_params=adamw_params,  # 1D/Embedding → 内部用AdamW
        lr=actual_lr,
        momentum=CONFIG['momentum'],
        weight_decay=CONFIG['weight_decay']
    )
总结
ManoOptimizer: 底层实现，直接使用会报错（因为模型有1D参数）
HybridManoAdamW: 实际使用的接口，自动分组并用合适的优化器
"""

from .mano import ManoOptimizer
from .hybrid_optimizer import HybridManoAdamW

__all__ = ['ManoOptimizer', 'HybridManoAdamW']
