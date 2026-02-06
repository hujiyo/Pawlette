"""
混合优化器：HybridManoAdamW

结合Mano和AdamW的优势：
- Mano优化2D矩阵参数（MambaBlock的权重矩阵）
- AdamW优化Embedding层、LM Head和1D参数（RMSNorm, bias）

这种混合策略遵循论文建议，利用不同优化器的优势：
1. Mano对矩阵参数进行流形归一化，提升训练效率
2. AdamW对稀疏激活的Embedding层进行自适应学习率调整
"""

import torch.optim as optim
from .mano import ManoOptimizer


class HybridManoAdamW:
    """
    混合优化器：Mano用于2D矩阵，AdamW用于1D向量和Embedding

    用法：
        mano_params = [p for p in model.parameters() if p.dim() >= 2]
        adamw_params = [p for p in model.parameters() if p.dim() < 2]

        optimizer = HybridManoAdamW(
            mano_params=mano_params,
            adamw_params=adamw_params,
            lr=5e-4,
            momentum=0.95,
            weight_decay=0.01
        )
    """

    def __init__(self, mano_params, adamw_params, lr, momentum, weight_decay):
        """
        初始化混合优化器

        参数：
            mano_params (list): Mano优化的参数列表（2D矩阵）
            adamw_params (list): AdamW优化的参数列表（1D向量）
            lr (float): 学习率
            momentum (float): Mano动量系数
            weight_decay (float): 权重衰减
        """
        # Mano优化器
        self.mano_optim = ManoOptimizer(
            mano_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )

        # AdamW优化器
        self.adamw_optim = optim.AdamW(
            adamw_params,
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=weight_decay
        )

    def zero_grad(self):
        """清空所有优化器的梯度"""
        self.mano_optim.zero_grad()
        self.adamw_optim.zero_grad()

    def step(self, closure=None):
        """
        执行一步优化

        参数：
            closure (callable, optional): 重新计算模型并返回loss的闭包

        返回：
            loss (Tensor or None): 闭包返回的loss值
        """
        loss_mano = self.mano_optim.step(closure)
        loss_adamw = self.adamw_optim.step(closure)
        return loss_mano if loss_mano is not None else loss_adamw

    def state_dict(self):
        """
        返回优化器状态

        返回：
            dict: 包含'mano'和'adamw'两个键的字典
        """
        return {
            'mano': self.mano_optim.state_dict(),
            'adamw': self.adamw_optim.state_dict()
        }

    def load_state_dict(self, state_dict):
        """
        加载优化器状态

        参数：
            state_dict (dict): 包含'mano'和'adamw'两个键的字典
        """
        self.mano_optim.load_state_dict(state_dict['mano'])
        self.adamw_optim.load_state_dict(state_dict['adamw'])
