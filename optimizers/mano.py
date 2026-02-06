"""
Mano: Restriking Manifold Optimization for LLM Training

基于流形优化的LLM训练优化器，实现自arXiv:2601.23000 Algorithm 1

核心创新：
1. 旋转Oblique流形：在列归一化和行归一化之间交替
2. 软流形约束：不限制参数在流形上，而是对更新步骤施加流形约束
3. 切空间投影：先在参数空间投影，再在动量空间归一化

论文：https://arxiv.org/abs/2601.23000
"""

import torch
import torch.optim as optim
import math


class ManoOptimizer(optim.Optimizer):
    """
    Mano优化器：专门用于优化2D矩阵参数

    算法流程（Algorithm 1）：
    1. 更新动量: M_t = μ * M_{t-1} + g_t
    2. 旋转流形: k = t mod 2
    3. 参数流形归一化: θ̂_t = θ_t ⊘ ‖θ_t‖_{2,k}
    4. 切空间投影: v_t = M_t - θ̂_t ⊙ ⟨M_t, θ̂_t⟩_k
    5. 更新向量流形归一化: v̂_t = v_t ⊘ ‖v_t‖_{2,k}
    6. 参数更新: θ_{t+1} = θ_t - η_t * (0.2*√(n_k)*v̂_t + λ*θ_t)

    参数：
        params (iterable): 可迭代的参数或参数组
        lr (float): 学习率 (默认: 5e-4)
        momentum (float): 动量系数 (默认: 0.95)
        weight_decay (float): 权重衰减 (默认: 0.1)
    """

    def __init__(self, params, lr=5e-4, momentum=0.95, weight_decay=0.1):
        if not 0.0 <= lr:
            raise ValueError(f"无效的学习率: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"动量系数必须在[0,1)范围内: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"无效的权重衰减: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.step_count = 0

    def _manifold_normalization(self, param, k):
        """
        流形归一化操作: N_OB(A) = A ⊘ ‖A‖_{2,k}

        参数：
            param (Tensor): 输入张量，形状为[m, n]
            k (int): 归一化维度
                - k=0: 列归一化（除以每列的L2范数）
                - k=1: 行归一化（除以每行的L2范数）

        返回：
            Tensor: 归一化后的张量
        """
        if k == 0:  # 列归一化
            norm = param.norm(dim=0, keepdim=True)
        else:  # 行归一化
            norm = param.norm(dim=1, keepdim=True)

        # 避免除零，添加小常数
        return param / norm.clamp(min=1e-8)

    def _dimension_wise_inner_product(self, A, B, k):
        """
        维度级内积: ⟨A, B⟩_k

        参数：
            A, B (Tensor): 输入张量，形状为[m, n]
            k (int): 内积维度
                - k=0: 列方向内积（每列的元素乘积和）
                - k=1: 行方向内积（每行的元素乘积和）

        返回：
            Tensor: 内积结果，形状取决于k
                - k=0: [1, n]
                - k=1: [m, 1]
        """
        if k == 0:  # 列方向内积
            return (A * B).sum(dim=0, keepdim=True)
        else:  # 行方向内积
            return (A * B).sum(dim=1, keepdim=True)

    def _tangent_projection(self, M, theta_hat, k):
        """
        切空间投影: v_t = M_t - θ̂_t ⊙ ⟨M_t, θ̂_t⟩_k

        将动量M投影到θ_hat在Oblique流形上的切空间

        参数：
            M (Tensor): 动量张量
            theta_hat (Tensor): 流形归一化后的参数
            k (int): 投影维度

        返回：
            Tensor: 投影后的切向量
        """
        inner_prod = self._dimension_wise_inner_product(M, theta_hat, k)
        return M - theta_hat * inner_prod

    @torch.no_grad()
    def step(self, closure=None):
        """
        执行一步优化

        参数：
            closure (callable, optional): 重新计算模型并返回loss的闭包

        返回：
            loss (Tensor or None): 闭包返回的loss值
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # Mano只支持2D矩阵参数
                if p.dim() != 2:
                    raise ValueError(
                        f"Mano优化器只支持2D张量参数，"
                        f"但参数 {p.name if hasattr(p, 'name') else ''} "
                        f"的形状为 {p.shape}"
                    )

                state = self.state[p]

                # 初始化动量缓冲区
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)

                buf = state['momentum_buffer']

                # 步骤1: 更新动量 M_t = μ * M_{t-1} + g_t
                buf.mul_(momentum).add_(grad)

                # 步骤2: 旋转流形 k = t mod 2
                k = self.step_count % 2

                # 步骤3: 参数流形归一化 θ̂_t = θ_t ⊘ ‖θ_t‖_{2,k}
                theta_hat = self._manifold_normalization(p.data, k)

                # 步骤4: 切空间投影 v_t = M_t - θ̂_t ⊙ ⟨M_t, θ̂_t⟩_k
                v_t = self._tangent_projection(buf, theta_hat, k)

                # 步骤5: 更新向量流形归一化 v̂_t = v_t ⊘ ‖v_t‖_{2,k}
                v_hat = self._manifold_normalization(v_t, k)

                # 步骤6: 计算缩放因子 0.2 * √(n_k)
                m, n = p.shape
                n_k = m if k == 0 else n
                scale = 0.2 * math.sqrt(n_k)

                # 步骤7: 参数更新 θ_{t+1} = θ_t - η_t * (0.2*√(n_k)*v̂_t + λ*θ_t)
                update = scale * v_hat + weight_decay * p.data
                p.data.add_(-lr, update)

        # 递增步数计数器
        self.step_count += 1

        return loss
