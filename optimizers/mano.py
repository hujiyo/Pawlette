"""
Mano: Restriking Manifold Optimization for LLM Training

Based on arXiv:2601.23000 Algorithm 1
"""

import torch
import torch.optim as optim
import math


class ManoOptimizer(optim.Optimizer):
    """
    Mano Optimizer: Manifold-based optimizer for 2D matrix parameters

    Algorithm 1 from the paper:
    1. Update momentum: M_t = μ * M_{t-1} + g_t
    2. Rotating manifold: k = t mod 2
    3. Manifold normalization: θ̂_t = θ_t ⊘ ‖θ_t‖_{2,k}
    4. Tangent space projection: v_t = M_t - θ̂_t ⊙ ⟨M_t, θ̂_t⟩_k
    5. Update vector normalization: v̂_t = v_t ⊘ ‖v_t‖_{2,k}
    6. Parameter update: θ_{t+1} = θ_t * (1 - η_t*λ) - η_t * 0.2*√(n_k)*v̂_t

    Args:
        params (iterable): iterable of parameters to optimize
        lr (float): learning rate (default: 5e-4)
        momentum (float): momentum coefficient (default: 0.95)
        weight_decay (float): weight decay (default: 0.1)
        nesterov (bool): whether to use Nesterov-style momentum (default: False)
        eps (float): epsilon for numerical stability (default: 1e-8)
    """

    def __init__(self, params, lr=5e-4, momentum=0.95, weight_decay=0.1, nesterov=False, eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Momentum should be in [0,1): {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=nesterov, eps=eps, steps=0)
        super().__init__(params, defaults)

    def _manifold_normalization(self, param, k, eps=1e-8):
        """
        Manifold normalization: N_OB(A) = A ⊘ ‖A‖_{2,k}

        Args:
            param (Tensor): input tensor of shape [m, n]
            k (int): normalization dimension
                - k=0: column-wise normalization
                - k=1: row-wise normalization
            eps (float): epsilon for numerical stability

        Returns:
            Tensor: normalized tensor
        """
        if k == 0:  # Column normalization
            norm = param.norm(dim=0, keepdim=True)
        else:  # Row normalization
            norm = param.norm(dim=1, keepdim=True)

        # Avoid division by zero
        return param / norm.clamp(min=eps)

    def _dimension_wise_inner_product(self, A, B, k):
        """
        Dimension-wise inner product: ⟨A, B⟩_k

        Args:
            A, B (Tensor): input tensors of shape [m, n]
            k (int): dimension for inner product
                - k=0: column-wise inner product
                - k=1: row-wise inner product

        Returns:
            Tensor: inner product result
        """
        if k == 0:  # Column-wise
            return (A * B).sum(dim=0, keepdim=True)
        else:  # Row-wise
            return (A * B).sum(dim=1, keepdim=True)

    def _tangent_projection(self, M, theta_hat, k):
        """
        Tangent space projection: v_t = M_t - θ̂_t ⊙ ⟨M_t, θ̂_t⟩_k

        Projects momentum M onto the tangent space of theta_hat on Oblique manifold

        Args:
            M (Tensor): momentum tensor
            theta_hat (Tensor): manifold-normalized parameters
            k (int): projection dimension

        Returns:
            Tensor: projected tangent vector
        """
        inner_prod = self._dimension_wise_inner_product(M, theta_hat, k)
        return M - theta_hat * inner_prod

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss

        Returns:
            loss (Tensor or None): loss value from closure
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            mu = group['momentum']
            weight_decay = group['weight_decay']
            lr = group['lr']
            nesterov = group['nesterov']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # Mano only supports 2D matrix parameters
                if p.dim() != 2:
                    raise ValueError(
                        f"Mano optimizer only supports 2D tensor parameters, "
                        f"but parameter shape is {p.shape}"
                    )

                state = self.state[p]

                # Initialize momentum buffer
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)

                buf = state['momentum_buffer']

                # Step 1: Update momentum M_t = μ * M_{t-1} + g_t
                buf.mul_(mu).add_(grad)

                # Use Nesterov accelerated gradient if enabled
                g = grad.add(buf, alpha=mu) if nesterov else buf

                # Step 2: Rotating manifold k = steps mod 2
                # (steps increments per-parameter, matching official implementation)
                k = int(group['steps'] % 2)

                # Step 3: Manifold normalization θ̂_t = θ_t ⊘ ‖θ_t‖_{2,k}
                theta_hat = self._manifold_normalization(p.data, k, eps)

                # Step 4: Tangent projection v_t = g - θ̂_t ⊙ ⟨g, θ̂_t⟩_k
                v_t = self._tangent_projection(g, theta_hat, k)

                # Step 5: Update vector normalization v̂_t = v_t ⊘ ‖v_t‖_{2,k}
                v_hat = self._manifold_normalization(v_t, k, eps)

                # Step 6: Decoupled weight decay (multiply first, then update)
                p.data.mul_(1 - lr * weight_decay)

                # Step 7: Parameter update with rescaled RMS
                adjusted_lr = lr * 0.2 * math.sqrt(g.shape[k])
                p.data.add_(v_hat, alpha=-adjusted_lr)

                # Increment step counter per-parameter (matching official implementation)
                group['steps'] += 1

        return loss


class HybridManoAdamW(optim.Optimizer):
    """
    Hybrid optimizer: Mano for 2D matrices, AdamW for 1D vectors and embeddings

    Combines the strengths of both optimizers:
    - Mano for matrix parameters (Transformer weight matrices)
    - AdamW for embeddings, LayerNorm, and 1D parameters (bias, LayerNorm weight)

    Args:
        mano_params (list): parameters for Mano optimization (2D matrices)
        adamw_params (list): parameters for AdamW optimization (1D vectors)
        lr (float): learning rate
        momentum (float): Mano momentum coefficient
        weight_decay (float): weight decay
        nesterov (bool): whether to use Nesterov-style momentum (default: False)
        eps (float): epsilon for numerical stability (default: 1e-8)
        betas (tuple): AdamW beta parameters (default: (0.9, 0.95))
    """

    def __init__(self, mano_params, adamw_params, lr, momentum, weight_decay, nesterov=False, eps=1e-8, betas=(0.9, 0.95)):
        # Filter out empty parameter lists
        mano_params = [p for p in mano_params if p is not None]
        adamw_params = [p for p in adamw_params if p is not None]

        # Build parameter groups for parent class initialization
        all_params = mano_params + adamw_params
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=nesterov, eps=eps, betas=betas)
        super().__init__(all_params, defaults)

        # Mano optimizer (only create if 2D params exist)
        if len(mano_params) > 0:
            self.mano_optim = ManoOptimizer(
                mano_params,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=nesterov,
                eps=eps
            )
        else:
            self.mano_optim = None

        # AdamW optimizer (only create if 1D params exist)
        if len(adamw_params) > 0:
            self.adamw_optim = optim.AdamW(
                adamw_params,
                lr=lr,
                betas=betas,
                weight_decay=weight_decay
            )
        else:
            self.adamw_optim = None

        # Reorganize param_groups to merge both optimizers' groups
        self.param_groups = []
        if self.mano_optim:
            self.param_groups.extend(self.mano_optim.param_groups)
        if self.adamw_optim:
            self.param_groups.extend(self.adamw_optim.param_groups)

    def zero_grad(self):
        """Clears gradients of all optimizers"""
        if self.mano_optim:
            self.mano_optim.zero_grad()
        if self.adamw_optim:
            self.adamw_optim.zero_grad()

    def step(self, closure=None):
        """
        Performs a single optimization step

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss

        Returns:
            loss (Tensor or None): loss value from closure
        """
        loss_mano = None
        loss_adamw = None

        if self.mano_optim:
            loss_mano = self.mano_optim.step(closure)
        if self.adamw_optim:
            loss_adamw = self.adamw_optim.step(closure)

        return loss_mano if loss_mano is not None else loss_adamw

    def state_dict(self):
        """
        Returns optimizer state

        Returns:
            dict: dictionary containing 'mano' and 'adamw' keys
        """
        state = {}
        if self.mano_optim:
            state['mano'] = self.mano_optim.state_dict()
        if self.adamw_optim:
            state['adamw'] = self.adamw_optim.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """
        Loads optimizer state

        Args:
            state_dict (dict): dictionary containing 'mano' and 'adamw' keys
        """
        if 'mano' in state_dict and self.mano_optim:
            self.mano_optim.load_state_dict(state_dict['mano'])
        if 'adamw' in state_dict and self.adamw_optim:
            self.adamw_optim.load_state_dict(state_dict['adamw'])


def create_optimizer(model, optimizer_type='mano', lr=1e-3, momentum=0.95, weight_decay=1e-5, betas=(0.9, 0.95)):
    """
    Factory function to create optimizer based on type

    Args:
        model: PyTorch model
        optimizer_type (str): optimizer type, 'mano' or 'adamw'
        lr (float): learning rate
        momentum (float): Mano momentum coefficient
        weight_decay (float): weight decay
        betas (tuple): AdamW beta parameters

    Returns:
        optimizer: created optimizer
    """
    optimizer_type = optimizer_type.lower()

    if optimizer_type == 'mano':
        # Separate 2D and 1D parameters
        mano_params = []
        adamw_params = []

        for name, param in model.named_parameters():
            if param.dim() >= 2:
                mano_params.append(param)
            else:
                adamw_params.append(param)

        print(f"Optimizer: HybridManoAdamW (Mano for 2D matrices, AdamW for 1D params)")
        print(f"  2D params: {len(mano_params)}")
        print(f"  1D params: {len(adamw_params)}")

        return HybridManoAdamW(
            mano_params=mano_params,
            adamw_params=adamw_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            betas=betas
        )
    else:
        if optimizer_type == 'adamw':
            print(f"Optimizer: AdamW (weight_decay={weight_decay})")
            return optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        else:
            print(f"Optimizer: Adam (weight_decay={weight_decay})")
            return optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
