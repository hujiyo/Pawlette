"""
Pawlette Optimizers Module

Includes multiple optimizer implementations:
- ManoOptimizer: Manifold-based optimizer for LLM training
- HybridManoAdamW: Mano for 2D matrices, AdamW for embeddings and 1D parameters
- create_optimizer: Factory function for easy optimizer creation

Usage:
    from optimizers import create_optimizer, HybridManoAdamW, ManoOptimizer
    
    # Create hybrid optimizer (recommended)
    optimizer = create_optimizer(model, optimizer_type='mano', lr=5e-4)
    
    # Or manually separate parameters
    mano_params = [p for p in model.parameters() if p.dim() >= 2]
    adamw_params = [p for p in model.parameters() if p.dim() < 2]
    optimizer = HybridManoAdamW(
        mano_params=mano_params,
        adamw_params=adamw_params,
        lr=5e-4,
        momentum=0.95,
        weight_decay=0.1
    )
"""

from .mano import ManoOptimizer, HybridManoAdamW, create_optimizer

__all__ = ['ManoOptimizer', 'HybridManoAdamW', 'create_optimizer']
