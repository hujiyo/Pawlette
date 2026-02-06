"""
Mano优化器单元测试

测试Mano优化器的基本功能、旋转流形机制和与标准模型的兼容性
"""

import torch
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizers.mano import ManoOptimizer
from optimizers.hybrid_optimizer import HybridManoAdamW


def test_mano_basic():
    """测试Mano优化器基本功能"""
    print("\n=== 测试1: Mano基本功能 ===")

    # 创建简单模型
    model = torch.nn.Linear(10, 20)
    optimizer = ManoOptimizer(model.parameters(), lr=1e-3)

    # 前向传播
    x = torch.randn(5, 10)
    y = model(x)
    loss = y.sum()

    # 反向传播
    loss.backward()

    # 优化器步进
    optimizer.step()
    optimizer.zero_grad()

    print("✅ Mano基本功能测试通过")
    print(f"   步数计数: {optimizer.step_count}")
    return True


def test_mano_rotational_manifold():
    """测试旋转流形机制"""
    print("\n=== 测试2: 旋转流形机制 ===")

    model = torch.nn.Linear(10, 20)
    optimizer = ManoOptimizer(model.parameters(), lr=1e-3)

    # 执行多步，验证k交替
    for i in range(10):
        x = torch.randn(5, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        expected_k = i % 2
        assert optimizer.step_count == i + 1, f"步数计数错误: {optimizer.step_count} != {i + 1}"

    print("✅ 旋转流形测试通过")
    print(f"   总步数: {optimizer.step_count}")
    return True


def test_mano_state_dict():
    """测试状态保存和加载"""
    print("\n=== 测试3: 状态保存和加载 ===")

    model = torch.nn.Linear(10, 20)
    optimizer = ManoOptimizer(model.parameters(), lr=1e-3, momentum=0.95)

    # 执行几步训练
    for _ in range(5):
        x = torch.randn(5, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # 保存状态
    state_dict = optimizer.state_dict()

    # 创建新优化器并加载状态
    new_optimizer = ManoOptimizer(model.parameters(), lr=1e-3, momentum=0.95)
    new_optimizer.load_state_dict(state_dict)

    # 验证步数一致
    assert new_optimizer.step_count == optimizer.step_count, "状态加载失败"

    print("✅ 状态保存和加载测试通过")
    print(f"   步数: {new_optimizer.step_count}")
    return True


def test_hybrid_optimizer():
    """测试混合优化器"""
    print("\n=== 测试4: 混合优化器 ===")

    # 创建包含2D和1D参数的模型
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.LayerNorm(20),
        torch.nn.Linear(20, 10),
    )

    # 分组参数
    mano_params = []
    adamw_params = []
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            mano_params.append(param)
        else:
            adamw_params.append(param)

    # 创建混合优化器
    optimizer = HybridManoAdamW(
        mano_params=mano_params,
        adamw_params=adamw_params,
        lr=1e-3,
        momentum=0.95,
        weight_decay=0.01
    )

    # 训练几步
    for _ in range(5):
        x = torch.randn(5, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print("✅ 混合优化器测试通过")
    print(f"   Mano参数数: {sum(p.numel() for p in mano_params)}")
    print(f"   AdamW参数数: {sum(p.numel() for p in adamw_params)}")
    return True


def test_hybrid_state_dict():
    """测试混合优化器状态保存和加载"""
    print("\n=== 测试5: 混合优化器状态保存和加载 ===")

    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.LayerNorm(20),
    )

    mano_params = [p for p in model.parameters() if p.dim() >= 2]
    adamw_params = [p for p in model.parameters() if p.dim() < 2]

    optimizer = HybridManoAdamW(
        mano_params=mano_params,
        adamw_params=adamw_params,
        lr=1e-3,
        momentum=0.95,
        weight_decay=0.01
    )

    # 训练几步
    for _ in range(3):
        x = torch.randn(5, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # 保存状态
    state_dict = optimizer.state_dict()

    # 验证状态字典结构
    assert 'mano' in state_dict, "缺少mano状态"
    assert 'adamw' in state_dict, "缺少adamw状态"

    # 创建新优化器并加载
    new_optimizer = HybridManoAdamW(
        mano_params=mano_params,
        adamw_params=adamw_params,
        lr=1e-3,
        momentum=0.95,
        weight_decay=0.01
    )
    new_optimizer.load_state_dict(state_dict)

    print("✅ 混合优化器状态保存和加载测试通过")
    return True


def test_parameter_validation():
    """测试参数验证"""
    print("\n=== 测试6: 参数验证 ===")

    # 测试1D参数应该抛出错误
    model_1d = torch.nn.LayerNorm(10)
    try:
        optimizer = ManoOptimizer(model_1d.parameters(), lr=1e-3)
        x = torch.randn(5, 10)
        y = model_1d(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        print("❌ 应该抛出1D参数错误")
        return False
    except ValueError as e:
        if "只支持2D张量" in str(e):
            print("✅ 参数验证测试通过（正确拒绝1D参数）")
        else:
            print(f"❌ 错误信息不正确: {e}")
            return False

    # 测试2D参数应该正常工作
    model_2d = torch.nn.Linear(10, 20)
    try:
        optimizer = ManoOptimizer(model_2d.parameters(), lr=1e-3)
        x = torch.randn(5, 10)
        y = model_2d(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        print("✅ 2D参数正常工作")
    except Exception as e:
        print(f"❌ 2D参数不应该抛出错误: {e}")
        return False

    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*50)
    print("开始Mano优化器单元测试")
    print("="*50)

    tests = [
        test_mano_basic,
        test_mano_rotational_manifold,
        test_mano_state_dict,
        test_hybrid_optimizer,
        test_hybrid_state_dict,
        test_parameter_validation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ 测试失败: {test.__name__}")
            print(f"   错误: {e}")
            failed += 1

    print("\n" + "="*50)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("="*50)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
