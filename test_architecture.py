#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pawlette纯Mamba2架构测试脚本
测试纯Mamba2架构的正确性
"""

import torch
import sys
from model.model_pawlette import (
    PawletteConfig, 
    PawletteModelLLM, 
    MambaBlock,
    count_parameters
)


def print_separator(title):
    """打印分隔符"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_architecture_layout():
    """测试1：验证架构布局"""
    print_separator("测试1: 架构布局验证")
    
    config = PawletteConfig()
    model = PawletteModelLLM(config)
    
    print(f"总层数: {config.num_hidden_layers}")
    print(f"架构类型: 纯Mamba2架构")
    print(f"\n层类型检查:")
    
    mamba_count = 0
    errors = []
    
    for i, layer in enumerate(model.model.layers):
        is_mamba = isinstance(layer, MambaBlock)
        
        status = "✅" if is_mamba else "❌"
        layer_type = "Mamba" if is_mamba else "Unknown"
        print(f"  层{i:2d}: {layer_type:11s} {status}")
        
        if not is_mamba:
            errors.append(f"Layer {i}: expected Mamba, got {type(layer).__name__}")
        else:
            mamba_count += 1
    
    print(f"\n统计:")
    print(f"  Mamba层: {mamba_count}")
    print(f"  总层数: {config.num_hidden_layers}")
    
    if errors:
        print(f"\n❌ 发现 {len(errors)} 个错误:")
        for error in errors:
            print(f"  - {error}")
        return False
    elif mamba_count != config.num_hidden_layers:
        print(f"\n❌ 层数不匹配: 期望{config.num_hidden_layers}，实际{mamba_count}")
        return False
    else:
        print(f"\n✅ 架构布局正确！所有层都是Mamba2层！")
        return True


def test_forward_pass():
    """测试2：前向传播"""
    print_separator("测试2: 前向传播测试")
    
    config = PawletteConfig()
    model = PawletteModelLLM(config)
    model.eval()
    
    batch_size = 2
    seq_len = 128
    
    print(f"输入形状: batch_size={batch_size}, seq_len={seq_len}")
    
    # 创建测试输入
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    try:
        # 训练模式（不使用缓存）
        print("\n训练模式（use_cache=False）:")
        model.train()
        with torch.no_grad():
            outputs_train = model(input_ids, use_cache=False)
        
        print(f"  ✅ logits形状: {outputs_train.logits.shape}")
        print(f"  ✅ past_key_values: {outputs_train.past_key_values}")
        
        # 推理模式（使用缓存）
        print("\n推理模式（use_cache=True）:")
        model.eval()
        with torch.no_grad():
            outputs_infer = model(input_ids, use_cache=True)
        
        print(f"  ✅ logits形状: {outputs_infer.logits.shape}")
        print(f"  ✅ past_key_values类型: {type(outputs_infer.past_key_values)}")
        
        if outputs_infer.past_key_values is not None:
            from mamba_ssm.utils.generation import InferenceParams
            is_inference_params = isinstance(outputs_infer.past_key_values, InferenceParams)
            print(f"  ✅ inference_params: {is_inference_params}")
            if is_inference_params:
                print(f"  ✅ Mamba2 缓存已正确初始化")
        
        print(f"\n✅ 前向传播测试通过!")
        return True
        
    except Exception as e:
        print(f"\n❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_kv_cache():
    """测试3：Mamba2缓存机制"""
    print_separator("测试3: Mamba2缓存机制测试")
    
    config = PawletteConfig()
    model = PawletteModelLLM(config)
    model.eval()
    
    # 第一次调用：处理初始序列
    seq_len_1 = 10
    input_ids_1 = torch.randint(0, config.vocab_size, (1, seq_len_1))
    
    print(f"第一次调用: seq_len={seq_len_1}")
    
    try:
        with torch.no_grad():
            outputs_1 = model(input_ids_1, use_cache=True)
        
        print(f"  ✅ 输出logits形状: {outputs_1.logits.shape}")
        
        # 验证Mamba2缓存
        from mamba_ssm.utils.generation import InferenceParams
        if outputs_1.past_key_values is not None:
            is_inference_params = isinstance(outputs_1.past_key_values, InferenceParams)
            print(f"  ✅ Mamba2 缓存类型正确: {is_inference_params}")
            if is_inference_params:
                print(f"  ✅ seqlen_offset: {outputs_1.past_key_values.seqlen_offset}")
        
        # 第二次调用：只处理1个新token，使用之前的缓存
        seq_len_2 = 1
        input_ids_2 = torch.randint(0, config.vocab_size, (1, seq_len_2))
        
        print(f"\n第二次调用: seq_len={seq_len_2}（使用缓存）")
        
        with torch.no_grad():
            outputs_2 = model(
                input_ids_2,
                past_key_values=outputs_1.past_key_values,
                use_cache=True
            )
        
        print(f"  ✅ 输出logits形状: {outputs_2.logits.shape}")
        
        # 验证缓存更新
        if outputs_2.past_key_values is not None:
            expected_offset = seq_len_1 + seq_len_2
            actual_offset = outputs_2.past_key_values.seqlen_offset
            print(f"  ✅ seqlen_offset: {actual_offset} (期望: {expected_offset})")
            if actual_offset == expected_offset:
                print(f"  ✅ 缓存偏移量正确!")
            else:
                print(f"  ❌ 缓存偏移量不正确!")
                return False
        
        print(f"\n✅ Mamba2缓存机制测试通过!")
        return True
        
    except Exception as e:
        print(f"\n❌ Mamba2缓存测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation():
    """测试4：生成功能"""
    print_separator("测试4: 生成功能测试")
    
    config = PawletteConfig()
    model = PawletteModelLLM(config)
    model.eval()
    
    batch_size = 1
    prompt_len = 10
    max_new_tokens = 20
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, prompt_len))
    
    print(f"输入长度: {prompt_len}")
    print(f"生成长度: {max_new_tokens}")
    
    try:
        # 贪婪解码
        print("\n贪婪解码（temperature=0）:")
        with torch.no_grad():
            outputs_greedy = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=config.pad_token_id,
                eos_token_id=config.eos_token_id,
            )
        
        expected_len = prompt_len + max_new_tokens
        actual_len = outputs_greedy.shape[1]
        
        print(f"  期望长度: {expected_len}")
        print(f"  实际长度: {actual_len}")
        
        if actual_len <= expected_len:
            print(f"  ✅ 生成成功（可能提前遇到EOS）")
        else:
            print(f"  ❌ 生成长度超出预期")
            return False
        
        # 采样解码
        print("\n采样解码（temperature=0.8）:")
        with torch.no_grad():
            outputs_sample = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=config.pad_token_id,
                eos_token_id=config.eos_token_id,
            )
        
        actual_len_sample = outputs_sample.shape[1]
        print(f"  期望长度: {expected_len}")
        print(f"  实际长度: {actual_len_sample}")
        
        if actual_len_sample <= expected_len:
            print(f"  ✅ 生成成功（可能提前遇到EOS）")
        else:
            print(f"  ❌ 生成长度超出预期")
            return False
        
        print(f"\n✅ 生成功能测试通过!")
        return True
        
    except Exception as e:
        print(f"\n❌ 生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_attention_mask():
    """测试5：注意力mask处理"""
    print_separator("测试5: 注意力Mask测试")
    
    config = PawletteConfig()
    model = PawletteModelLLM(config)
    model.eval()
    
    # 创建有padding的输入
    pad_token_id = config.pad_token_id
    
    input_ids = torch.tensor([
        [1, 2, 3, 4, 5, pad_token_id, pad_token_id, pad_token_id],  # 有padding
        [1, 2, 3, 4, 5, 6, 7, 8],  # 无padding
    ])
    
    attention_mask = (input_ids != pad_token_id).long()
    
    print(f"输入形状: {input_ids.shape}")
    print(f"第1个样本有效长度: {attention_mask[0].sum().item()}")
    print(f"第2个样本有效长度: {attention_mask[1].sum().item()}")
    
    try:
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        
        print(f"\n✅ 输出logits形状: {outputs.logits.shape}")
        print(f"✅ 注意力mask测试通过!")
        return True
        
    except Exception as e:
        print(f"\n❌ 注意力mask测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parameter_count():
    """测试6：参数量统计"""
    print_separator("测试6: 参数量统计")
    
    config = PawletteConfig()
    model = PawletteModelLLM(config)
    
    params = count_parameters(model)
    
    print(f"总参数量: {params['total_M']:.2f}M")
    print(f"可训练参数: {params['trainable_M']:.2f}M")
    
    # 分层统计
    print(f"\n分层参数量统计:")
    
    # 嵌入层
    embed_params = sum(p.numel() for p in model.model.embed_tokens.parameters())
    print(f"  嵌入层: {embed_params/1e6:.2f}M")
    
    # Mamba层
    mamba_params = 0
    
    for i, layer in enumerate(model.model.layers):
        layer_params = sum(p.numel() for p in layer.parameters())
        mamba_params += layer_params
        if i == 0:  # 只打印第一个Mamba层作为示例
            print(f"  层{i:2d} (Mamba): {layer_params/1e6:.2f}M")
    
    print(f"  ... (省略其他Mamba层)")
    print(f"\n  总Mamba参数: {mamba_params/1e6:.2f}M")
    
    # 输出层
    output_params = sum(p.numel() for p in model.lm_head.parameters())
    print(f"  输出层: {output_params/1e6:.2f}M")
    
    # 归一化层
    norm_params = sum(p.numel() for p in model.model.norm.parameters())
    print(f"  归一化层: {norm_params/1e3:.2f}K")
    
    print(f"\n✅ 参数量统计完成!")
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("  Pawlette纯Mamba2架构测试套件")
    print("="*60)
    
    tests = [
        ("架构布局验证", test_architecture_layout),
        ("前向传播测试", test_forward_pass),
        ("KV缓存机制测试", test_kv_cache),
        ("生成功能测试", test_generation),
        ("注意力Mask测试", test_attention_mask),
        ("参数量统计", test_parameter_count),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ 测试 '{test_name}' 发生异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # 打印总结
    print_separator("测试总结")
    
    passed = 0
    failed = 0
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{status} - {test_name}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("\n🎉 所有测试通过！纯Mamba2架构运行正常。")
        return True
    else:
        print(f"\n⚠️ 有 {failed} 个测试失败，请检查代码。")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)


