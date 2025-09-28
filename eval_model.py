import argparse
import random
import warnings
import numpy as np
import torch
from model.model_pawlette import PawletteConfig, PawletteModelLLM
from transformers import AutoTokenizer
from typing import List, Dict

warnings.filterwarnings('ignore')


def manual_generate_optimized(model, tokenizer, input_ids, max_new_tokens=4096, temperature=0.85, top_p=0.85):
    """优化的生成函数，专门针对Mamba架构

    关键优化：
    - 首步传入完整提示；后续步仅传入最后一个 token
    - 在循环中复用 outputs.past_key_values (InferenceParams) 作为缓存
    """
    model.eval()
    generated_ids = input_ids.clone()
    inference_params = None  # 循环内复用 Mamba2 缓存

    with torch.no_grad():
        for i in range(max_new_tokens):
            # 仅在第一步传完整序列；之后只传最后一个token，并复用缓存
            step_input = generated_ids if inference_params is None else generated_ids[:, -1:]

            outputs = model(
                step_input,
                use_cache=True,
                inference_params=inference_params,
            )
            logits = outputs.logits[:, -1, :]
            inference_params = outputs.past_key_values  # 复用缓存

            if temperature > 0:
                logits = logits / temperature

            if top_p < 1.0:
                probs = torch.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                # 保留使累计概率<=top_p的最小前缀
                cutoff = (cumulative_probs > top_p).float()
                cutoff[..., 1:] = cutoff[..., :-1].clone()
                cutoff[..., 0] = 0
                # 将被移除的概率置零并重新归一化
                filtered_probs = sorted_probs.masked_fill(cutoff.bool(), 0.0)
                filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
                # 从过滤后的分布采样，再映射回原索引
                next_sorted_index = torch.multinomial(filtered_probs, num_samples=1)
                next_token = sorted_indices.gather(1, next_sorted_index)
            else:
                if temperature == 0.0:
                    # 贪婪
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
            

            # 结束符
            eos_id = tokenizer.eos_token_id
            if eos_id is None:
                eos_id = getattr(model.config, 'eos_token_id', None)
            if eos_id is not None and next_token.item() == eos_id:
                break

            generated_ids = torch.cat([generated_ids, next_token], dim=1)

    return generated_ids


def apply_chat_template_simple(messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
    """简易对话模板拼接，避免依赖具体模型模板。"""
    prompt = ""
    for msg in messages:
        if msg["role"] == "user":
            prompt += f"[INST] {msg['content']} [/INST]"
        elif msg["role"] == "assistant":
            prompt += f" {msg['content']} </s>"
    if add_generation_prompt:
        prompt += " "
    return prompt

def _resize_embeddings_if_needed(model: PawletteModelLLM, tokenizer):
    """当分词器词表大于模型词表时，在推理侧安全扩展嵌入与输出头。"""
    with torch.no_grad():
        current_vocab = model.model.embed_tokens.num_embeddings
        vocab_size = getattr(tokenizer, 'vocab_size', current_vocab) or current_vocab
        if vocab_size <= current_vocab:
            return
        old_emb = model.model.embed_tokens.weight.data
        old_out = model.lm_head.weight.data
        hidden = old_emb.size(1)
        num_new = vocab_size - current_vocab
        mean = old_emb.mean().item()
        std = old_emb.std().item()
        new_emb_rows = mean + std * 0.02 * torch.randn(num_new, hidden, device=old_emb.device, dtype=old_emb.dtype)
        new_out_rows = mean + std * 0.02 * torch.randn(num_new, hidden, device=old_out.device, dtype=old_out.dtype)
        # 扩展嵌入
        new_emb = torch.cat([old_emb, new_emb_rows], dim=0)
        model.model.embed_tokens = torch.nn.Embedding.from_pretrained(new_emb, freeze=False)
        # 扩展输出头
        new_lm = torch.nn.Linear(hidden, vocab_size, bias=False)
        new_lm.weight.data[:old_out.size(0)] = old_out
        new_lm.weight.data[old_out.size(0):] = new_out_rows
        model.lm_head = new_lm


def init_model(args):
    # 纯官方方式加载分词器
    tokenizer = AutoTokenizer.from_pretrained('./model/', use_fast=True)

    if args.load == 0:
        config = PawletteConfig()
        modes = {0: 'pretrain', 1: 'full_sft', 2: 'rlhf', 3: 'reason', 4: 'grpo'}
        if args.model_mode == 0:
            ckp = f'./{args.out_dir}/pawlette/pawlette.pth'
        else:
            ckp = f'./{args.out_dir}/{modes[args.model_mode]}_{config.hidden_size}.pth'

        model = PawletteModelLLM(config)
        model.load_state_dict(torch.load(ckp, map_location='cpu'), strict=False)
        # 推理侧对齐词表大小（如分词器更大）
        _resize_embeddings_if_needed(model, tokenizer)
        # 对齐特殊 token id（若缺省则回落到 config）
        if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        if model.config.eos_token_id is None and tokenizer.eos_token_id is not None:
            model.config.eos_token_id = tokenizer.eos_token_id
        if model.config.bos_token_id is None and tokenizer.bos_token_id is not None:
            model.config.bos_token_id = tokenizer.bos_token_id
        model = model.to(args.device)
    else:
        raise NotImplementedError("transformers model loading not supported in this version.")

    print(f'Pawlette模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')
    return model.eval(), tokenizer


def get_prompt_datas(args):
    if args.model_mode == 0:
        prompt_datas = [
            '马克思主义基本原理',
            '人类大脑的主要功能',
            '万有引力原理是',
            '世界上最高的山峰是',
            '二氧化碳在空气中',
            '地球上最大的动物有',
            '杭州市的美食有'
        ]
    else:
        if args.lora_name == 'None':
            prompt_datas = [
                '请介绍一下自己。',
                '你更擅长哪一个学科？',
                '鲁迅的《狂人日记》是如何批判封建礼教的？',
                '我咳嗽已经持续了两周，需要去医院检查吗？',
                '详细的介绍光速的物理概念。',
                '推荐一些杭州的特色美食吧。',
                '请为我讲解"大语言模型"这个概念。',
                '如何理解ChatGPT？',
                'Introduce the history of the United States, please.'
            ]
        else:
            lora_prompt_datas = {
                'lora_identity': [
                    "你是ChatGPT吧。",
                    "你叫什么名字？",
                    "你和openai是什么关系？"
                ],
                'lora_medical': [
                    '我最近经常感到头晕，可能是什么原因？',
                    '我咳嗽已经持续了两周，需要去医院检查吗？',
                    '服用抗生素时需要注意哪些事项？',
                    '体检报告中显示胆固醇偏高，我该怎么办？',
                    '孕妇在饮食上需要注意什么？',
                    '老年人如何预防骨质疏松？',
                    '我最近总是感到焦虑，应该怎么缓解？',
                    '如果有人突然晕倒，应该如何急救？'
                ],
            }
            prompt_datas = lora_prompt_datas[args.lora_name]
    return prompt_datas


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Chat with Pawlette")
    parser.add_argument('--lora_name', default='None', type=str)
    parser.add_argument('--out_dir', default='out', type=str)
    parser.add_argument('--temperature', default=0.85, type=float)
    parser.add_argument('--top_p', default=0.85, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--max_seq_len', default=8192, type=int)
    parser.add_argument('--history_cnt', default=0, type=int)
    parser.add_argument('--load', default=0, type=int, help="0: 原生torch权重，1: transformers加载")
    parser.add_argument('--model_mode', default=0, type=int,
                        help="0: 预训练模型，1: SFT-Chat模型，2: RLHF-Chat模型，3: Reason模型，4: RLAIF-Chat模型")
    args = parser.parse_args()

    model, tokenizer = init_model(args)

    prompts = get_prompt_datas(args)

    try:
        test_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    except EOFError:
        test_mode = 0
        print('使用自动测试模式')

    try:
        temp_choice = int(input('[0] 使用默认温度 (0.85)\n[1] 设置自定义温度\n[2] 使用温度0 (可复现输出)\n'))
        if temp_choice == 1:
            custom_temp = float(input('请输入温度值 (0.0-2.0): '))
            args.temperature = max(0.0, min(2.0, custom_temp))
            print(f'使用自定义温度: {args.temperature}')
        elif temp_choice == 2:
            args.temperature = 0.0
            print('使用温度0，输出将完全可复现')
        else:
            print(f'使用默认温度: {args.temperature}')
    except (EOFError, ValueError):
        print(f'使用默认温度: {args.temperature}')

    messages = []
    for idx, prompt in enumerate(prompts if test_mode == 0 else iter(lambda: input('👶: '), '')):
        if args.temperature == 0.0:
            setup_seed(2025)
        else:
            setup_seed(random.randint(0, 2048))
        if test_mode == 0:
            print(f'👶: {prompt}')

        # 重置或维护对话历史
        messages = messages[-args.history_cnt:] if args.history_cnt else []
        messages.append({"role": "user", "content": prompt})

        # 根据模型模式选择提示格式
        new_prompt = apply_chat_template_simple(messages, add_generation_prompt=True) if args.model_mode != 0 else prompt

        # 编码输入
        input_ids = tokenizer.encode(new_prompt, add_special_tokens=True)
        inputs = torch.tensor([input_ids], device=args.device)

        print('🤖️: ', end='')

        try:
            generated_ids = manual_generate_optimized(
                model,
                tokenizer,
                inputs,
                max_new_tokens=min(args.max_seq_len - len(input_ids), 4096),
                temperature=args.temperature,
                top_p=args.top_p
            )
            response_ids = generated_ids[0][inputs.shape[1]:].tolist()
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            print(response)
        except Exception as e:
            print(f"生成失败: {e}")
            response = "生成失败"
        
        # 只有在非测试模式下才添加助手回复到对话历史中
        if test_mode == 0 or args.history_cnt > 0:
            messages.append({"role": "assistant", "content": response})
        
        print('\n')


if __name__ == "__main__":
    main()