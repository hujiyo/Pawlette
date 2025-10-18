import argparse
import random
import warnings
import numpy as np
import torch
from model.model_pawlette import PawletteConfig, PawletteModelLLM
from transformers import AutoTokenizer, TextStreamer
from typing import List, Dict
warnings.filterwarnings('ignore')

def apply_chat_template_simple(messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
    """使用与tokenizer_config.json一致的对话模板。"""
    prompt = ""    
    # 处理系统消息（如果有）
    if messages and messages[0]["role"] == "system":
        prompt += f"[SYS]{messages[0]['content']}[/SYS]\n"
        messages = messages[1:]    
    # 处理对话历史
    for msg in messages:
        if msg["role"] == "user":
            prompt += f"[OTHER]user[SEP]{msg['content']}[/OTHER]\n"
        elif msg["role"] == "assistant":
            prompt += f"[AI]{msg['content']}[/AI]\n"    
    # 添加生成提示符
    if add_generation_prompt and messages and messages[-1]["role"] == "user":
        prompt += "[AI]"    
    return prompt

def init_model(args):
    # 纯官方方式加载分词器
    tokenizer = AutoTokenizer.from_pretrained('./model/', use_fast=True)
    if args.load == 0:
        config = PawletteConfig()
        modes = {0: 'pretrain', 1: 'full_sft', 2: 'rlhf'}
        if args.model_mode == 0:
            ckp = f'./{args.out_dir}/pawlette/pawlette.pth'
        else:
            ckp = f'./{args.out_dir}/{modes[args.model_mode]}_{config.hidden_size}.pth'

        model = PawletteModelLLM(config)
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=False)
        
        # 将tokenizer的特殊token对齐到模型config（避免警告）
        tokenizer.pad_token_id = model.config.pad_token_id
        tokenizer.eos_token_id = model.config.eos_token_id
        tokenizer.bos_token_id = model.config.bos_token_id
        
        # 预训练模式：禁用自动添加特殊token
        if args.model_mode == 0:
            tokenizer.add_bos_token = False
            tokenizer.add_eos_token = False
        
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
    parser.add_argument('--device', default='cuda', type=str, help='Pawlette需要CUDA支持（Mamba2依赖）')
    parser.add_argument('--max_seq_len', default=8192, type=int)
    parser.add_argument('--history_cnt', default=0, type=int)
    parser.add_argument('--load', default=0, type=int, help="0: 原生torch权重，1: transformers加载")
    parser.add_argument('--model_mode', default=0, type=int,
                        help="0: 预训练模型，1: SFT-Chat模型，2: RLHF-Chat模型")
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

        # 编码输入（预训练模式不添加特殊token）
        if args.model_mode == 0:
            # 预训练模式：完全不添加任何特殊token
            input_ids = tokenizer.encode(new_prompt, add_special_tokens=False)
        else:
            # 指令微调模式：添加特殊token
            input_ids = tokenizer.encode(new_prompt, add_special_tokens=True)
        inputs = torch.tensor([input_ids], device=args.device)

        print('🤖️: ', end='', flush=True)

        # 使用transformers官方的TextStreamer进行流式输出
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        try:
            # 使用模型的标准generate方法，支持官方TextStreamer
            if args.temperature == 0.0:
                # 温度为0时使用贪婪解码
                generated_ids = model.generate(
                    inputs,
                    max_new_tokens=min(args.max_seq_len - len(input_ids), 128),
                    do_sample=False,  # 贪婪解码
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    streamer=streamer,
                )
            else:
                # 温度>0时使用采样
                generated_ids = model.generate(
                    inputs,
                    max_new_tokens=min(args.max_seq_len - len(input_ids), 128),
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    streamer=streamer,
                )
            response_ids = generated_ids[0][inputs.shape[1]:].tolist()
            response_final = tokenizer.decode(response_ids, skip_special_tokens=True)
        except Exception as e:
            print(f"生成失败: {e}")
            response_final = "生成失败"
        
        # 只有在非测试模式下才添加助手回复到对话历史中
        if test_mode == 0 or args.history_cnt > 0:
            messages.append({"role": "assistant", "content": response_final})
        print('\n')

if __name__ == "__main__":
    main()