import argparse
import random
import warnings
import numpy as np
import torch
from model.model_pawlette import PawletteConfig, PawletteForCausalLM
from tokenizers import Tokenizer
from typing import List, Dict

warnings.filterwarnings('ignore')


def manual_generate(model, tokenizer_wrapper, input_ids, max_new_tokens=50, temperature=0.85, top_p=0.85):
    """手动生成函数，避免transformers generate方法的兼容性问题"""
    model.eval()
    device = input_ids.device
    generated_ids = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            generated_ids = generated_ids.to(device)
            outputs = model(generated_ids)
            logits = outputs.logits[:, -1, :]

            if temperature > 0:
                logits = logits / temperature

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == tokenizer_wrapper.eos_token_id:
                break

            generated_ids = torch.cat([generated_ids, next_token], dim=1)

    return generated_ids


class TokenizerWrapper:
    """包装 tokenizers.Tokenizer 以适配 transformers 接口"""
    def __init__(self, tokenizer_path: str = "./model/tokenizer.json"):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.bos_token_id = self.tokenizer.token_to_id("<s>")  # 根据实际 tokenizer 调整
        self.eos_token_id = self.tokenizer.token_to_id("</s>")
        self.pad_token_id = self.tokenizer.token_to_id("<pad>")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids
        if add_special_tokens and self.bos_token_id is not None:
            ids = [self.bos_token_id] + ids
        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            token_ids = [tid for tid in token_ids if tid not in [self.bos_token_id, self.eos_token_id, self.pad_token_id]]
        return self.tokenizer.decode(token_ids)

    def apply_chat_template(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        # 简单拼接对话模板（需根据实际模板调整）
        prompt = ""
        for msg in messages:
            if msg["role"] == "user":
                prompt += f"[INST] {msg['content']} [/INST]"
            elif msg["role"] == "assistant":
                prompt += f" {msg['content']} </s>"
        if add_generation_prompt:
            prompt += " "
        return prompt


def init_model(args):
    # 使用自定义 TokenizerWrapper 替代 AutoTokenizer
    tokenizer = TokenizerWrapper("./model/tokenizer.json")

    if args.load == 0:
        config = PawletteConfig()
        modes = {0: 'pretrain', 1: 'full_sft', 2: 'rlhf', 3: 'reason', 4: 'grpo'}
        if args.model_mode == 0:
            ckp = f'./{args.out_dir}/pawlette/pawlette.pth'
        else:
            ckp = f'./{args.out_dir}/{modes[args.model_mode]}_{config.hidden_size}.pth'

        model = PawletteForCausalLM(config)
        model.load_state_dict(torch.load(ckp, map_location='cpu'), strict=False)
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
                '请为我讲解“大语言模型”这个概念。',
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

        messages = messages[-args.history_cnt:] if args.history_cnt else []
        messages.append({"role": "user", "content": prompt})

        new_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True) if args.model_mode != 0 else prompt

        # 手动编码
        input_ids = tokenizer.encode(new_prompt, add_special_tokens=True)
        inputs = torch.tensor([input_ids], device=args.device)

        print('🤖️: ', end='')

        try:
            generated_ids = manual_generate(
                model,
                tokenizer,
                inputs,
                max_new_tokens=min(args.max_seq_len, 100),
                temperature=args.temperature,
                top_p=args.top_p
            )
            response_ids = generated_ids[0][inputs.shape[1]:].tolist()
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            print(response)
        except Exception as e:
            print(f"生成失败: {e}")
            response = "生成失败"
        messages.append({"role": "assistant", "content": response})
        print('\n')


if __name__ == "__main__":
    main()