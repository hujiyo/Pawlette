import json
import random
import re

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import os
import ast

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def dynamic_collate_fn(batch):
    """
    动态填充的collate函数，用于处理变长序列
    对Mamba架构特别优化：只填充到批次内最大长度，不浪费计算
    
    🔧 标准化：使用Hugging Face标准的-100填充labels
    """
    input_ids_list, labels_list, loss_mask_list = zip(*batch)
    
    # 找到批次内的最大长度
    max_len = max(x.size(0) for x in input_ids_list)
    
    # 动态填充到批次内最大长度
    batch_size = len(input_ids_list)
    device = input_ids_list[0].device if input_ids_list[0].is_cuda else torch.device('cpu')
    
    # 初始化填充后的张量 - 使用pad_token_id填充input_ids
    input_ids_padded = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    labels_padded = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)  # 🔧 填充-100
    loss_mask_padded = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    
    # 填充数据
    for i, (input_ids, labels, mask) in enumerate(zip(input_ids_list, labels_list, loss_mask_list)):
        seq_len = input_ids.size(0)
        input_ids_padded[i, :seq_len] = input_ids
        labels_padded[i, :seq_len] = labels
        loss_mask_padded[i, :seq_len] = mask
    
    return input_ids_padded, labels_padded, loss_mask_padded


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # 构建输入文本
        if self.max_length is None:
            # 不限制长度，不截断不填充
            encoding = self.tokenizer(
                str(sample['text']),
                return_tensors='pt'
            )
            input_ids = encoding.input_ids.squeeze()
            loss_mask = torch.ones_like(input_ids, dtype=torch.long)
        else:
            # 限制长度，截断和填充
            encoding = self.tokenizer(
                str(sample['text']),
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoding.input_ids.squeeze()
            loss_mask = (input_ids != self.tokenizer.pad_token_id)

        # 🔧 标准化：不在数据集中移位，让模型自动处理
        # 将pad token位置设为-100（Hugging Face标准）
        labels = torch.where(input_ids == self.tokenizer.pad_token_id, -100, input_ids)
        
        return input_ids, labels, loss_mask


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        # 使用新的特殊token
        self.asst_start_id = tokenizer('[ASST]', add_special_tokens=False).input_ids
        self.end_id = tokenizer('[END]', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """构建符合新tokenizer格式的对话"""
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.asst_start_id)] == self.asst_start_id:
                start = i + len(self.asst_start_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.end_id)] == self.end_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.end_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.end_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建对话提示
        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 生成动态损失掩码
        loss_mask = self._generate_loss_mask(input_ids)

        # 🔧 标准化：不在数据集中移位，让模型自动处理
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.where(input_ids == self.tokenizer.pad_token_id, -100, input_ids)
        loss_mask = torch.tensor(loss_mask, dtype=torch.long)

        return input_ids, labels, loss_mask


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        # 使用新的特殊token
        self.asst_start_id = tokenizer('[ASST]', add_special_tokens=False).input_ids
        self.end_id = tokenizer('[END]', add_special_tokens=False).input_ids
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = []
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        chosen = item['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = item['rejected']  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)
        # 🔧 标准化：不在数据集中移位，让模型自动处理
        x_chosen = torch.tensor(chosen_input_ids, dtype=torch.long)
        y_chosen = torch.where(torch.tensor(chosen_input_ids) == self.padding, -100, torch.tensor(chosen_input_ids))
        mask_chosen = torch.tensor(chosen_loss_mask, dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids, dtype=torch.long)
        y_rejected = torch.where(torch.tensor(rejected_input_ids) == self.padding, -100, torch.tensor(rejected_input_ids))
        mask_rejected = torch.tensor(rejected_loss_mask, dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.asst_start_id)] == self.asst_start_id:
                start = i + len(self.asst_start_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.end_id)] == self.end_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.end_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.end_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        # 使用新的特殊token
        self.asst_start_id = tokenizer('[ASST]', add_special_tokens=False).input_ids
        self.end_id = tokenizer('[END]', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """构建符合新tokenizer格式的对话"""
        messages = []
        answer = ''
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
            answer = turn['content']
        return self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True
        ), answer

    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建对话提示
        prompt, answer = self._create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': answer
        }


if __name__ == "__main__":
    pass
