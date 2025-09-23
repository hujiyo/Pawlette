import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import math
import warnings
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from transformers import AutoTokenizer
from model.model_pawlette import PawletteConfig, PawletteForCausalLM, count_parameters
from dataset.lm_dataset import PretrainDataset, dynamic_collate_fn

warnings.filterwarnings('ignore')

# 全局配置 - 只包含训练相关参数
CONFIG = {
    
    # 训练配置
    'epochs': 1,
    'batch_size': 16,  # 减小批次大小以降低内存占用
    'learning_rate': 5e-4,
    'warmup_steps': 100, #指预热步数
    'accumulation_steps': 8,  # 增加梯度累积步数以保持有效批次大小
    'grad_clip': 1.0,
    'weight_decay': 0.01,
    
    # 数据配置
    'data_path': '../dataset/pretrain_data.jsonl',
    'eval_data_path': None,
    'max_seq_len': None,  # 不限制序列长度
    'tokenizer_path': '../model/',
    
    # 输出配置
    'out_dir': '../out',
    'log_interval': 100,
    'save_interval': 500,
    'eval_interval': 500,
    
    # 继续训练配置
    'continue_pretrain': False,
    'pretrained_path': None,
    'resume': False,
    'checkpoint_path': None,
    
    # 分布式训练
    'ddp': False,
    'num_workers': 4,
    
    # 其他配置
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'dtype': 'bfloat16',
    'seed': 42,
    'use_wandb': False,
    'wandb_project': 'Pawlette-Pretrain',
    
    # 内存优化配置
    'gradient_checkpointing': True,  # 启用梯度检查点以节省内存
}

# 全局变量
ddp = CONFIG['ddp']


def Logger(content):
    """统一的日志输出函数"""
    try:
        if not ddp or dist.get_rank() == 0:
            print(f"[Pawlette] {content}")
    except NameError:
        print(f"[Pawlette] {content}")


def get_cosine_schedule_with_warmup(current_step, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """余弦退火学习率调度（带warmup）"""
    if current_step < num_warmup_steps:
        # Warmup阶段
        return float(current_step) / float(max(1, num_warmup_steps))
    
    # 余弦退火阶段
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    lr_mult = min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_mult


def save_checkpoint(epoch, step, model, optimizer, scaler, best_loss, save_path):
    """保存检查点"""
    state = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_loss': best_loss,
        'config': model.module.config if isinstance(model, DistributedDataParallel) else model.config,
    }
    torch.save(state, save_path)
    Logger(f"✅ 已保存检查点至 {save_path}")


def load_checkpoint(model, optimizer, scaler, checkpoint_path, device, strict=True):
    """加载检查点"""
    # 添加安全的全局类以支持PyTorch 2.6的weights_only模式
    from model.model_pawlette import PawletteConfig

    # 如果 PyTorch 版本 >= 2.6，则添加安全全局类
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([PawletteConfig])
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 加载模型状态
    if isinstance(model, DistributedDataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    else:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    # 加载优化器和scaler状态
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0)
    start_step = checkpoint.get('step', 0)
    best_loss = checkpoint.get('best_loss', float('inf'))
    
    Logger(f"✅ 已从 {checkpoint_path} 加载检查点")
    Logger(f"   继续训练: epoch={start_epoch}, step={start_step}, best_loss={best_loss:.4f}")
    
    return start_epoch, start_step, best_loss


def evaluate_model(model, eval_loader, ctx, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for X, Y, loss_mask in eval_loader:
            X = X.to(device)
            Y = Y.to(device)
            loss_mask = loss_mask.to(device)
            
            with ctx:
                outputs = model(input_ids=X, labels=Y)
                
                # 使用mask计算损失
                if loss_mask is not None:
                    loss_values = nn.functional.cross_entropy(
                        outputs.logits.view(-1, outputs.logits.size(-1)),
                        Y.view(-1),
                        reduction='none'
                    ).view(Y.size())
                    loss = (loss_values * loss_mask).sum()
                    num_tokens = loss_mask.sum()
                else:
                    loss = outputs.loss * X.size(0) * X.size(1)
                    num_tokens = X.size(0) * X.size(1)
                
                total_loss += loss.item()
                total_tokens += num_tokens.item()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    model.train()
    return avg_loss


def train_epoch(epoch, start_step, model, train_loader, optimizer, scaler, 
                scheduler_fn, ctx, wandb=None, start_epoch=0):
    """训练一个epoch"""
    model.train()
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    
    total_loss = 0
    total_tokens = 0
    start_time = time.time()
    first_step = True  # 标记是否是第一个实际执行的步骤
    executed_steps = 0  # 实际执行的步数计数器
    
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 跳过已训练的步骤（用于断点续训）
        if epoch == start_epoch and step < start_step:
            continue
        
        # 如果是第一个实际执行的步骤，重新设置开始时间
        if first_step:
            start_time = time.time()
            first_step = False
        
        # 增加实际执行的步数计数
        executed_steps += 1
        
        X = X.to(CONFIG['device'])
        Y = Y.to(CONFIG['device'])
        loss_mask = loss_mask.to(CONFIG['device'])
        
        # 计算当前步的全局步数（考虑断点续训）
        if epoch == start_epoch:
            global_step = step
        else:
            global_step = epoch * len(train_loader) + step
        
        # 更新学习率
        lr_mult = scheduler_fn(global_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = CONFIG['learning_rate'] * lr_mult
        
        # 前向传播
        with ctx:
            outputs = model(input_ids=X, labels=Y)
            
            # 使用mask计算损失
            if loss_mask is not None:
                loss_values = loss_fct(
                    outputs.logits.view(-1, outputs.logits.size(-1)),
                    Y.view(-1)
                ).view(Y.size())
                loss = (loss_values * loss_mask).sum() / loss_mask.sum()
            else:
                loss = outputs.loss
            
            # 梯度累积
            loss = loss / CONFIG['accumulation_steps']
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 梯度累积步骤
        if (step + 1) % CONFIG['accumulation_steps'] == 0:
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
            
            # 优化器步骤
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        # 统计
        total_loss += loss.item() * CONFIG['accumulation_steps']
        if loss_mask is not None:
            total_tokens += loss_mask.sum().item()
        else:
            total_tokens += X.numel()
        
        # 日志输出
        if step % CONFIG['log_interval'] == 0:
            elapsed_time = time.time() - start_time
            
            if executed_steps > 0:
                avg_time_per_step = elapsed_time / executed_steps
                # 计算剩余步数
                remaining_steps = len(train_loader) - step - 1
                remaining_time = avg_time_per_step * remaining_steps / 60
            else:
                remaining_time = 0
            
            current_loss = loss.item() * CONFIG['accumulation_steps']
            current_lr = optimizer.param_groups[0]['lr']
            tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0
            
            Logger(
                f'Epoch:[{epoch+1}/{CONFIG["epochs"]}]({step}/{len(train_loader)}) '
                f'loss:{current_loss:.4f} lr:{current_lr:.2e} '
                f'tokens/s:{tokens_per_sec:.0f} eta:{remaining_time:.1f}min'
            )
            
            # WandB日志
            if wandb is not None and (not ddp or dist.get_rank() == 0):
                wandb.log({
                    "train/loss": current_loss,
                    "train/lr": current_lr,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/global_step": global_step,
                })
        
        # 定期保存检查点
        if (step + 1) % CONFIG['save_interval'] == 0 and (not ddp or dist.get_rank() == 0):
            checkpoint_path = os.path.join(CONFIG['save_dir'], 'checkpoint_latest.pth')
            save_checkpoint(epoch, step + 1, model, optimizer, scaler, total_loss / (step + 1), checkpoint_path)
            
            # 保存模型权重
            model_path = os.path.join(CONFIG['save_dir'], 'pawlette.pth')
            if isinstance(model, DistributedDataParallel):
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
            Logger(f"✅ 已保存模型权重至 {model_path}")
    
    # 使用实际执行的步数计算平均损失
    avg_loss = total_loss / max(1, executed_steps)
    return avg_loss


def init_model():
    """初始化模型和分词器"""
    
    # 直接使用PawletteConfig的默认配置
    config = PawletteConfig()
    
    Logger(f"📝 模型配置:")
    Logger(f"   - hidden_size: {config.hidden_size}")
    Logger(f"   - num_layers: {config.num_hidden_layers}")
    Logger(f"   - state_size: {config.state_size}")
    Logger(f"   - vocab_size: {config.vocab_size}")
    
    # 创建模型
    model = PawletteForCausalLM(config)
    
    # 加载预训练权重（如果指定）
    if CONFIG['continue_pretrain'] and CONFIG['pretrained_path']:
        if os.path.exists(CONFIG['pretrained_path']):
            Logger(f"📂 加载预训练模型: {CONFIG['pretrained_path']}")
            state_dict = torch.load(CONFIG['pretrained_path'], map_location=CONFIG['device'])
            model.load_state_dict(state_dict, strict=False)
            Logger("✅ 预训练模型加载成功")
        else:
            Logger(f"⚠️ 预训练模型文件不存在: {CONFIG['pretrained_path']}")
    
    # 加载分词器
    if os.path.exists(CONFIG['tokenizer_path']):
        tokenizer = AutoTokenizer.from_pretrained(CONFIG['tokenizer_path'], trust_remote_code=True)
    else:
        Logger(f"⚠️ 使用默认分词器，未找到: {CONFIG['tokenizer_path']}")
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    
    # 移动模型到设备
    model = model.to(CONFIG['device'])
    
    # 启用梯度检查点以节省内存
    if CONFIG['gradient_checkpointing']:
        model.gradient_checkpointing_enable()
        Logger("✅ 已启用梯度检查点以优化内存使用")
    
    # 统计参数量
    params = count_parameters(model)
    Logger(f"📊 模型参数量: {params['trainable_M']:.2f}M (可训练) / {params['total_M']:.2f}M (总计)")
    
    return model, tokenizer, config


def init_distributed_mode():
    """初始化分布式训练"""
    if not CONFIG['ddp']:
        return
    
    dist.init_process_group(backend="nccl")
    CONFIG['ddp_rank'] = int(os.environ["RANK"])
    CONFIG['ddp_local_rank'] = int(os.environ["LOCAL_RANK"])
    CONFIG['ddp_world_size'] = int(os.environ["WORLD_SIZE"])
    CONFIG['device'] = f"cuda:{CONFIG['ddp_local_rank']}"
    torch.cuda.set_device(CONFIG['device'])
    
    Logger(f"🌐 分布式训练初始化: rank={CONFIG['ddp_rank']}, world_size={CONFIG['ddp_world_size']}")


def main():
    """主训练函数"""
    global ddp
    
    # 设置随机种子
    torch.manual_seed(CONFIG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CONFIG['seed'])
    
    # 设置全局变量
    ddp = CONFIG['ddp']
    
    # 初始化分布式训练
    if CONFIG['ddp']:
        init_distributed_mode()
    
    # 创建输出目录
    CONFIG['save_dir'] = os.path.join(CONFIG['out_dir'], "pawlette")
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    
    # 初始化WandB
    wandb = None
    if CONFIG['use_wandb'] and (not CONFIG['ddp'] or dist.get_rank() == 0):
        import wandb
        run_name = f"Pawlette-bs{CONFIG['batch_size']}-lr{CONFIG['learning_rate']}"
        wandb.init(project=CONFIG['wandb_project'], name=run_name, config=CONFIG)
    
    Logger("🐾 Pawlette预训练开始")
    Logger(f"📁 输出目录: {CONFIG['save_dir']}")
    
    # 设置设备和混合精度
    device_type = "cuda" if "cuda" in CONFIG['device'] else "cpu"
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[CONFIG['dtype']]
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=torch_dtype)
    
    # 初始化模型和分词器
    model, tokenizer, config = init_model()
    
    
    # 准备数据集
    train_dataset = PretrainDataset(CONFIG['data_path'], tokenizer, max_length=CONFIG['max_seq_len'])
    Logger(f"📚 训练数据集大小: {len(train_dataset)}")
    if CONFIG['max_seq_len'] is None:
        Logger("📏 序列长度: 无限制（动态长度）")
    else:
        Logger(f"📏 序列长度: {CONFIG['max_seq_len']}")
    
    # 数据加载器
    train_sampler = DistributedSampler(train_dataset) if CONFIG['ddp'] else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
        drop_last=True,
        collate_fn=dynamic_collate_fn if CONFIG['max_seq_len'] is None else None,
    )
    
    # 验证数据集（如果有）
    eval_loader = None
    if CONFIG['eval_data_path'] and os.path.exists(CONFIG['eval_data_path']):
        eval_dataset = PretrainDataset(CONFIG['eval_data_path'], tokenizer, max_length=CONFIG['max_seq_len'])
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=False,
            num_workers=CONFIG['num_workers'],
            pin_memory=True,
            collate_fn=dynamic_collate_fn if CONFIG['max_seq_len'] is None else None,
        )
        Logger(f"📚 验证数据集大小: {len(eval_dataset)}")
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        betas=(0.9, 0.95),
    )
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=(CONFIG['dtype'] in ['float16', 'bfloat16']))
    
    # 学习率调度器
    total_steps = len(train_loader) * CONFIG['epochs']
    scheduler_fn = lambda step: get_cosine_schedule_with_warmup(
        step, CONFIG['warmup_steps'], total_steps, min_lr_ratio=0.1
    )
    
    # 断点续训 - 自动检测检查点文件
    start_epoch, start_step = 0, 0
    best_loss = float('inf')
    
    # 自动检测检查点文件
    checkpoint_path = os.path.join(CONFIG['save_dir'], 'checkpoint_latest.pth')
    if os.path.exists(checkpoint_path):
        Logger(f"🔍 自动检测到检查点文件: {checkpoint_path}")
        Logger("🔄 自动启用断点续训...")
        start_epoch, start_step, best_loss = load_checkpoint(
            model, optimizer, scaler, checkpoint_path, CONFIG['device']
        )
    elif CONFIG['resume'] and CONFIG['checkpoint_path']:
        # 如果手动指定了检查点路径
        checkpoint_path = CONFIG['checkpoint_path']
        if os.path.exists(checkpoint_path):
            start_epoch, start_step, best_loss = load_checkpoint(
                model, optimizer, scaler, checkpoint_path, CONFIG['device']
            )
        else:
            Logger(f"⚠️ 未找到检查点文件: {checkpoint_path}")
    else:
        # 检查是否已存在训练好的模型文件
        existing_models = []
        model_files = [
            'pawlette.pth',
            'pawlette_best.pth', 
            'pawlette_final.pth'
        ]
        
        for model_file in model_files:
            model_path = os.path.join(CONFIG['save_dir'], model_file)
            if os.path.exists(model_path):
                existing_models.append(model_file)
        
        if existing_models:
            Logger("⚠️ 检测到已存在的模型文件:")
            for model_file in existing_models:
                Logger(f"   - {model_file}")
            Logger("🛑 为防止覆盖已训练的模型，训练已停止！")
            Logger("💡 如需继续训练，请:")
            Logger("   1. 删除或重命名现有模型文件")
            Logger("   2. 或者使用 --resume 参数指定检查点文件")
            Logger("   3. 或者修改输出目录 --out_dir")
            return
        else:
            Logger("🆕 未检测到检查点文件和模型文件，开始全新训练...")
    
    # DDP包装
    if CONFIG['ddp']:
        model = DistributedDataParallel(model, device_ids=[CONFIG['ddp_local_rank']])
    
    # 训练循环
    Logger("🚀 开始训练...")
    for epoch in range(start_epoch, CONFIG['epochs']):
        if CONFIG['ddp'] and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # 训练一个epoch
        avg_loss = train_epoch(
            epoch, 
            start_step if epoch == start_epoch else 0,
            model, train_loader, optimizer, scaler,
            scheduler_fn, ctx, wandb, start_epoch
        )
        
        Logger(f"📈 Epoch {epoch+1}/{CONFIG['epochs']} - 平均损失: {avg_loss:.4f}")
        
        # 评估（如果有验证集）
        if eval_loader is not None and (not CONFIG['ddp'] or dist.get_rank() == 0):
            eval_loss = evaluate_model(model, eval_loader, ctx, CONFIG['device'])
            Logger(f"📊 验证损失: {eval_loss:.4f}")
            
            if wandb is not None:
                wandb.log({"eval/loss": eval_loss, "epoch": epoch})
            
            # 保存最佳模型
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_model_path = os.path.join(CONFIG['save_dir'], 'pawlette_best.pth')
                if isinstance(model, DistributedDataParallel):
                    torch.save(model.module.state_dict(), best_model_path)
                else:
                    torch.save(model.state_dict(), best_model_path)
                Logger(f"🏆 保存最佳模型 (loss={best_loss:.4f})")
        
        # 保存epoch检查点
        if not CONFIG['ddp'] or dist.get_rank() == 0:
            checkpoint_path = os.path.join(CONFIG['save_dir'], f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(epoch + 1, 0, model, optimizer, scaler, best_loss, checkpoint_path)
    
    # 保存最终模型
    if not CONFIG['ddp'] or dist.get_rank() == 0:
        final_model_path = os.path.join(CONFIG['save_dir'], 'pawlette_final.pth')
        if isinstance(model, DistributedDataParallel):
            torch.save(model.module.state_dict(), final_model_path)
        else:
            torch.save(model.state_dict(), final_model_path)
        Logger(f"✅ 训练完成！最终模型保存至: {final_model_path}")
    
    # 清理
    if CONFIG['ddp']:
        dist.destroy_process_group()
    
    if wandb is not None:
        wandb.finish()
    
    Logger("🎉 Pawlette预训练完成！")


if __name__ == "__main__":
    main()