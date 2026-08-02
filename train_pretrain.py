import os
import time
import math
import warnings
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from model.model_pawlette import PawletteConfig, PawletteModelLLM, count_parameters
from dataset.lm_dataset import PretrainDataset, dynamic_collate_fn

warnings.filterwarnings('ignore')

# 训练相关参数配置
CONFIG = {
    # 训练配置
    'epochs': 1,
    'batch_size': 32,  
    'learning_rate': 5e-4,
    'warmup_steps': 104, #指预热步数
    'accumulation_steps': 8,  # 增加梯度累积步数以保持有效批次大小
    'grad_clip': 1.0,
    'weight_decay': 0.01,

    # 数据配置
    'data_path': 'dataset/pretrain_data.jsonl',
    'max_seq_len': None,  # 不限制序列长度
    'tokenizer_path': 'model/',

    # 输出配置
    'out_dir': 'out',
    'log_interval': 80,
    'save_interval': 800,
    
    # 继续训练配置
    'continue_pretrain': False,
    'pretrained_path': None,
    'resume': False,
    'checkpoint_path': None,
    
    # 分布式训练
    'ddp': False,
    'num_workers': 4,
    
    # 其他配置
    'device': 'cuda:0',  # Pawlette必须使用GPU训练（Mamba2依赖CUDA）
    'seed': 42,
    'use_wandb': False,
    'wandb_project': 'Pawlette-Pretrain',
}

def Logger(content):
    """统一的日志输出函数"""
    try:
        if not CONFIG['ddp'] or dist.get_rank() == 0:
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

def save_checkpoint(epoch, step, model, optimizer, scaler, save_path, global_step=None):
    """保存检查点"""
    state = {
        'epoch': epoch,
        'step': step,
        'global_step': global_step,  # 新增：保存全局步数
        'model_state_dict': model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'config': model.module.config if isinstance(model, DistributedDataParallel) else model.config,
        'optimizer_type': 'adamw',  # 记录优化器类型
    }
    torch.save(state, save_path)
    Logger(f"✅ 已保存检查点至 {save_path}")

def load_checkpoint(model, optimizer, scaler, checkpoint_path, device, strict=True):
    """加载检查点（支持优化器类型切换）"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # 加载模型状态
    if isinstance(model, DistributedDataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    else:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

    # 检查优化器类型是否匹配
    saved_optimizer_type = checkpoint.get('optimizer_type', 'adamw')
    current_optimizer_type = 'adamw'

    if saved_optimizer_type != current_optimizer_type:
        Logger(f"⚠️  优化器类型不匹配: checkpoint使用{saved_optimizer_type}，当前配置{current_optimizer_type}")
        Logger("   重新初始化优化器状态")
        # 不加载优化器状态，使用新初始化的优化器
        # scaler状态仍然加载
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
    else:
        # 加载优化器和scaler状态
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    start_epoch = checkpoint.get('epoch', 0)
    start_step = checkpoint.get('step', 0)
    start_global_step = checkpoint.get('global_step', 0)  # 新增：加载全局步数
    Logger(f"✅ 已从 {checkpoint_path} 加载检查点")
    Logger(f"   继续训练: epoch={start_epoch}, step={start_step}, global_step={start_global_step}")

    return start_epoch, start_step, start_global_step

def train_epoch(epoch, start_step, model, train_loader, optimizer, scaler, 
                scheduler_fn, ctx, wandb=None, start_epoch=0, start_global_step=0):
    """训练一个epoch"""
    model.train()    
    total_loss = 0
    start_time = time.time()
    first_step = True  # 标记是否是第一个实际执行的步骤
    executed_steps = 0  # 实际执行的步数计数器
    accumulation_counter = 0  # 梯度累积计数器
    
    # 使用checkpoint/上层传入的全局步数作为唯一真值
    current_global_step = start_global_step
    
    for step, (input_ids, labels, attention_mask) in enumerate(train_loader):
        # 跳过已训练的步骤（用于断点续训）
        if epoch == start_epoch and step < start_step:
            # 跳过时不递增global_step；global_step仅计数“实际执行过”的batch
            continue
        
        # 如果是第一个实际执行的步骤，重新设置开始时间
        if first_step:
            start_time = time.time()
            first_step = False
        
        # 增加实际执行的步数计数
        executed_steps += 1
        accumulation_counter += 1
        
        input_ids = input_ids.to(CONFIG['device'])
        labels = labels.to(CONFIG['device'])
        attention_mask = attention_mask.to(CONFIG['device'])
        
        # 使用正确的全局步数
        global_step = current_global_step
        
        # 前向传播 - 使用LongSSM隐藏状态复用机制
        # 在batch内,第一个序列零初始化,后续序列复用前一个序列的隐藏状态以改善长度外推
        batch_size = input_ids.size(0)
        all_losses = []
        
        with ctx:
            # 初始化inference_params用于存储隐藏状态
            from mamba_ssm.utils.generation import InferenceParams
            inference_params = None
            
            for batch_idx in range(batch_size):
                # 获取当前序列
                current_input_ids = input_ids[batch_idx:batch_idx+1]  # [1, seq_len]
                current_labels = labels[batch_idx:batch_idx+1]  # [1, seq_len]

                # 第一个序列使用零初始化(inference_params=None),后续序列复用隐藏状态
                if batch_idx == 0:
                    inference_params = None  # 第一个序列零初始化
                # else: 复用上一个序列的inference_params

                # 前向传播
                # Mamba2不需要attention_mask（因为labels中已经通过-100标记了需要计算loss的位置）
                outputs = model(
                    input_ids=current_input_ids,
                    labels=current_labels,
                    attention_mask=None,  # Mamba2架构不需要attention_mask
                    inference_params=inference_params,
                    use_cache=True  # 启用缓存以保存隐藏状态
                )
                
                all_losses.append(outputs.loss)
                
                # 更新inference_params供下一个序列使用
                inference_params = outputs.past_key_values
                if inference_params is not None:
                    # 重置seqlen_offset,让下一个序列从头开始但复用隐藏状态
                    inference_params.seqlen_offset = 0
            
            # 计算batch的平均损失
            loss = torch.stack(all_losses).mean()
            loss = loss / CONFIG['accumulation_steps']# 梯度累积
        # 反向传播
        scaler.scale(loss).backward()
        
        # 更新学习率（每个batch都更新）
        # 使用实际学习率（已经根据预训练模型调整过）
        lr_mult = scheduler_fn(global_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = CONFIG['actual_learning_rate'] * lr_mult
        
        # 梯度累积步骤 - 使用独立计数器避免断点续训时错位
        if accumulation_counter % CONFIG['accumulation_steps'] == 0:
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
            
            # 优化器步骤
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        # 每个batch后都要递增global_step
        current_global_step += 1
        # 统计
        total_loss += loss.item() * CONFIG['accumulation_steps']
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
            Logger(
                f'Epoch:[{epoch+1}/{CONFIG["epochs"]}]({step}/{len(train_loader)}) '
                f'loss:{current_loss:.4f} lr:{current_lr:.2e} '
                f'eta:{remaining_time:.1f}min'
            )            
            # WandB日志
            if wandb is not None and (not CONFIG['ddp'] or dist.get_rank() == 0):
                wandb.log({
                    "train/loss": current_loss,
                    "train/lr": current_lr,
                    "train/global_step": global_step,
                })
        
        # 定期保存检查点
        if (step + 1) % CONFIG['save_interval'] == 0 and (not CONFIG['ddp'] or dist.get_rank() == 0):
            checkpoint_path = os.path.join(CONFIG['save_dir'], 'checkpoint_latest.pth')
            save_checkpoint(epoch, step + 1, model, optimizer, scaler, checkpoint_path, current_global_step)
            
            # 保存模型权重
            model_path = os.path.join(CONFIG['save_dir'], 'pawlette.pth')
            if isinstance(model, DistributedDataParallel):
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
            Logger(f"✅ 已保存模型权重至 {model_path}")
    # 使用实际执行的步数计算平均损失
    avg_loss = total_loss / max(1, executed_steps)
    return avg_loss, current_global_step  #返回最新的global_step

def init_model():
    """初始化模型和分词器"""
    config = PawletteConfig()
    Logger(f"📝 模型配置:")
    Logger(f"   - hidden_size: {config.hidden_size}")
    Logger(f"   - num_layers: {config.num_hidden_layers}")
    Logger(f"   - state_size: {config.state_size}")
    Logger(f"   - vocab_size: {config.vocab_size}")

    model = PawletteModelLLM(config)
    
    # 加载预训练权重（如果指定）
    loaded_pretrained = False  # 标记是否加载了预训练模型
    if CONFIG['continue_pretrain'] and CONFIG['pretrained_path']:
        if os.path.exists(CONFIG['pretrained_path']):
            Logger(f"📂 加载预训练模型: {CONFIG['pretrained_path']}")
            state_dict = torch.load(CONFIG['pretrained_path'], map_location=CONFIG['device'])
            model.load_state_dict(state_dict, strict=False)
            Logger("✅ 预训练模型加载成功")
            loaded_pretrained = True
        else:
            Logger(f"⚠️ 预训练模型文件不存在: {CONFIG['pretrained_path']}")
    
    # 加载分词器
    tokenizer_path = CONFIG['tokenizer_path']
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id

    # 移动模型到设备
    model = model.to(CONFIG['device'])
    # 统计参数量
    params = count_parameters(model)
    Logger(f"📊 模型参数量: {params['trainable_M']:.2f}M (可训练) / {params['total_M']:.2f}M (总计)")
    return model, tokenizer, config, loaded_pretrained

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
    """
    Pawlette模型预训练主函数
    只支持使用GPU训练（Mamba2依赖CUDA）
    """
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        raise RuntimeError(
            "❌ Pawlette需要GPU支持！\n"
            "Mamba2架构依赖CUDA kernel，无法在CPU上训练。\n"
        )
    Logger(f"🎮 检测到GPU: {torch.cuda.get_device_name(0)}")
    
    # 设置随机种子
    torch.manual_seed(CONFIG['seed'])
    torch.cuda.manual_seed(CONFIG['seed'])
    
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
    
    # 设置混合精度（固定使用bfloat16）
    ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
    Logger(f"🔢 训练精度: bfloat16 (混合精度训练)")
    
    # 初始化模型和分词器
    model, tokenizer, config, loaded_pretrained = init_model()
    
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
    
    # 优化器 - 根据是否使用预训练模型调整学习率
    actual_lr = CONFIG['learning_rate']
    if loaded_pretrained:
        actual_lr = CONFIG['learning_rate'] * 0.5  # 使用预训练模型时，学习率降低到50%
        Logger(f"🔧 检测到使用预训练模型初始化，学习率调整为: {actual_lr:.2e} (原学习率的50%)")
    
    # 保存实际学习率到CONFIG，供学习率调度器使用
    CONFIG['actual_learning_rate'] = actual_lr

    # 优化器 - 使用标准AdamW
    Logger("🔧 使用AdamW优化器")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=actual_lr,
        weight_decay=CONFIG['weight_decay'],
        betas=(0.9, 0.95),
    )
    
    # 混合精度训练（bfloat16 + CUDA）
    scaler = torch.cuda.amp.GradScaler()
    
    # 学习率调度器 - 基于batch步数（与old版本一致）
    total_steps = len(train_loader) * CONFIG['epochs']
    
    scheduler_fn = lambda step: get_cosine_schedule_with_warmup(
        step, CONFIG['warmup_steps'], total_steps, min_lr_ratio=0.1
    )
    
    # 断点续训 - 自动检测检查点文件
    start_epoch, start_step, start_global_step = 0, 0, 0
    
    # 自动检测检查点文件
    checkpoint_path = os.path.join(CONFIG['save_dir'], 'checkpoint_latest.pth')
    if os.path.exists(checkpoint_path):
        Logger(f"🔍 自动检测到检查点文件: {checkpoint_path}")
        Logger("🔄 自动启用断点续训...")
        start_epoch, start_step, start_global_step = load_checkpoint(
            model, optimizer, scaler, checkpoint_path, CONFIG['device']
        )
    elif CONFIG['resume'] and CONFIG['checkpoint_path']:
        # 如果手动指定了检查点路径
        checkpoint_path = CONFIG['checkpoint_path']
        if os.path.exists(checkpoint_path):
            start_epoch, start_step, start_global_step = load_checkpoint(
                model, optimizer, scaler, checkpoint_path, CONFIG['device']
            )
        else:
            Logger(f"⚠️ 未找到检查点文件: {checkpoint_path}")
    else:
        # 检查是否已存在训练好的模型文件
        existing_models = []
        model_files = [
            'pawlette.pth',
        ]
        
        for model_file in model_files:
            model_path = os.path.join(CONFIG['save_dir'], model_file)
            if os.path.exists(model_path):
                existing_models.append(model_file)
        
        if existing_models:
            Logger("⚠️ 检测到已存在的模型文件:")
            for model_file in existing_models:
                Logger(f"   - {model_file}")
            Logger("")
            Logger("请选择操作:")
            Logger("  1. 加载 pawlette.pth 作为初始化参数，从头开始训练")
            Logger("  2. 终止训练（防止覆盖）")
            Logger("")
            
            # 只在主进程询问用户
            if not CONFIG['ddp'] or dist.get_rank() == 0:
                try:
                    choice = input("请输入选择 (1/2): ").strip()
                    
                    if choice == "1":
                        Logger("✅ 用户选择：加载 pawlette.pth 作为初始化参数")
                        model_path = os.path.join(CONFIG['save_dir'], 'pawlette.pth')
                        state_dict = torch.load(model_path, map_location=CONFIG['device'])
                        # 此时model还未被DDP包装，直接加载即可
                        model.load_state_dict(state_dict, strict=False)
                        Logger(f"📂 已加载模型权重: {model_path}")
                        Logger("🆕 开始从头训练（使用已有模型作为初始化）...")
                        
                        # 重新调整学习率（因为此时优化器已经创建）
                        actual_lr = CONFIG['learning_rate'] * 0.5
                        CONFIG['actual_learning_rate'] = actual_lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = actual_lr
                        Logger(f"🔧 检测到使用预训练模型初始化，学习率调整为: {actual_lr:.2e} (原学习率的50%)")
                    elif choice == "2":
                        Logger("🛑 用户选择：终止训练")
                        Logger("💡 如需继续训练，请:")
                        Logger("   1. 删除或重命名现有模型文件")
                        Logger("   2. 或者使用 --resume 参数指定检查点文件")
                        Logger("   3. 或者修改输出目录 --out_dir")
                        return
                    else:
                        Logger("❌ 无效的选择，训练已终止")
                        return
                except (KeyboardInterrupt, EOFError):
                    Logger("\n🛑 用户取消操作，训练已终止")
                    return
            else:
                # 非主进程等待主进程的决定
                # 这里可以通过分布式通信同步决定，但简化处理
                pass
        else:
            Logger("🆕 未检测到检查点文件和模型文件，开始全新训练...")
    
    # DDP包装
    if CONFIG['ddp']:
        model = DistributedDataParallel(model, device_ids=[CONFIG['ddp_local_rank']])
    
    #初始化当前的全局步数跟踪
    current_global_step = start_global_step
    
    # 训练循环
    Logger("🚀 开始训练...")
    for epoch in range(start_epoch, CONFIG['epochs']):
        if CONFIG['ddp'] and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # 训练一个epoch
        avg_loss, current_global_step = train_epoch(
            epoch, 
            start_step if epoch == start_epoch else 0,
            model, train_loader, optimizer, scaler,
            scheduler_fn, ctx, wandb, start_epoch, current_global_step
        )        
        Logger(f"📈 Epoch {epoch+1}/{CONFIG['epochs']} - 平均损失: {avg_loss:.4f}")
        
        # 记录训练损失到WandB
        if wandb is not None and (not CONFIG['ddp'] or dist.get_rank() == 0):
            wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch})
        
        # 保存epoch检查点
        if not CONFIG['ddp'] or dist.get_rank() == 0:
            checkpoint_path = os.path.join(CONFIG['save_dir'], f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(epoch + 1, 0, model, optimizer, scaler, checkpoint_path, current_global_step)
    
    # 保存最终模型
    if not CONFIG['ddp'] or dist.get_rank() == 0:
        final_model_path = os.path.join(CONFIG['save_dir'], 'pawlette.pth')
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