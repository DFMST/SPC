import argparse
import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F

# 导入新模型和损失
from model import SPCModel
from losses import SPCLossCalculator
from utils import (
    create_dataloaders,
    visualize_batch,
    save_checkpoint,
    load_checkpoint,
    compute_semantic_metrics,
    generate_adverse_weather
)
import torch.hub

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description="Train SPC Framework for Adverse Weather Generation")

    # 数据参数
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to dataset directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--image_size", type=int, default=256, help="Input image size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--lr_g", type=float, default=2e-4, help="Generator learning rate")
    parser.add_argument("--lr_d", type=float, default=2e-4, help="Discriminator learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="Adam: beta1")
    parser.add_argument("--b2", type=float, default=0.999, help="Adam: beta2")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--save_interval", type=int, default=20, help="Save model every N epochs")
    parser.add_argument("--val_interval", type=int, default=5, help="Validate every N epochs")
    parser.add_argument("--sample_interval", type=int, default=30, help="Sample images every N batches")

    # 模型参数
    parser.add_argument("--weather_type", type=str, default="snow",
                        choices=["rain", "snow", "fog", "night"],
                        help="Type of weather to generate")

    # 损失权重（初始值，训练过程中会动态调整）
    parser.add_argument("--lambda_adv", type=float, default=1.0, help="Adversarial loss weight")
    parser.add_argument("--lambda_sem", type=float, default=2.0, help="Semantic consistency loss weight")
    parser.add_argument("--lambda_phys", type=float, default=0.1, help="Physics prior loss weight")
    parser.add_argument("--lambda_perc", type=float, default=0.3, help="Perceptual loss weight")

    # 输出设置
    parser.add_argument("--output_dir", type=str, default="output", help="Directory for output files")
    parser.add_argument("--experiment_name", type=str, default="spc", help="Experiment name for logging")

    # 设备设置
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")

    parser.add_argument("--loss_schedule", type=str, default=None,
                        help="JSON string defining loss schedule, e.g., "
                             "'{\"stages\": [...]}'")
    parser.add_argument("--auto_schedule", action="store_true",default=True,
                        help="Use built-in automatic loss schedule (recommended)")

    return parser.parse_args()


def main():
    args = parse_args()

    import os
    import torch.hub

    # 1. 设置 hub 缓存目录（确保 EfficientNet 也在此目录下）
    torch.hub.set_dir(os.path.expanduser(r'C:\Users\Administrator\.cache\torch\hub'))

    # 2. 禁止联网检查（PyTorch >=1.13 支持）
    os.environ['TORCH_HUB_NO_INTERNET'] = '1'

    # 3. 然后加载 MiDaS（使用本地路径）
    midas = torch.hub.load(
        r'C:\Users\Administrator\.cache\torch\hub\intel-isl_MiDaS_master',
        'MiDaS_small',
        source='local',
        trust_repo=True
    ).to(args.device)
    midas.eval()

    midas_transforms = torch.hub.load(
        r'C:\Users\Administrator\.cache\torch\hub\intel-isl_MiDaS_master',
        'transforms',
        source='local',
        trust_repo=True
    ).small_transform

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    # 设置TensorBoard日志
    log_dir = os.path.join(args.output_dir, "logs", args.experiment_name)
    writer = SummaryWriter(log_dir=log_dir)

    # 打印配置信息
    print("\n" + "=" * 70)
    print(f"Training SPC Framework for {args.weather_type.upper()} Generation")
    print("=" * 70)
    print(f"Experiment name: {args.experiment_name}")
    print(f"Device: {args.device}")
    print(f"Dataset directory: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.image_size}")
    print(f"Generator LR: {args.lr_g}, Discriminator LR: {args.lr_d}")
    print(f"Epochs: {args.epochs}")
    print(f"Weather type: {args.weather_type}")
    print(f"Loss weights - Adv: {args.lambda_adv}, Sem: {args.lambda_sem}, Phys: {args.lambda_phys}, Perc: {args.lambda_perc}")
    print("=" * 70 + "\n")

    # 创建数据加载器
    # 注意：create_dataloaders 应返回 (train_loader_clear, train_loader_adverse, val_loader_clear, val_loader_adverse)
    # 每个 loader 产出 (image, semantic_label)
    train_loader_clear, train_loader_adverse, val_loader_clear, val_loader_adverse = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )

    # 初始化模型
    model = SPCModel(weather_type=args.weather_type)
    model.to(args.device)

    # 优化器
    optimizer_G = optim.Adam(model.generator.parameters(), lr=args.lr_g, betas=(args.b1, args.b2))
    optimizer_D = optim.Adam(model.discriminator.parameters(), lr=args.lr_d, betas=(args.b1, args.b2))

    # 学习率调度器
    scheduler_G = lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=args.epochs, eta_min=1e-5)
    scheduler_D = lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=args.epochs, eta_min=1e-5)

    # 损失计算器
    loss_calculator = SPCLossCalculator(weather_type=args.weather_type)
    loss_calculator.to(args.device)

    # 解析损失调度表
    schedule = None
    if args.auto_schedule:
        # 内置默认调度表（可根据需要调
        # 整）
        schedule = {
            "stages": [
                {"end_epoch": 20, "adv": 0.5, "perc": 2.0, "sem": 1.0, "phys": 0.1},
                {"end_epoch": 80, "adv": 0.5, "perc": 2.0, "sem": 1.0, "phys": 0.1},
                {"end_epoch": 140, "adv": 0.5, "perc": 2.0, "sem": 1.0, "phys": 0.1},
                {"end_epoch": 200, "adv": 0.5, "perc": 2.0, "sem": 1.0, "phys": 0.1},
            ]
        }
    elif args.loss_schedule:
        import json
        schedule = json.loads(args.loss_schedule)

    # 初始化当前阶段
    current_stage_idx = 0
    if schedule:
        stages = schedule['stages']
        current_stage = stages[0]
        loss_calculator.set_stage(current_stage)
        print(f"初始阶段: {current_stage}")

    # 加载检查点（如果有）
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(
            model, optimizer_G, optimizer_D,
            args.resume, device=args.device
        )
        print(f"从epoch {start_epoch}恢复训练")

    # 训练统计
    history = {
        'G_loss': [], 'D_loss': [],
        'adv_loss': [], 'sem_loss': [], 'phys_loss': [], 'perc_loss': [],
        'val_sem_acc': [], 'val_sem_iou': []
    }

    global_step = 0
    best_val_iou = 0.0

    # 训练循环
    print("开始训练...")
    for epoch in range(start_epoch, args.epochs):
        if schedule:
            while current_stage_idx < len(stages) - 1 and epoch >= stages[current_stage_idx + 1]['end_epoch']:
                current_stage_idx += 1
                current_stage = stages[current_stage_idx]
                loss_calculator.set_stage(current_stage)
                print(f"\n>>> 切换到阶段 {current_stage_idx + 1}: {current_stage} <<<\n")


        epoch_start_time = time.time()
        model.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_batches = 0

        # 使用 zip 同时遍历晴天和天气数据
        pbar = tqdm(zip(train_loader_clear, train_loader_adverse),
                    total=min(len(train_loader_clear), len(train_loader_adverse)),
                    desc=f"Epoch {epoch + 1}/{args.epochs}")

        for (clear_img, clear_sem), (adv_img, adv_sem) in pbar:
            clear_img = clear_img.to(args.device)
            clear_sem = clear_sem.to(args.device)
            adv_img = adv_img.to(args.device)
            adv_sem = adv_sem.to(args.device)

            # ========== 训练生成器 ==========
            optimizer_G.zero_grad()

            # 前向传播生成天气图像
            fake_adv, gen_info = model.generator(clear_img, weather_type=args.weather_type,
                                                  weather_intensity=1.0, enable_aux=True)

            # 判别器对生成图像的预测
            disc_fake = model.discriminator(fake_adv)

            # ========== 计算伪深度图（仅雾天需要）==========
            if args.weather_type == 'fog':
                with torch.no_grad():
                    # 将 clear_img 从 [-1,1] 转换到 [0,1]（MiDaS 期望输入范围）
                    input_rgb = (clear_img + 1) / 2
                    # MiDaS 推理
                    depth_raw = midas(input_rgb)  # 形状可能是 (B, H, W) 或 (B, 1, H, W)

                    # 统一转换为四维 (B, 1, H, W)
                    if depth_raw.dim() == 3:
                        depth_raw = depth_raw.unsqueeze(1)  # (B, 1, H, W)
                    elif depth_raw.dim() == 4 and depth_raw.size(1) != 1:
                        # 如果通道数不为1，取平均或第一个通道
                        depth_raw = depth_raw[:, 0:1, :, :]

                    # 缩放到生成器输出尺寸（256x256）
                    pseudo_depth = F.interpolate(depth_raw, size=(256, 256), mode='bilinear', align_corners=False)
            else:
                pseudo_depth = None

            # 计算生成器总损失
            loss_G, loss_dict_G = loss_calculator(
                (fake_adv, gen_info),
                disc_fake,
                clear_img,  # 输入晴天图像（用于感知损失）
                adv_img,  # 真实天气图像（当前未使用）
                clear_sem,  # 语义标签
                epoch,
                args.epochs,
                pseudo_depth=pseudo_depth  # 新增参数
            )

            loss_G.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.generator.parameters(), max_norm=1.0)
            optimizer_G.step()

            # ========== 训练判别器 ==========
            optimizer_D.zero_grad()

            # 真实图像判别
            disc_real = model.discriminator(adv_img)
            # 生成图像判别（detach）
            disc_fake_detach = model.discriminator(fake_adv.detach())

            # 判别器损失（LSGAN）
            loss_D_real = sum([nn.MSELoss()(pred, torch.ones_like(pred)) for pred in disc_real])
            loss_D_fake = sum([nn.MSELoss()(pred, torch.zeros_like(pred)) for pred in disc_fake_detach])
            loss_D = (loss_D_real + loss_D_fake) * 0.5

            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), max_norm=1.0)
            if global_step % 1 == 0:
                optimizer_D.step()

            # ========== 记录 ==========
            epoch_g_loss += loss_G.item()
            epoch_d_loss += loss_D.item()
            epoch_batches += 1

            # 在每次生成器更新后添加
            # print(f"Grad norm: {torch.nn.utils.clip_grad_norm_(model.generator.parameters(), float('inf'))}")
            # print(
            #     f"Fake image stats: mean={fake_adv.mean().item():.3f}, std={fake_adv.std().item():.3f}, min={fake_adv.min().item():.3f}, max={fake_adv.max().item():.3f}")
            # print(f"Disc output on fake: {[p.mean().item() for p in disc_fake]}")

            # TensorBoard记录
            writer.add_scalar('Loss/Generator', loss_G.item(), global_step)
            writer.add_scalar('Loss/Discriminator', loss_D.item(), global_step)
            writer.add_scalar('Loss/Adv', loss_dict_G['loss_adv'], global_step)
            writer.add_scalar('Loss/Sem', loss_dict_G['loss_sem'], global_step)
            writer.add_scalar('Loss/Phys', loss_dict_G['loss_phys'], global_step)
            writer.add_scalar('Loss/Perc', loss_dict_G['loss_perc'], global_step)

            # 定期保存样本
            if global_step % args.sample_interval == 0:
                with torch.no_grad():
                    visualize_batch(
                        clear_img, fake_adv, adv_img,
                        clear_sem, adv_sem,
                        save_path=os.path.join(args.output_dir, "samples", f"step_{global_step}"),
                        title=f"Step {global_step}",
                        step=global_step
                    )

            global_step += 1
            pbar.set_postfix({
                'G': f"{loss_G.item():.3f}",
                'D': f"{loss_D.item():.3f}"
            })

        # 计算epoch平均损失
        avg_g_loss = epoch_g_loss / epoch_batches if epoch_batches > 0 else 0
        avg_d_loss = epoch_d_loss / epoch_batches if epoch_batches > 0 else 0

        # TensorBoard记录epoch损失
        writer.add_scalar('Epoch/Loss_Generator', avg_g_loss, epoch)
        writer.add_scalar('Epoch/Loss_Discriminator', avg_d_loss, epoch)

        # 学习率调度
        scheduler_G.step()
        scheduler_D.step()

        # 定期验证
        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            model.eval()
            val_dir = os.path.join(args.output_dir, "val_samples", f"epoch_{epoch + 1}")
            os.makedirs(val_dir, exist_ok=True)

            # 计算语义指标（使用验证集）
            with torch.no_grad():
                sem_metrics = compute_semantic_metrics(model, val_loader_clear, args.device)
                sem_acc = sem_metrics['accuracy']
                sem_iou = sem_metrics['iou']

            history['val_sem_acc'].append(sem_acc)
            history['val_sem_iou'].append(sem_iou)

            writer.add_scalar('Validation/Semantic_Accuracy', sem_acc, epoch)
            writer.add_scalar('Validation/Semantic_IoU', sem_iou, epoch)

            print(f"验证语义准确率: {sem_acc:.4f}")
            print(f"验证语义IoU: {sem_iou:.4f}")

            # 保存最佳模型
            if sem_iou > best_val_iou:
                best_val_iou = sem_iou
                save_checkpoint(
                    model, optimizer_G, optimizer_D, epoch + 1,
                    os.path.join(args.output_dir, "checkpoints"),
                    filename_prefix="best"
                )
                print(f"保存最佳模型，IoU: {best_val_iou:.4f}")

            model.train()

        # 定期保存检查点
        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            save_checkpoint(
                model, optimizer_G, optimizer_D, epoch + 1,
                os.path.join(args.output_dir, "checkpoints"),
                filename_prefix=args.experiment_name
            )

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1}完成 - 时间: {epoch_time:.2f}s")
        print(f"  生成器损失: {avg_g_loss:.4f}, 判别器损失: {avg_d_loss:.4f}")

    # 训练结束
    print("\n训练完成!")

    # 保存最终模型
    save_checkpoint(
        model, optimizer_G, optimizer_D, args.epochs,
        os.path.join(args.output_dir, "checkpoints"),
        filename_prefix="final"
    )

    # 生成示例图像
    sample_input = os.path.join(args.data_dir, "sample_input.jpg")
    if os.path.exists(sample_input):
        output_path = os.path.join(args.output_dir, "generated_adverse.jpg")
        generate_adverse_weather(model, sample_input, output_path, args.device)
        print(f"生成的恶劣天气图像已保存至 {output_path}")

    writer.close()


if __name__ == '__main__':
    main()