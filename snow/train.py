import argparse
import os
import time
import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import SemanticCycleGAN
from utils import (
    create_dataloaders,
    visualize_batch,
    save_checkpoint,
    load_checkpoint,
    compute_semantic_metrics,
    generate_adverse_weather,
    visualize_physics, save_individual_components, visualize_physics_separate, generate_paper_figures,
    create_workflow_diagram
)

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description="Train Semantic-Constrained CycleGAN for Autonomous Driving")

    # 数据参数
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to dataset directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--image_size", type=int, default=256, help="Input image size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="Adam: beta1")
    parser.add_argument("--b2", type=float, default=0.999, help="Adam: beta2")
    parser.add_argument("--resume", type=str, default="./output/checkpoints/final_epoch_200.pth", help="Path to checkpoint to resume from")
    parser.add_argument("--save_interval", type=int, default=20, help="Save model every N epochs")
    parser.add_argument("--val_interval", type=int, default=20, help="Validate every N epochs")
    parser.add_argument("--sample_interval", type=int, default=50, help="Sample images every N batches")
    parser.add_argument("--physics_sample_interval", type=int, default=1,
                        help="Sample physics visualization every N batches")

    # 模型参数
    parser.add_argument("--semantic_dim", type=int, default=128, help="Dimension of semantic features")

    # 损失权重
    parser.add_argument("--lambda_cycle", type=float, default=10.0, help="Cycle consistency loss weight")
    parser.add_argument("--lambda_semantic", type=float, default=5.0, help="Semantic consistency loss weight")
    parser.add_argument("--lambda_physics", type=float, default=0.5, help="Physics constraint loss weight")

    # 输出设置
    parser.add_argument("--output_dir", type=str, default="output", help="Directory for output files")
    parser.add_argument("--experiment_name", type=str, default="sem_cyclegan", help="Experiment name for logging")

    # 设备设置
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--weather_type", type=str, default="snow",
                        choices=["rain", "snow", "fog", "night"],
                        help="Type of weather to generate")
    return parser.parse_args()


def freeze(module):
    """冻结模块参数"""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze(module):
    """解冻模块参数"""
    for param in module.parameters():
        param.requires_grad = True


def unfreeze_all(model):
    """解冻所有参数"""
    for param in model.parameters():
        param.requires_grad = True


def pretrain_semantic(model, dataloader_A, dataloader_B, optimizer, device, epochs=50,
                      save_path="./checkpoints/semantic"):
    """增强型语义预训练-支持自动保存检查点"""
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    best_loss = float('inf')

    # 冻结生成器相关参数
    freeze(model.G_A2B)
    freeze(model.G_B2A)

    # 创建辅助解码器
    model.auxiliary_decoders = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 6, 1)
        ).to(device),
        nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 6, 1)
        ).to(device)
    ])

    # 优化器添加辅助解码器
    optimizer.add_param_group({'params': model.auxiliary_decoders.parameters()})

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler()

    # 训练开始前打印参数信息
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"开始语义预训练，可训练参数: {total_params:,}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        total_samples = 0

        # 合并数据加载器
        dataloaders = [dataloader_A, dataloader_B]
        batches = []
        for dataloader in dataloaders:
            batches.extend([(images, sem_targets) for images, sem_targets in dataloader])

        pbar = tqdm(batches, desc=f"Epoch {epoch + 1}/{epochs}", dynamic_ncols=True)

        for images, sem_targets in pbar:
            images = images.to(device)
            sem_targets = sem_targets.to(device)

            with torch.cuda.amp.autocast():
                # 获取语义特征
                sem_features, _ = model.semantic_encoder(images)

                # 语义重建
                sem_pred = model.semantic_decoder(sem_features)

                # 计算损失
                loss, _ = model.compute_semantic_loss(
                    sem_features,
                    sem_pred,
                    sem_targets,
                    epoch
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            current_loss = loss.item()
            batch_size = images.size(0)
            epoch_loss += current_loss * batch_size
            total_samples += batch_size

            avg_epoch_loss = epoch_loss / total_samples if total_samples > 0 else 0
            pbar.set_postfix({
                'batch_loss': f"{current_loss:.4f}",
                'epoch_loss': f"{avg_epoch_loss:.4f}"
            })

        # 学习率衰减
        if (epoch + 1) % 10 == 0:
            new_lr = optimizer.param_groups[0]['lr'] * 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            pbar.write(f"\nEpoch {epoch + 1}完成 - 平均损失: {avg_epoch_loss:.4f}, 学习率衰减至 {new_lr:.1e}")

        # ========= 每5个epoch保存检查点 =========
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            is_best = True
        else:
            is_best = False

        if (epoch + 1) % 5 == 0 or is_best or (epoch + 1) == epochs:
            # 保存完整的训练状态
            checkpoint = {
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scaler_state': scaler.state_dict(),
                'loss': avg_epoch_loss,
                'best_loss': best_loss
            }

            # 常规检查点
            torch.save(checkpoint, f"{save_path}/semantic_epoch{epoch + 1}.pth")

            # 最佳模型单独保存
            if is_best:
                torch.save(checkpoint, f"{save_path}/best_semantic_model.pth")
                pbar.write(f"★ 发现最佳模型! 保存为 {save_path}/best_semantic_model.pth")

    # 训练结束后保存最终模型
    final_path = f"{save_path}/semantic_final_model.pth"
    torch.save({
        'epoch': epochs,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss': avg_epoch_loss
    }, final_path)

    print(f"\n√ 语义预训练完成! 最终损失: {avg_epoch_loss:.4f}")
    print(f"最终模型已保存至: {final_path}")
    print(f"最佳模型已保存至: {save_path}/best_semantic_model.pth")

    # 移除辅助解码器
    del model.auxiliary_decoders

    # 解冻生成器参数
    unfreeze(model.G_A2B)
    unfreeze(model.G_B2A)

    # 冻结编码器参数
    freeze(model.semantic_encoder)
    print("冻结语义编码器参数")





def main():
    # 解析参数
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "physics_samples"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    # 设置TensorBoard日志
    log_dir = os.path.join(args.output_dir, "logs", args.experiment_name)
    writer = SummaryWriter(log_dir=log_dir)

    # 打印配置信息
    print("\n" + "=" * 60)
    print(f"Training Semantic-Constrained CycleGAN")
    print("=" * 60)
    print(f"Experiment name: {args.experiment_name}")
    print(f"Device: {args.device}")
    print(f"Dataset directory: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.image_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Semantic dimension: {args.semantic_dim}")
    print(f"Cycle loss weight: {args.lambda_cycle}")
    print(f"Semantic loss weight: {args.lambda_semantic}")
    print(f"Physics loss weight: {args.lambda_physics}")
    print("=" * 60 + "\n")

    # 创建数据加载器
    train_loader_A, train_loader_B, val_loader_A, val_loader_B = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )

    # 初始化模型
    model = SemanticCycleGAN(
        device=args.device,
        semantic_dim=args.semantic_dim,
        weather_type=args.weather_type
    )
    model.to(args.device)

    # 优化器设置
    # 为语义编解码器单独创建优化器
    optimizer_sem = optim.Adam(
        list(model.semantic_encoder.parameters()) +
        list(model.semantic_decoder.parameters()),
        lr=0.001  # 更高的学习率
    )

    # 生成器和判别器优化器
    optimizer_G = optim.Adam(
        list(model.G_A2B.parameters()) + list(model.G_B2A.parameters()),
        lr=args.lr,
        betas=(args.b1, args.b2)
    )

    optimizer_D = optim.Adam(
        list(model.D_A.parameters()) + list(model.D_B.parameters()),
        lr=args.lr * 0.5,
        betas=(args.b1, args.b2)
    )

    # 学习率调度器
    scheduler_G = lr_scheduler.ReduceLROnPlateau(
        optimizer_G,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    scheduler_D = lr_scheduler.StepLR(optimizer_D, step_size=50, gamma=0.5)

    save_path = "./checkpoints/semantic"
    best_model_path = os.path.join(save_path, "best_semantic_model.pth")

    # 尝试加载最佳语义模型
    if os.path.exists(best_model_path):
        try:
            checkpoint = torch.load(best_model_path, map_location=args.device)
            model.load_state_dict(checkpoint['model_state'], strict=False)
            print(f"√ 成功加载预训练语义模型 (Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f})")
        except:
            print("X 模型加载失败，开始训练新模型...")
            pretrain_semantic(model, train_loader_A, train_loader_B, optimizer_sem,
                              args.device, epochs=30, save_path=save_path)
    else:
        print("未检测到最佳模型，开始训练新模型...")
        pretrain_semantic(model, train_loader_A, train_loader_B, optimizer_sem,
                          args.device, epochs=20, save_path=save_path)

    # 解冻所有参数（预训练后）
    unfreeze_all(model)

    # 加载检查点
    start_epoch = 0
    if args.resume:
        checkpoint_path = args.resume
        start_epoch = load_checkpoint(
            model, optimizer_G, optimizer_D,
            checkpoint_path, device=args.device
        )
        print(f"从epoch {start_epoch}恢复训练")

    # 训练统计信息
    history = {
        'G_loss': [],
        'D_loss': [],
        'cycle_loss': [],
        'sem_loss': [],
        'physics_loss': [],
        'val_sem_acc': [],
        'val_sem_iou': []
    }
    scaler = torch.cuda.amp.GradScaler() if args.device == 'cuda' else None

    # 训练循环
    global_step = 0
    best_val_iou = 0.0
    print("开始训练...")

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_cycle_loss = 0.0
        epoch_sem_loss = 0.0
        epoch_physics_loss = 0.0
        epoch_batches = 0
        if epoch < 50:
            d_train = 3
        elif epoch < 100:
            d_train = 2
        elif epoch < 150:
            d_train = 1
        else:
            d_train = 1
        if epoch % 20 == 0:
            for param_group in optimizer_G.param_groups:
                param_group['lr'] *= 0.9

        # 训练一个epoch
        model.train()
        pbar = tqdm(zip(train_loader_A, train_loader_B),
                    total=min(len(train_loader_A), len(train_loader_B)),
                    desc=f"Epoch {epoch + 1}/{args.epochs} ]")
        batch_idx = 0
        for (real_A, sem_A), (real_B, sem_B) in pbar:
            real_A = real_A.to(args.device)
            sem_A = sem_A.to(args.device)
            real_B = real_B.to(args.device)
            sem_B = sem_B.to(args.device)

            # ========== 训练生成器 ==========
            optimizer_G.zero_grad()

            # 混合精度前向传播
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model(real_A, real_B)
                loss_G, loss_dict = model.compute_generator_loss(
                    real_A, real_B, sem_A, sem_B, outputs, epoch, args.epochs
                )

            # 混合精度反向传播
            if scaler:
                scaler.scale(loss_G).backward()
                # 梯度裁剪（在scaler作用后进行）
                scaler.unscale_(optimizer_G)
                torch.nn.utils.clip_grad_norm_(model.G_A2B.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(model.G_B2A.parameters(), max_norm=1.0)
                scaler.step(optimizer_G)
            else:
                loss_G.backward()
                torch.nn.utils.clip_grad_norm_(model.G_A2B.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(model.G_B2A.parameters(), max_norm=1.0)
                optimizer_G.step()

            # ========== 训练判别器（基于d_train频率）==========
            if batch_idx % d_train == 0:
                optimizer_D.zero_grad()

                # 混合精度计算判别器损失
                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    loss_D = model.compute_discriminator_loss(outputs)

                # 混合精度反向传播
                if scaler:
                    scaler.scale(loss_D).backward()
                    # 梯度裁剪（在scaler作用后进行）
                    scaler.unscale_(optimizer_D)
                    torch.nn.utils.clip_grad_norm_(model.D_A.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(model.D_B.parameters(), max_norm=1.0)
                    scaler.step(optimizer_D)
                else:
                    loss_D.backward()
                    torch.nn.utils.clip_grad_norm_(model.D_A.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(model.D_B.parameters(), max_norm=1.0)
                    optimizer_D.step()
            else:
                loss_D = torch.zeros(1)

            # 更新scaler - 每次迭代后都需要更新
            if scaler:
                scaler.update()

            batch_idx += 1
            # ========== 监控和记录 ==========
            # 记录损失
            history['G_loss'].append(loss_G.item())
            history['D_loss'].append(loss_D.item())
            history['cycle_loss'].append(loss_dict['cycle'])
            history['sem_loss'].append(loss_dict['sem'])
            history['physics_loss'].append(loss_dict['physics'])

            # 更新epoch统计
            epoch_g_loss += loss_G.item()
            epoch_d_loss += loss_D.item()
            epoch_cycle_loss += loss_dict['cycle']
            epoch_sem_loss += loss_dict['sem']
            epoch_physics_loss += loss_dict['physics']
            epoch_batches += 1

            # TensorBoard记录
            writer.add_scalar('Loss/Generator', loss_G.item(), global_step)
            writer.add_scalar('Loss/Discriminator', loss_D.item(), global_step)
            writer.add_scalar('Loss/Cycle', loss_dict['cycle'], global_step)
            writer.add_scalar('Loss/Semantic', loss_dict['sem'], global_step)
            writer.add_scalar('Loss/Physics', loss_dict['physics'], global_step)

            # 定期保存样本
            if global_step % args.sample_interval == 0:
                with torch.no_grad():
                    # 可视化
                    save_filename = f"batch_{global_step}"
                    save_path = os.path.join(args.output_dir, "samples", save_filename)

                    # 调用可视化函数
                    visualize_batch(
                        real_A, outputs['fake_B'], outputs['rec_A'],
                        save_path=save_path,
                        step=global_step
                    )

                    # 使用正确的文件名读取
                    image_path = f"{save_path}_batch_{global_step}.png"
                    if os.path.exists(image_path):
                        writer.add_image('Samples/Train',
                                         plt.imread(image_path).transpose(2, 0, 1),
                                         global_step)

            # 在训练循环中，定期生成论文图表
            # if global_step % args.sample_interval == 0:
            #     with torch.no_grad():
                    # 生成论文机制图
                    paper_dir = os.path.join(args.output_dir, "paper_figures", f"step_{global_step}")
                    # generate_paper_figures(
                    #     model, real_A, sem_A, outputs['fake_B_dict'],
                    #     paper_dir, step=global_step
                    # )

                    # 生成工作流程图（只需要生成一次）
                    # if global_step == 0:
                    #     workflow_path = os.path.join(args.output_dir, "workflow_diagram.png")
                    #     create_workflow_diagram(workflow_path)

            global_step += 1
            pbar.set_postfix({
                'G_loss': f"{loss_G.item():.4f}",
                'D_loss': f"{loss_D.item():.4f}",
                'Physics': f"{loss_dict['physics']:.4f}"
            })

        # 计算epoch平均损失
        avg_g_loss = epoch_g_loss / epoch_batches if epoch_batches > 0 else 0
        avg_d_loss = epoch_d_loss / epoch_batches if epoch_batches > 0 else 0
        avg_cycle_loss = epoch_cycle_loss / epoch_batches if epoch_batches > 0 else 0
        avg_sem_loss = epoch_sem_loss / epoch_batches if epoch_batches > 0 else 0
        avg_physics_loss = epoch_physics_loss / epoch_batches if epoch_batches > 0 else 0

        # TensorBoard记录
        writer.add_scalar('Epoch/Loss_Generator', avg_g_loss, epoch)
        writer.add_scalar('Epoch/Loss_Discriminator', avg_d_loss, epoch)
        writer.add_scalar('Epoch/Loss_Cycle', avg_cycle_loss, epoch)
        writer.add_scalar('Epoch/Loss_Semantic', avg_sem_loss, epoch)
        writer.add_scalar('Epoch/Loss_Physics', avg_physics_loss, epoch)

        # 学习率调整
        scheduler_G.step(avg_g_loss)
        scheduler_D.step()

        # 定期验证
        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            model.eval()
            val_dir = os.path.join(args.output_dir, "val_samples", f"epoch_{epoch + 1}")
            os.makedirs(val_dir, exist_ok=True)

            # 计算语义指标
            with torch.no_grad():
                sem_metrics = compute_semantic_metrics(model, val_loader_A, args.device)
                sem_acc = sem_metrics['accuracy']
                sem_iou = sem_metrics['iou']

            # 记录历史
            history['val_sem_acc'].append(sem_acc)
            history['val_sem_iou'].append(sem_iou)

            # TensorBoard记录
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

            # 切换回训练模式
            model.train()

        # 定期保存检查点
        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            save_checkpoint(model, optimizer_G, optimizer_D, epoch + 1,
                            os.path.join(args.output_dir, "checkpoints"),
                            filename_prefix=args.experiment_name)

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1}完成 - 时间: {epoch_time:.2f}s")
        print(f"  生成器损失: {avg_g_loss:.4f}, 判别器损失: {avg_d_loss:.4f}")
        print(f"  循环损失: {avg_cycle_loss:.4f}, 语义损失: {avg_sem_loss:.4f}, 物理损失: {avg_physics_loss:.4f}")

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

    # 关闭TensorBoard写入器
    writer.close()



if __name__ == '__main__':
    main()