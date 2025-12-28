import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import torchvision.models as models
import time

plt.switch_backend('Agg')  # 避免GUI问题
# 在 train.py 开头添加
import matplotlib as mpl
from matplotlib import font_manager

try:
    # 尝试使用系统支持的字体
    font_path = font_manager.findfont('SimHei')  # 查找黑体
    if font_path:
        mpl.rcParams['font.family'] = 'SimHei'
    else:
        # 回退到支持中文的字体
        supported_fonts = ['DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
        for font_name in supported_fonts:
            if font_manager.findfont(font_name):
                mpl.rcParams['font.family'] = font_name
                break
except:
    # 如果所有尝试都失败，使用默认字体
    mpl.rcParams['font.family'] = 'sans-serif'

# 全局配置映射表
SEMANTIC_MAPPING = {
    'background': [0],
    'road': [7, 8, 9, 10],
    'sign': [20],
    'vegetation': [21, 22],
    'sky': [23],
    'dynamic': list(range(24, 34))
}

def map_acdc_to_custom(semantic):
    mapped = np.zeros_like(semantic, dtype=np.uint8)
    for cls_idx, class_name in enumerate(SEMANTIC_MAPPING.keys()):
        labels = SEMANTIC_MAPPING[class_name]
        for label in labels:
            mapped[semantic == label] = cls_idx
    return mapped


def visualize_physics(model, real_A, fake_B, outputs_dict, save_path):
    """可视化物理约束效果 - 通用版本"""
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    # 反归一化函数
    def safe_denorm(tensor):
        denormed = tensor.clone()
        for c in range(3):
            denormed[c] = denormed[c] * 0.5 + 0.5
        return denormed.permute(1, 2, 0).detach().cpu().numpy()

    # 原始图像
    img_real = safe_denorm(real_A[0])
    axs[0, 0].imshow(img_real)
    axs[0, 0].set_title("原始图像")
    axs[0, 0].axis('off')

    # 基础场景
    base_scene = outputs_dict['base_scene'][0]
    base_img = safe_denorm(base_scene)
    axs[0, 1].imshow(base_img)
    axs[0, 1].set_title("基础场景")
    axs[0, 1].axis('off')

    # 天气特效
    weather_effect = outputs_dict['weather_effect'][0].cpu()
    weather_img = (weather_effect - weather_effect.min()) / (weather_effect.max() - weather_effect.min() + 1e-6)
    weather_img = weather_img.permute(1, 2, 0).numpy()
    axs[0, 2].imshow(weather_img)
    axs[0, 2].set_title("天气特效")
    axs[0, 2].axis('off')

    # 区域掩码
    region_mask = outputs_dict['region_mask'][0].cpu().detach()
    region_img = region_mask.permute(1, 2, 0).numpy()
    axs[1, 0].imshow(region_img)
    axs[1, 0].set_title("区域掩码(天空/动态/地面)")
    axs[1, 0].axis('off')

    # 物理约束可视化
    # 1. 动态区域天气效果
    dynamic_mask = region_mask[0, 1].cpu().detach().numpy()  # [H, W]
    effect_in_dynamic = weather_effect.mean(dim=0).cpu().numpy() * dynamic_mask
    axs[1, 1].imshow(effect_in_dynamic, cmap='viridis')
    axs[1, 1].set_title("动态区域天气效果")
    axs[1, 1].axis('off')

    # 2. 天气效果梯度图
    # 修复：确保卷积核是浮点张量
    sobel_x = torch.tensor([[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]],
                           dtype=torch.float32, device=weather_effect.device)
    sobel_y = torch.tensor([[[[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]]],
                           dtype=torch.float32, device=weather_effect.device)

    # 计算梯度
    weather_mean = weather_effect.mean(dim=0, keepdim=True).to(device=weather_effect.device)
    grad_x = F.conv2d(weather_mean, sobel_x, padding=1)
    grad_y = F.conv2d(weather_mean, sobel_y, padding=1)
    gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6).squeeze().cpu().numpy()

    axs[1, 2].imshow(gradient_magnitude, cmap='hot')
    axs[1, 2].set_title("天气效果梯度强度")
    axs[1, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}_physics.png", dpi=120, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# 新增深度图可视化函数
def visualize_depth(depth_map, save_path=None):
    """将深度图转换为可视化的彩色图像"""
    # 归一化深度图
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)

    # 应用颜色映射
    depth_colored = plt.get_cmap('viridis')(depth_map)[:, :, :3]

    if save_path:
        plt.imsave(save_path, depth_colored)

    return depth_colored


# 新增风场向量可视化函数
def visualize_wind_field(wind_vector, save_path=None):
    """可视化风场向量"""
    fig, ax = plt.subplots(figsize=(6, 6))

    # 绘制风场箭头
    ax.quiver(0, 0, wind_vector[0], wind_vector[1],
              angles='xy', scale_units='xy', scale=1,
              color='r', width=0.01)

    # 设置坐标轴范围
    max_val = max(abs(wind_vector[0]), abs(wind_vector[1]), 0.1)
    ax.set_xlim(-max_val * 1.2, max_val * 1.2)
    ax.set_ylim(-max_val * 1.2, max_val * 1.2)

    # 添加网格和标签
    ax.grid(True)
    ax.set_xlabel('X方向')
    ax.set_ylabel('Y方向')
    ax.set_title(f'风场向量: [{wind_vector[0]:.2f}, {wind_vector[1]:.2f}]')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# 新增物理损失监控函数
def plot_physics_loss(loss_history, save_path=None):
    """可视化物理约束损失"""
    plt.figure(figsize=(10, 6))

    # 绘制物理损失
    plt.plot(loss_history['physics_loss'], label='物理损失', color='purple')

    # 绘制其他相关损失
    if 'G_loss' in loss_history:
        plt.plot(loss_history['G_loss'], label='生成器损失', alpha=0.5)
    if 'sem_loss' in loss_history:
        plt.plot(loss_history['sem_loss'], label='语义损失', alpha=0.5)

    plt.title('物理约束损失监控')
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# 新增解耦生成结果可视化
def visualize_disentangled(outputs_dict, save_path=None):
    """可视化解耦生成结果"""
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # 基础场景
    base_scene = outputs_dict['base_scene'][0].cpu()
    axs[0].imshow(denormalize(base_scene))
    axs[0].set_title("基础场景")

    # 天气特效
    weather_effect = outputs_dict['weather_effect'][0].cpu().permute(1, 2, 0)
    weather_effect = (weather_effect - weather_effect.min()) / (weather_effect.max() - weather_effect.min())
    axs[1].imshow(weather_effect.detach().numpy())
    axs[1].set_title("天气特效")

    # 最终结果
    final_output = outputs_dict['output'][0].cpu()
    axs[2].imshow(denormalize(final_output))
    axs[2].set_title("最终输出")

    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}_disentangled.png", dpi=120, bbox_inches='tight')
        plt.close()


# 新增辅助函数：反归一化图像
def denormalize(tensor):
    """将[-1,1]范围的张量转换为[0,1]范围的图像"""
    return (tensor.clamp(-1, 1) + 1) / 2


# 新增区域掩码可视化
def visualize_region_mask(region_mask, save_path=None):
    """可视化区域掩码"""
    # region_mask: [B, 3, H, W]
    mask = region_mask[0].cpu().detach()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['天空区域', '动态区域', '地面区域']

    for i in range(3):
        axs[i].imshow(mask[i], cmap='viridis')
        axs[i].set_title(titles[i])
        axs[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()


# 新增天气参数可视化
def visualize_weather_params(rain_intensity, fog_intensity, save_path=None):
    """可视化天气参数分布"""
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # 雨强度直方图
    axs[0].hist(rain_intensity.cpu().detach().numpy(), bins=20, color='blue', alpha=0.7)
    axs[0].set_title('雨强度分布')
    axs[0].set_xlabel('强度值')
    axs[0].set_ylabel('频率')

    # 雾强度直方图
    axs[1].hist(fog_intensity.cpu().detach().numpy(), bins=20, color='gray', alpha=0.7)
    axs[1].set_title('雾强度分布')
    axs[1].set_xlabel('强度值')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()


class SemanticDrivingDataset(Dataset):
    def __init__(self, root_dir, domain='A', transform=None, phase='train',
                 semantic_classes=5, image_size=256):
        super().__init__()
        # 分别设置图像和语义图的目录
        self.image_dir = os.path.join(root_dir, phase, domain, 'images')
        self.semantic_dir = os.path.join(root_dir, phase, domain, 'semantics')

        self.domain = domain
        self.transform = transform
        self.semantic_classes = semantic_classes
        self.image_size = image_size
        self.crop_size = 1080

        # 确保目录存在
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"图像目录不存在: {self.image_dir}")
        if not os.path.exists(self.semantic_dir):
            raise FileNotFoundError(f"语义图目录不存在: {self.semantic_dir}")

        # 收集图像文件并匹配对应的语义图
        self.image_files = []
        self.semantic_files = []

        # 支持的图像格式
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

        # 遍历图像目录
        for filename in sorted(os.listdir(self.image_dir)):
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in image_extensions:
                # 获取不带扩展名的文件名
                base_name = os.path.splitext(filename)[0]

                # 在语义图目录中查找匹配的文件
                semantic_candidates = []
                for sem_file in os.listdir(self.semantic_dir):
                    sem_base = os.path.splitext(sem_file)[0]
                    # 匹配规则：文件名相同（忽略扩展名）
                    if sem_base == base_name:
                        semantic_candidates.append(sem_file)

                if len(semantic_candidates) == 1:
                    self.image_files.append(filename)
                    self.semantic_files.append(semantic_candidates[0])
                elif len(semantic_candidates) > 1:
                    print(f"警告: 找到多个匹配的语义图文件 {semantic_candidates}, 使用第一个")
                    self.image_files.append(filename)
                    self.semantic_files.append(semantic_candidates[0])
                else:
                    # 尝试常见的语义图命名模式
                    semantic_patterns = [
                        f"{base_name}_semantic.png",
                        f"{base_name}_label.png",
                        f"{base_name}_mask.png",
                        f"{base_name}_gt.png"
                    ]

                    for pattern in semantic_patterns:
                        pattern_path = os.path.join(self.semantic_dir, pattern)
                        if os.path.exists(pattern_path):
                            self.image_files.append(filename)
                            self.semantic_files.append(pattern)
                            break
                    else:
                        print(f"警告: 未找到图像 {filename} 对应的语义图，跳过该样本")

        if len(self.image_files) == 0:
            raise FileNotFoundError(f"在 {self.image_dir} 中未找到有效的图像-语义图对")

        print(f"加载了 {len(self.image_files)} 个 {domain} 域的图像-语义图对，用于 {phase} 阶段")

        # 空间变换
        self.spatial_transform = transforms.Compose([
            transforms.CenterCrop(self.crop_size),
            transforms.Resize((image_size, image_size)),
        ])

        # 标记是否已保存示例
        self.sample_saved = False

    def __len__(self):
        """返回数据集中的样本数量"""
        return len(self.image_files)

    def __getitem__(self, idx):
        # 加载图像
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        # 加载语义图
        semantic_path = os.path.join(self.semantic_dir, self.semantic_files[idx])
        if not os.path.exists(semantic_path):
            raise FileNotFoundError(f"语义图文件不存在: {semantic_path}")

        semantic = Image.open(semantic_path).convert('L')

        # 应用空间变换
        image = self.spatial_transform(image)
        semantic = self.spatial_transform(semantic)

        # 转换为NumPy数组并映射
        semantic = np.array(semantic)
        semantic = map_acdc_to_custom(semantic)

        # 验证映射后的值范围
        if np.any(semantic > 5) or np.any(semantic < 0):
            print(f"警告: 映射后仍有无效值 [{semantic.min()}, {semantic.max()}], 正在裁剪")
            semantic = np.clip(semantic, 0, 5)

        # 在第一次读取时保存示例（可选）
        if not self.sample_saved and idx == 0:
            sample_dir = os.path.join(self.image_dir, "..", "samples")
            os.makedirs(sample_dir, exist_ok=True)

            # 保存原始图像和语义图
            image.save(os.path.join(sample_dir, f"{self.domain}_original.png"))
            Image.fromarray(semantic.astype(np.uint8)).save(
                os.path.join(sample_dir, f"{self.domain}_semantic_raw.png"))

            self.sample_saved = True

        # 应用图像变换
        if self.transform:
            image = self.transform(image)

        return image, torch.from_numpy(semantic).long()


def create_dataloaders(data_dir, batch_size=4, image_size=256, num_workers=4):
    """创建训练和验证数据加载器"""
    # 获取变换
    train_img_transform, _ = get_transforms(image_size, 'train')
    val_img_transform, _ = get_transforms(image_size, 'val')

    # 创建数据集
    train_set_A = SemanticDrivingDataset(
        data_dir, domain='A', transform=train_img_transform,
        phase='train', image_size=image_size
    )
    train_set_B = SemanticDrivingDataset(
        data_dir, domain='B', transform=train_img_transform,
        phase='train', image_size=image_size
    )
    val_set_A = SemanticDrivingDataset(
        data_dir, domain='A', transform=val_img_transform,
        phase='val', image_size=image_size
    )
    val_set_B = SemanticDrivingDataset(
        data_dir, domain='B', transform=val_img_transform,
        phase='val', image_size=image_size
    )

    # 创建数据加载器
    train_loader_A = DataLoader(
        train_set_A, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    train_loader_B = DataLoader(
        train_set_B, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader_A = DataLoader(val_set_A, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    val_loader_B = DataLoader(val_set_B, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    return train_loader_A, train_loader_B, val_loader_A, val_loader_B

def semantic_to_onehot(semantic, num_classes=6):
    """将语义图转换为one-hot编码"""
    if not isinstance(semantic, np.ndarray):
        semantic = np.array(semantic)

    # 确保值在[0, num_classes-1]范围内
    if np.any(semantic >= num_classes) or np.any(semantic < 0):
        print(
            f"Warning: Semantic map contains invalid values in range [{semantic.min()}, {semantic.max()}]. Clamping to [0, {num_classes - 1}]")
        semantic = np.clip(semantic, 0, num_classes - 1)

    semantic = semantic.astype(np.int64)
    one_hot = F.one_hot(torch.from_numpy(semantic), num_classes=num_classes)
    one_hot = one_hot.permute(2, 0, 1).float()
    return one_hot[:num_classes]


def get_transforms(image_size=512, phase='train'):
    """获取无数据增强的变换"""
    # 仅包含必要转换
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 语义图无需额外变换
    return image_transform, None





def visualize_batch(real_A, fake_B, rec_A, semantic_A, semantic_hat,
                    save_path=None, title="Training Samples", step=0):

    # 反归一化函数(确保正确)
    def safe_denorm(tensor):
        # 显式处理每个通道
        denormed = tensor.clone()
        for c in range(3):
            denormed[c] = denormed[c] * 0.5 + 0.5
        return denormed

    # 应用反归一化
    img_A = safe_denorm(real_A[0]).cpu().permute(1, 2, 0).numpy()
    img_fake_B = safe_denorm(fake_B[0]).cpu().permute(1, 2, 0).detach().numpy()
    img_rec_A = safe_denorm(rec_A[0]).cpu().permute(1, 2, 0).detach().numpy()

    # 创建子图
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(title, fontsize=16)

    # 原始A域图像
    axes[0].imshow(img_A)
    axes[0].set_title("Real A")
    axes[0].axis('off')

    # 生成B域图像
    axes[1].imshow(img_fake_B)
    axes[1].set_title("Fake B")
    axes[1].axis('off')

    # 重建A域图像
    axes[2].imshow(img_rec_A)
    axes[2].set_title("Reconstructed A")
    axes[2].axis('off')

    # 语义差异图
    sem_A_idx = semantic_A[0].cpu().numpy()
    sem_hat_idx = semantic_hat.argmax(1)[0].cpu().numpy()
    diff = np.abs(sem_A_idx - sem_hat_idx)

    axes[3].imshow(diff, cmap='hot')
    axes[3].set_title("Semantic Difference")
    axes[3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}_batch_{step}.png", dpi=120, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def save_checkpoint(model, optimizer_G, optimizer_D, epoch, save_dir, filename_prefix='sem_cyclegan'):
    """保存模型检查点"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    checkpoint_path = os.path.join(save_dir, f"{filename_prefix}_epoch_{epoch}.pth")

    state = {
        'epoch': epoch,
        'G_A2B_state': model.G_A2B.state_dict(),
        'G_B2A_state': model.G_B2A.state_dict(),
        'D_A_state': model.D_A.state_dict(),
        'D_B_state': model.D_B.state_dict(),
        'optimizer_G_state': optimizer_G.state_dict(),
        'optimizer_D_state': optimizer_D.state_dict(),
        'semantic_encoder_state': model.semantic_encoder.state_dict(),
        'semantic_decoder_state': model.semantic_decoder.state_dict(),
    }

    torch.save(state, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(model, optimizer_G, optimizer_D, checkpoint_path, device='cpu'):
    """加载模型检查点"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return 0

    state = torch.load(checkpoint_path, map_location=device)

    # 加载模型状态
    model.G_A2B.load_state_dict(state['G_A2B_state'])
    model.G_B2A.load_state_dict(state['G_B2A_state'])
    model.D_A.load_state_dict(state['D_A_state'])
    model.D_B.load_state_dict(state['D_B_state'])
    model.semantic_encoder.load_state_dict(state['semantic_encoder_state'])
    model.semantic_decoder.load_state_dict(state['semantic_decoder_state'])

    # 加载优化器状态
    if optimizer_G is not None:
        optimizer_G.load_state_dict(state['optimizer_G_state'])
    if optimizer_D is not None:
        optimizer_D.load_state_dict(state['optimizer_D_state'])

    epoch = state['epoch']
    print(f"Loaded checkpoint from epoch {epoch}")

    return epoch


def visualize_losses(loss_history, save_path=None):
    """可视化训练损失"""
    plt.figure(figsize=(12, 8))

    # 绘制生成器损失
    plt.subplot(2, 2, 1)
    plt.plot(loss_history['G_loss'], label='Generator Loss')
    plt.title('Generator Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制判别器损失
    plt.subplot(2, 2, 2)
    plt.plot(loss_history['D_loss'], label='Discriminator Loss', color='orange')
    plt.title('Discriminator Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制语义损失
    plt.subplot(2, 2, 3)
    plt.plot(loss_history['sem_loss'], label='Semantic Loss', color='green')
    plt.title('Semantic Consistency Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制循环损失
    plt.subplot(2, 2, 4)
    plt.plot(loss_history['cycle_loss'], label='Cycle Loss', color='red')
    plt.title('Cycle Consistency Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def calculate_iou(pred, target, n_classes):
    """计算IoU(交并比)"""
    ious = []
    # 展平批次和空间维度
    pred = pred.view(-1)  # [B*H*W]
    target = target.view(-1)  # [B*H*W]

    # 忽略背景类0
    for cls in range(1, n_classes):  # 从1开始，跳过背景
        pred_inds = (pred == cls)
        target_inds = (target == cls)

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())

    # 计算非背景类的平均IoU，忽略NaN
    return np.nanmean(ious)


def compute_semantic_metrics(model, dataloader, device, n_classes=6):
    """计算语义相关指标"""
    model.eval()
    total_iou = 0.0
    total_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, sem_targets in tqdm(dataloader, desc="Computing metrics"):
            images = images.to(device)
            sem_targets = sem_targets.to(device)

            # 获取语义特征
            sem_features, _ = model.semantic_encoder(images)
            # 重建语义图
            sem_pred = model.semantic_decoder(sem_features)

            # 转换为类别索引
            pred_classes = sem_pred.argmax(1)  # [B, H, W]

            # === 修复开始 ===
            # 处理目标语义图：如果是one-hot编码则转换为类别索引
            if sem_targets.dim() == 4:  # one-hot编码 [B, C, H, W]
                target_classes = sem_targets.argmax(1)  # [B, H, W]
            else:  # 已经是类别索引 [B, H, W]
                target_classes = sem_targets.long()
            # === 修复结束 ===

            # 计算准确率
            # 展平空间维度 [B, H, W] -> [B*H*W]
            acc = (pred_classes.view(-1) == target_classes.view(-1)).float().mean().item()
            total_acc += acc * images.size(0)

            # 计算IoU
            iou = calculate_iou(pred_classes, target_classes, n_classes)
            if not np.isnan(iou):
                total_iou += iou * images.size(0)

            total_samples += images.size(0)

    mean_acc = total_acc / total_samples
    mean_iou = total_iou / total_samples

    model.train()
    return {'accuracy': mean_acc, 'iou': mean_iou}


def generate_adverse_weather(model, input_image, output_path, device='cpu'):
    """生成恶劣天气图像"""
    model.eval()

    # 预处理输入图像
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if isinstance(input_image, str):
        image = Image.open(input_image).convert('RGB')
    else:
        image = input_image

    input_tensor = transform(image).unsqueeze(0).to(device)

    # 生成语义特征
    with torch.no_grad():
        sem_feature = model.semantic_encoder(input_tensor)[0]

        # 生成恶劣天气图像
        adverse_weather = model.generate_A_to_B(input_tensor, sem_feature)

        # 保存结果
        save_image(adverse_weather, output_path, normalize=True)

    print(f"Generated adverse weather image saved to {output_path}")
    return output_path


def visualize_complete_workflow(model, real_A, sem_A, fake_B_dict, rec_A,
                                epoch, global_step, save_dir, device='cuda', num_samples=3):
    """完整的可视化函数 - 确保正确显示天气特效"""

    os.makedirs(save_dir, exist_ok=True)

    # 处理多个样本
    for sample_idx in range(min(num_samples, real_A.size(0))):
        real_A_single = real_A[sample_idx:sample_idx + 1]
        sem_A_single = sem_A[sample_idx:sample_idx + 1]

        with torch.no_grad():
            # 获取语义特征
            sem_features, aux_features = model.semantic_encoder(real_A_single)

            # 直接使用fake_B_dict中已有的组件，而不是重新计算
            # 确保这些组件确实存在
            base_scene = fake_B_dict.get('base_scene', real_A_single)
            weather_effect = fake_B_dict.get('weather_effect', None)

            # 如果weather_effect不存在，尝试从其他键获取
            if weather_effect is None:
                # 尝试不同的键名
                for key in ['weather_layer_output', 'rain_effect', 'fog_effect', 'snow_effect']:
                    if key in fake_B_dict:
                        weather_effect = fake_B_dict[key]
                        break

            # 如果还是找不到天气特效，创建一个空的
            if weather_effect is None:
                print("警告: 未找到天气特效，使用零张量代替")
                weather_effect = torch.zeros_like(real_A_single)

            # 获取最终输出
            final_output = fake_B_dict.get('output', real_A_single)

            # 获取区域掩码
            region_mask = fake_B_dict.get('region_mask', None)
            if region_mask is None:
                # 如果没有区域掩码，创建一个默认的
                region_mask = torch.zeros(real_A_single.size(0), 3, 32, 32, device=device)
                region_mask[:, 0] = 1.0  # 默认全部为天空

            # 创建可视化
            create_detailed_visualization(model,
                real_A_single, base_scene, weather_effect, final_output, region_mask,
                sem_features, sem_A_single, epoch, global_step, sample_idx, save_dir
            )

            # 创建对比可视化
            create_comparison_visualization(
                real_A_single, base_scene, weather_effect, final_output,
                epoch, global_step, sample_idx, save_dir
            )


def create_detailed_visualization(model,real_A, base_scene, weather_effect, final_output,
                                  region_mask, sem_features, sem_target, epoch, step, sample_idx, save_dir):
    """创建详细的可视化"""
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    def denorm(tensor):
        return (tensor + 1) / 2

    # 第一行：输入和语义
    axes[0, 0].imshow(denorm(real_A[0].cpu()).permute(1, 2, 0).numpy())
    axes[0, 0].set_title("1. 原始输入", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # 语义分割结果
    sem_pred = model.semantic_decoder(sem_features)
    sem_classes = sem_pred.argmax(1)[0].cpu().numpy()
    color_map = np.array(
        [[0, 0, 0], [128, 64, 128], [70, 70, 70], [107, 142, 35], [70, 130, 180], [220, 20, 60]]) / 255.0
    sem_colored = color_map[sem_classes]
    axes[0, 1].imshow(sem_colored)
    axes[0, 1].set_title("2. 语义分割", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # 区域掩码 - 上采样到与天气特效相同的尺寸
    if weather_effect is not None:
        target_size = weather_effect.shape[-2:]
    else:
        target_size = real_A.shape[-2:]

    region_mask_up = F.interpolate(region_mask, size=target_size, mode='bilinear', align_corners=False)
    region_vis = region_mask_up[0].cpu().permute(1, 2, 0).numpy()
    axes[0, 2].imshow(region_vis)
    axes[0, 2].set_title("3. 区域掩码", fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    # 语义特征
    sem_feat_vis = sem_features[0].mean(0).cpu().numpy()
    axes[0, 3].imshow(sem_feat_vis, cmap='viridis')
    axes[0, 3].set_title("4. 语义特征", fontsize=12, fontweight='bold')
    axes[0, 3].axis('off')

    # 第二行：生成过程
    axes[1, 0].imshow(denorm(base_scene[0].cpu()).permute(1, 2, 0).numpy())
    axes[1, 0].set_title("5. 基础场景", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    # 天气特效（确保正确显示）
    if weather_effect is not None:
        weather_vis = weather_effect[0].cpu()
        if weather_vis.shape[0] == 3:  # RGB图像
            weather_vis = denorm(weather_vis)
        else:  # 单通道或多通道特征
            weather_vis = (weather_vis - weather_vis.min()) / (weather_vis.max() - weather_vis.min() + 1e-6)
            if weather_vis.shape[0] > 3:  # 多通道，取平均
                weather_vis = weather_vis.mean(0, keepdim=True)
            weather_vis = weather_vis.repeat(3, 1, 1)  # 转为RGB

        axes[1, 1].imshow(weather_vis.permute(1, 2, 0).numpy())
        axes[1, 1].set_title("6. 纯天气特效", fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')

        # 天气特效叠加（半透明）
        base_np = denorm(base_scene[0].cpu()).permute(1, 2, 0).numpy()
        weather_np = weather_vis.permute(1, 2, 0).numpy()
        blended = 0.7 * base_np + 0.3 * weather_np
        axes[1, 2].imshow(blended)
        axes[1, 2].set_title("7. 特效叠加演示", fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')

        # 区域融合效果 - 使用上采样后的区域掩码
        sky_effect = weather_effect * region_mask_up[:, 0:1]
        dynamic_effect = weather_effect * region_mask_up[:, 1:2]
        ground_effect = weather_effect * region_mask_up[:, 2:3]
        regional_combined = (sky_effect + dynamic_effect + ground_effect)[0].cpu()

        if regional_combined.shape[0] == 3:
            regional_combined = denorm(regional_combined)
        else:
            regional_combined = (regional_combined - regional_combined.min()) / (
                        regional_combined.max() - regional_combined.min() + 1e-6)
            if regional_combined.shape[0] > 3:
                regional_combined = regional_combined.mean(0, keepdim=True)
            regional_combined = regional_combined.repeat(3, 1, 1)

        axes[1, 3].imshow(regional_combined.permute(1, 2, 0).numpy())
        axes[1, 3].set_title("8. 区域加权特效", fontsize=12, fontweight='bold')
        axes[1, 3].axis('off')
    else:
        # 如果没有天气特效，显示占位符
        for i in range(1, 4):
            axes[1, i].text(0.5, 0.5, "无天气特效数据", ha='center', va='center', transform=axes[1, i].transAxes)
            axes[1, i].set_title(f"{5 + i}. 无数据", fontsize=12, fontweight='bold')
            axes[1, i].axis('off')

    # 第三行：输出和比较
    axes[2, 0].imshow(denorm(final_output[0].cpu()).permute(1, 2, 0).numpy())
    axes[2, 0].set_title("9. 最终输出", fontsize=12, fontweight='bold')
    axes[2, 0].axis('off')

    # 差异可视化
    diff = torch.abs(real_A[0].cpu() - final_output[0].cpu()).mean(0)
    axes[2, 1].imshow(diff.numpy(), cmap='hot')
    axes[2, 1].set_title("10. 输入输出差异", fontsize=12, fontweight='bold')
    axes[2, 1].axis('off')

    # 边缘保持
    orig_edges = torch.abs(real_A[0, :, 1:] - real_A[0, :, :-1]).mean(0)
    fake_edges = torch.abs(final_output[0, :, 1:] - final_output[0, :, :-1]).mean(0)
    axes[2, 2].imshow(orig_edges.cpu().numpy(), cmap='gray')
    axes[2, 2].set_title("11. 原始边缘", fontsize=12, fontweight='bold')
    axes[2, 2].axis('off')

    axes[2, 3].imshow(fake_edges.cpu().numpy(), cmap='gray')
    axes[2, 3].set_title("12. 生成边缘", fontsize=12, fontweight='bold')
    axes[2, 3].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"detailed_epoch{epoch + 1}_step{step}_sample{sample_idx}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ 详细可视化已保存: {save_path}")


def create_comparison_visualization(real_A, base_scene, weather_effect, final_output,
                                    epoch, step, sample_idx, save_dir):
    """创建对比可视化"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    def denorm(tensor):
        return (tensor + 1) / 2

    # 第一行：生成流程
    axes[0, 0].imshow(denorm(real_A[0].cpu()).permute(1, 2, 0).numpy())
    axes[0, 0].set_title("输入图像", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(denorm(base_scene[0].cpu()).permute(1, 2, 0).numpy())
    axes[0, 1].set_title("基础场景", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # 天气特效显示
    if weather_effect is not None:
        weather_vis = weather_effect[0].cpu()
        if weather_vis.shape[0] == 3:
            weather_vis = denorm(weather_vis)
        else:
            weather_vis = (weather_vis - weather_vis.min()) / (weather_vis.max() - weather_vis.min() + 1e-6)
            if weather_vis.shape[0] > 3:
                weather_vis = weather_vis.mean(0, keepdim=True)
            weather_vis = weather_vis.repeat(3, 1, 1)

        axes[0, 2].imshow(weather_vis.permute(1, 2, 0).numpy())
        axes[0, 2].set_title("天气特效", fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
    else:
        axes[0, 2].text(0.5, 0.5, "无天气特效", ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title("天气特效", fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')

    # 第二行：结果对比
    axes[1, 0].imshow(denorm(final_output[0].cpu()).permute(1, 2, 0).numpy())
    axes[1, 0].set_title("最终输出", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    # 并排对比
    combined = np.concatenate([
        denorm(real_A[0].cpu()).permute(1, 2, 0).numpy(),
        denorm(final_output[0].cpu()).permute(1, 2, 0).numpy()
    ], axis=1)
    axes[1, 1].imshow(combined)
    axes[1, 1].set_title("输入 vs 输出", fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    # 差异图
    diff = torch.abs(real_A[0].cpu() - final_output[0].cpu()).mean(0)
    im = axes[1, 2].imshow(diff.numpy(), cmap='hot')
    axes[1, 2].set_title("差异热力图", fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"comparison_epoch{epoch + 1}_step{step}_sample{sample_idx}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ 对比可视化已保存: {save_path}")