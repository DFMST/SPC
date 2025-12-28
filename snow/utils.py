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




def visualize_semantic_mapping(image, semantic_map, save_path):
    """
    可视化图像和对应的语义分割图，用于验证类别映射是否正确
    """

    # 反归一化图像
    def safe_denorm(tensor):
        denormed = tensor.clone()
        for c in range(3):
            denormed[c] = denormed[c] * 0.5 + 0.5
        return denormed.permute(1, 2, 0).detach().cpu().numpy()

    # 创建颜色映射 - 为每个类别分配不同的颜色
    colors = [
        [0, 0, 0],  # 0: 背景 - 黑色
        [255, 0, 0],  # 1: 道路 - 红色
        [0, 255, 0],  # 2: 标志 - 绿色
        [0, 0, 255],  # 3: 植被 - 蓝色
        [255, 255, 0],  # 4: 天空 - 黄色
        [255, 0, 255]  # 5: 动态物体 - 紫色
    ]

    # 将语义图转换为彩色图像
    semantic_np = semantic_map.cpu().numpy()
    semantic_color = np.zeros((semantic_np.shape[0], semantic_np.shape[1], 3), dtype=np.uint8)

    for class_id in range(6):
        mask = semantic_np == class_id
        semantic_color[mask] = colors[class_id]

    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 原始图像
    img = safe_denorm(image)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    ax1.imshow(img)
    ax1.set_title("原始图像")
    ax1.axis('off')

    # 语义分割图
    ax2.imshow(semantic_color)
    ax2.set_title("语义分割图 (彩色编码)")
    ax2.axis('off')

    # 添加图例
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=np.array(colors[i]) / 255) for i in range(6)
    ]
    legend_labels = ["背景", "道路", "标志", "植被", "天空", "动态物体"]
    ax2.legend(legend_elements, legend_labels, loc='upper right', bbox_to_anchor=(1.3, 1))

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()

    print(f"语义映射可视化已保存到: {save_path}")


def map_acdc_to_custom(semantic):
    """
    将ACDC语义标签正确映射到自定义类别
    基于Cityscapes标准19个语义类别
    """
    mapped = np.zeros_like(semantic, dtype=np.uint8)

    # 使用Cityscapes标准ID映射
    # 道路类别 (1)
    mapped[semantic == 7] = 1  # road
    mapped[semantic == 8] = 1  # sidewalk

    # 建筑类别 (0 - 背景)
    mapped[semantic == 11] = 0  # building
    mapped[semantic == 12] = 0  # wall
    mapped[semantic == 13] = 0  # fence
    mapped[semantic == 17] = 0  # pole

    # 交通标志类别 (2)
    mapped[semantic == 19] = 2  # traffic light
    mapped[semantic == 20] = 2  # traffic sign

    # 植被类别 (3)
    mapped[semantic == 21] = 3  # vegetation
    mapped[semantic == 22] = 3  # terrain

    # 天空类别 (4)
    mapped[semantic == 23] = 4  # sky

    # 动态物体类别 (5)
    mapped[semantic == 24] = 5  # person
    mapped[semantic == 25] = 5  # rider
    mapped[semantic == 26] = 5  # car
    mapped[semantic == 27] = 5  # truck
    mapped[semantic == 28] = 5  # bus
    mapped[semantic == 31] = 5  # train
    mapped[semantic == 32] = 5  # motorcycle
    mapped[semantic == 33] = 5  # bicycle

    # 其他类别映射到背景 (0)
    # 包括: 0:unlabeled, 1:ego vehicle, 2:rectification border, 3:out of roi
    # 4:static, 5:dynamic, 6:ground, 9:parking, 10:rail track
    # 14:guard rail, 15:bridge, 16:tunnel, 18:polegroup
    # 29:caravan, 30:trailer, 34:license plate
    other_ids = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, 34]
    for id in other_ids:
        mapped[semantic == id] = 0

    return mapped

def visualize_physics(model, real_A, fake_B, outputs_dict, save_path):
    """可视化物理约束效果"""
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    # 反归一化函数（确保正确）
    def safe_denorm(tensor):
        denormed = tensor.clone()
        for c in range(3):
            denormed[c] = denormed[c] * 0.5 + 0.5
        return denormed.permute(1, 2, 0).detach().cpu().numpy()

    # 原始图像 [C, H, W] -> [H, W, C]
    img_real = safe_denorm(real_A[0])
    axs[0, 0].imshow(img_real)
    axs[0, 0].set_title("原始图像")
    axs[0, 0].axis('off')

    # 基础场景 [C, H, W] -> [H, W, C]
    base_scene = outputs_dict['base_scene'][0]
    base_img = safe_denorm(base_scene)
    axs[0, 1].imshow(base_img)
    axs[0, 1].set_title("基础场景")
    axs[0, 1].axis('off')

    # 天气特效 [C, H, W] -> [H, W, C]
    weather_effect = outputs_dict['weather_effect'][0].cpu()
    weather_img = (weather_effect - weather_effect.min()) / (weather_effect.max() - weather_effect.min() + 1e-6)
    weather_img = weather_img.permute(1, 2, 0).numpy()
    axs[0, 2].imshow(weather_img)
    axs[0, 2].set_title("天气特效")
    axs[0, 2].axis('off')

    # 区域掩码 [C, H, W] -> [H, W, C]
    region_mask = outputs_dict['region_mask'][0].cpu().detach()
    region_img = region_mask.permute(1, 2, 0).numpy()
    axs[1, 0].imshow(region_img)
    axs[1, 0].set_title("区域掩码(天空/动态/地面)")
    axs[1, 0].axis('off')

    # 深度图 [1, H, W] -> [H, W]
    depth_map = model.G_A2B.weather_layer.fog_generator.depth_predictor(real_A)
    depth_img = depth_map[0, 0].cpu().detach().numpy()
    axs[1, 1].imshow(depth_img, cmap='viridis')
    axs[1, 1].set_title("深度图")
    axs[1, 1].axis('off')

    # 最终结果 [C, H, W] -> [H, W, C]
    fake_img = safe_denorm(fake_B[0])
    axs[1, 2].imshow(fake_img)
    axs[1, 2].set_title("最终输出")
    axs[1, 2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_physics.png", dpi=120, bbox_inches='tight')
    plt.close()


def save_individual_components(outputs_dict, save_dir, step):
    """分别保存各个组件为单独的图像文件"""
    os.makedirs(save_dir, exist_ok=True)

    # 反归一化函数
    def safe_denorm(tensor):
        denormed = tensor.clone()
        for c in range(3):
            denormed[c] = denormed[c] * 0.5 + 0.5
        return denormed

    # 保存真实图像
    if 'real_A' in outputs_dict:
        real_img = safe_denorm(outputs_dict['real_A'][0]).cpu()
        save_image(real_img, os.path.join(save_dir, f"real_{step}.png"))

    # 保存基础场景
    if 'base_scene' in outputs_dict:
        base_scene = safe_denorm(outputs_dict['base_scene'][0]).cpu()
        save_image(base_scene, os.path.join(save_dir, f"base_scene_{step}.png"))

    # 保存天气特效
    if 'weather_effect' in outputs_dict:
        weather_effect = outputs_dict['weather_effect'][0].cpu()
        # 归一化到0-1范围
        weather_effect = (weather_effect - weather_effect.min()) / (weather_effect.max() - weather_effect.min() + 1e-6)
        save_image(weather_effect, os.path.join(save_dir, f"weather_effect_{step}.png"))

    # 保存区域掩码
    if 'region_mask' in outputs_dict:
        region_mask = outputs_dict['region_mask'][0].cpu()
        # 保存每个通道的掩码
        for i, channel_name in enumerate(['sky', 'dynamic', 'ground']):
            mask_channel = region_mask[i].unsqueeze(0)  # 添加通道维度
            save_image(mask_channel, os.path.join(save_dir, f"region_mask_{channel_name}_{step}.png"))

    # 保存最终生成图像
    if 'fake_B' in outputs_dict:
        fake_img = safe_denorm(outputs_dict['fake_B'][0]).cpu()
        save_image(fake_img, os.path.join(save_dir, f"final_output_{step}.png"))

    # 保存语义分割结果
    if 'sem_hat_A' in outputs_dict:
        sem_pred = outputs_dict['sem_hat_A'][0].argmax(0).float().cpu() / 5.0  # 归一化到0-1
        save_image(sem_pred.unsqueeze(0), os.path.join(save_dir, f"semantic_pred_{step}.png"))

    print(f"保存单独组件到: {save_dir}")

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


def get_transforms(image_size=256, phase='train'):
    """获取无数据增强的变换"""
    # 仅包含必要转换
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 语义图无需额外变换
    return image_transform, None





def visualize_batch(real_A, fake_B, rec_A, save_path=None, title="Training Samples", step=0):
    # 反归一化函数
    def denorm(tensor):
        return tensor * 0.5 + 0.5

    # 处理批量维度并转换为numpy
    img_A = denorm(real_A[0]).cpu().permute(1, 2, 0).numpy().astype(np.float32)
    img_fake_B = denorm(fake_B[0]).cpu().permute(1, 2, 0).numpy().astype(np.float32)
    img_rec_A = denorm(rec_A[0]).cpu().permute(1, 2, 0).numpy().astype(np.float32)

    # 确保值在[0,1]范围内
    img_A = np.clip(img_A, 0, 1)
    img_fake_B = np.clip(img_fake_B, 0, 1)
    img_rec_A = np.clip(img_rec_A, 0, 1)

    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)

    # 显示图像
    axes[0].imshow(img_A)
    axes[0].set_title("Real A")
    axes[0].axis('off')

    axes[1].imshow(img_fake_B)
    axes[1].set_title("Fake B")
    axes[1].axis('off')

    axes[2].imshow(img_rec_A)
    axes[2].set_title("Reconstructed A")
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(f"{save_path}_batch_{step}.png", dpi=120, bbox_inches='tight')
        except Exception as e:
            print(f"保存图像失败: {e}")
            # 尝试使用不同的后端
            try:
                plt.savefig(f"{save_path}_batch_{step}.png", dpi=100)
            except Exception as e2:
                print(f"简化保存也失败: {e2}")
        finally:
            plt.close(fig)
    else:
        plt.show()

def safe_denorm(tensor):
    """
    将 [-1, 1] 范围的张量转换为 [0, 1] 范围的图像
    """
    denormed = tensor.clone()
    for c in range(3):
        denormed[c] = denormed[c] * 0.5 + 0.5
    return denormed
def visualize_clean_base(model, real_A, save_dir, step=0):
    """
    生成并可视化未经天气特效处理的"干净"基础场景
    参数:
        model: 训练好的模型
        real_A: 原始输入图像 [B, C, H, W]
        save_dir: 保存图像的目录
        step: 当前训练步数（用于文件名）
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        # 1. 获取语义特征
        sem_feature_A, _ = model.semantic_encoder(real_A)

        # 2. 关键步骤：只运行到base_generator，不经过weather_layer
        # 注意：这里假设您的模型结构允许这样访问
        clean_base_scene = model.G_A2B.base_generator(real_A, sem_feature_A)

        # 3. 反归一化并保存
        clean_base_img = safe_denorm(clean_base_scene[0]).cpu().permute(1, 2, 0).numpy()

        # 确保数据类型正确
        if clean_base_img.dtype != np.uint8:
            clean_base_img = (clean_base_img * 255).astype(np.uint8)

        # 保存图像
        plt.figure(figsize=(8, 8))
        plt.imshow(clean_base_img)
        plt.axis('off')
        plt.title("Clean Base Scene (No Weather Effects)")
        plt.savefig(os.path.join(save_dir, f"00_clean_base_{step}.png"),
                    dpi=120, bbox_inches='tight', pad_inches=0)
        plt.close()

    print(f"干净基础场景已保存到: {save_dir}/00_clean_base_{step}.png")
    return clean_base_scene


def visualize_physics_separate(model, real_A, fake_B, semantic_A, outputs_dict, save_dir, step=0):
    """
    可视化物理约束效果-分离版本（增强版，包含干净基础场景）
    参数:
        model: 模型实例
        real_A: 原始输入图像 [B, C, H, W]
        fake_B: 生成的恶劣天气图像 [B, C, H, W]
        semantic_A: 语义分割图 [B, H, W] (类别索引)
        outputs_dict: 生成器的输出字典
        save_dir: 保存图像的目录
        step: 当前训练步数（用于文件名）
    """
    os.makedirs(save_dir, exist_ok=True)

    # 获取第一个样本
    idx = 0

    # 1. 保存原始图像
    img_real = safe_denorm(real_A[idx])
    img_real = img_real.permute(1, 2, 0).detach().cpu().numpy()
    if img_real.dtype != np.uint8:
        img_real = (img_real * 255).astype(np.uint8)
    plt.figure(figsize=(8, 8))
    plt.imshow(img_real)
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f"01_original_{step}.png"),
                dpi=120, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 2. 生成并保存干净的基础场景（未添加天气特效）
    model.eval()
    with torch.no_grad():
        # 获取语义特征
        sem_feature_A, _ = model.semantic_encoder(real_A)

        # 只运行到base_generator，不经过weather_layer
        clean_base_scene = model.G_A2B.base_generator(real_A, sem_feature_A)

        # 反归一化并保存
        clean_base_img = safe_denorm(clean_base_scene[idx])
        clean_base_img = clean_base_img.permute(1, 2, 0).detach().cpu().numpy()
        if clean_base_img.dtype != np.uint8:
            clean_base_img = (clean_base_img * 255).astype(np.uint8)

        plt.figure(figsize=(8, 8))
        plt.imshow(clean_base_img)
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f"00_clean_base_{step}.png"),
                    dpi=120, bbox_inches='tight', pad_inches=0)
        plt.close()

    # 3. 保存当前的基础场景（可能已经为天气调整过）
    if 'base_scene' in outputs_dict and outputs_dict['base_scene'] is not None:
        base_scene = outputs_dict['base_scene'][idx]
        base_img = safe_denorm(base_scene)
        base_img = base_img.permute(1, 2, 0).detach().cpu().numpy()
        if base_img.dtype != np.uint8:
            base_img = (base_img * 255).astype(np.uint8)
        plt.figure(figsize=(8, 8))
        plt.imshow(base_img)
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f"02_base_scene_{step}.png"),
                    dpi=120, bbox_inches='tight', pad_inches=0)
        plt.close()

    # 4. 保存天气特效
    if 'weather_effect' in outputs_dict and outputs_dict['weather_effect'] is not None:
        weather_effect = outputs_dict['weather_effect'][idx].cpu()
        weather_img = (weather_effect - weather_effect.min()) / (weather_effect.max() - weather_effect.min() + 1e-6)
        weather_img = weather_img.permute(1, 2, 0).numpy()
        if weather_img.dtype != np.uint8:
            weather_img = (weather_img * 255).astype(np.uint8)
        plt.figure(figsize=(8, 8))
        plt.imshow(weather_img)
        plt.axis('off')

        plt.savefig(os.path.join(save_dir, f"03_weather_effect_{step}.png"),
                    dpi=120, bbox_inches='tight', pad_inches=0)
        plt.close()

    # 5. 保存语义掩码
    sem_map = semantic_A[idx].cpu().numpy()

    # 天空掩码(假设天空类别为4)
    sky_mask = (sem_map == 4).astype(np.float32)
    plt.figure(figsize=(8, 8))
    plt.imshow(sky_mask, cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f"04_sky_mask_{step}.png"),
                dpi=120, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 道路掩码(假设道路类别为1)
    road_mask = (sem_map == 1).astype(np.float32)
    plt.figure(figsize=(8, 8))
    plt.imshow(road_mask, cmap='gray')
    plt.axis('off')

    plt.savefig(os.path.join(save_dir, f"05_road_mask_{step}.png"),
                dpi=120, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 动态物体掩码(假设动态物体类别为5)
    dynamic_mask = (sem_map == 5).astype(np.float32)
    plt.figure(figsize=(8, 8))
    plt.imshow(dynamic_mask, cmap='gray')
    plt.axis('off')

    plt.savefig(os.path.join(save_dir, f"06_dynamic_mask_{step}.png"),
                dpi=120, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 新增：保存彩色结构化语义分割图
    # 创建颜色映射
    colors = [
        [0, 0, 0],  # 0: 背景 - 黑色
        [255, 0, 0],  # 1: 道路 - 红色
        [0, 255, 0],  # 2: 标志 - 绿色
        [0, 0, 255],  # 3: 植被 - 蓝色
        [255, 255, 0],  # 4: 天空 - 黄色
        [255, 0, 255]  # 5: 动态物体 - 紫色
    ]

    # 创建彩色语义图
    semantic_color = np.zeros((sem_map.shape[0], sem_map.shape[1], 3), dtype=np.uint8)
    for class_id in range(6):
        mask = sem_map == class_id
        semantic_color[mask] = colors[class_id]

    # 保存彩色语义图（无标题、无边框）
    plt.figure(figsize=(sem_map.shape[1] / 100, sem_map.shape[0] / 100), dpi=100)
    plt.imshow(semantic_color)
    plt.axis('off')  # 关闭坐标轴
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 移除所有边距
    plt.savefig(os.path.join(save_dir, f"07_semantic_color_{step}.png"),
                dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"彩色结构化语义分割图已保存到: {os.path.join(save_dir, f'07_semantic_color_{step}.png')}")

    # 6. 保存深度图
    try:
        if hasattr(model.G_A2B.weather_layer, 'fog_generator') and \
                hasattr(model.G_A2B.weather_layer.fog_generator, 'depth_predictor'):
            depth_map = model.G_A2B.weather_layer.fog_generator.depth_predictor(real_A)
            depth_img = depth_map[idx, 0].cpu().detach().numpy()
            plt.figure(figsize=(8, 8))
            plt.imshow(depth_img, cmap='viridis')
            plt.axis('off')
            plt.colorbar()

            plt.savefig(os.path.join(save_dir, f"08_depth_map_{step}.png"),
                        dpi=120, bbox_inches='tight', pad_inches=0)
            plt.close()
    except Exception as e:
        print(f"深度图可视化失败: {e}")

    # 7. 保存区域掩码
    if 'region_mask' in outputs_dict and outputs_dict['region_mask'] is not None:
        region_mask = outputs_dict['region_mask'][idx].cpu().detach()
        region_img = region_mask.permute(1, 2, 0).numpy()
        plt.figure(figsize=(8, 8))
        plt.imshow(region_img)
        plt.axis('off')

        plt.savefig(os.path.join(save_dir, f"09_region_mask_{step}.png"),
                    dpi=120, bbox_inches='tight', pad_inches=0)
        plt.close()

    # 8. 保存天气特效在不同区域的分布
    if 'weather_effect' in outputs_dict and outputs_dict['weather_effect'] is not None:
        weather_effect = outputs_dict['weather_effect'][idx].cpu()
        weather_img = (weather_effect - weather_effect.min()) / (weather_effect.max() - weather_effect.min() + 1e-6)
        weather_img = weather_img.permute(1, 2, 0).numpy()

        # 天空区域天气特效
        sky_weather = weather_img * np.stack([sky_mask] * 3, axis=2)
        if sky_weather.dtype != np.uint8:
            sky_weather = (sky_weather * 255).astype(np.uint8)
        plt.figure(figsize=(8, 8))
        plt.imshow(sky_weather)
        plt.axis('off')

        plt.savefig(os.path.join(save_dir, f"10_sky_weather_{step}.png"),
                    dpi=120, bbox_inches='tight', pad_inches=0)
        plt.close()

        # 道路区域天气特效
        road_weather = weather_img * np.stack([road_mask] * 3, axis=2)
        if road_weather.dtype != np.uint8:
            road_weather = (road_weather * 255).astype(np.uint8)
        plt.figure(figsize=(8, 8))
        plt.imshow(road_weather)
        plt.axis('off')

        plt.savefig(os.path.join(save_dir, f"11_road_weather_{step}.png"),
                    dpi=120, bbox_inches='tight', pad_inches=0)
        plt.close()

    # 9. 保存最终输出
    fake_img = safe_denorm(fake_B[idx])
    fake_img = fake_img.permute(1, 2, 0).detach().cpu().numpy()
    if fake_img.dtype != np.uint8:
        fake_img = (fake_img * 255).astype(np.uint8)
    plt.figure(figsize=(8, 8))
    plt.imshow(fake_img)
    plt.axis('off')

    plt.savefig(os.path.join(save_dir, f"12_final_output_{step}.png"),
                dpi=120, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 10. 创建一个对比图，将所有图像放在一起
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    # 加载所有保存的图像
    images_to_show = []

    # 尝试加载所有可能的图像
    image_files = [
        f"01_original_{step}.png",
        f"00_clean_base_{step}.png",
        f"02_base_scene_{step}.png",
        f"03_weather_effect_{step}.png",
        f"07_semantic_color_{step}.png",  # 新增彩色语义图
        f"04_sky_mask_{step}.png",
        f"05_road_mask_{step}.png",
        f"06_dynamic_mask_{step}.png",
        f"08_depth_map_{step}.png",
        f"09_region_mask_{step}.png",
        f"10_sky_weather_{step}.png",
        f"12_final_output_{step}.png"
    ]

    # 只加载实际存在的图像
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(save_dir, img_file)
        if os.path.exists(img_path):
            images_to_show.append(plt.imread(img_path))
        else:
            # 如果图像不存在，创建一个空白图像
            blank_img = np.ones((256, 256, 3)) if i < len(image_files) - 1 else np.ones((256, 256, 3))
            images_to_show.append(blank_img)

    # 显示图像
    titles = [
        "原始图像", "干净基础场景", "基础场景", "天气特效",
        "彩色语义图", "天空掩码", "道路掩码", "动态物体掩码",
        "深度图", "区域掩码", "天空区域天气特效", "最终输出"
    ]

    for i, ax in enumerate(axes.flat):
        if i < len(images_to_show):
            ax.imshow(images_to_show[i])
            ax.set_title(titles[i])
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"13_comparison_{step}.png"),
                dpi=120, bbox_inches='tight')
    plt.close()

    print(f"物理可视化图像已保存到: {save_dir}")

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
        transforms.Resize((256, 256)),
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


def plot_learning_curves(train_loss, val_metrics, save_path=None):
    """绘制学习曲线"""
    plt.figure(figsize=(15, 10))

    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(train_loss['G_loss'], label='Generator Loss')
    plt.plot(train_loss['D_loss'], label='Discriminator Loss')
    plt.title('Training Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 语义准确性
    plt.subplot(2, 2, 2)
    plt.plot(val_metrics['sem_acc'], label='Semantic Accuracy')
    plt.plot(val_metrics['sem_iou'], label='Semantic IoU')
    plt.title('Semantic Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()

    # 循环一致性损失
    plt.subplot(2, 2, 3)
    plt.plot(train_loss['cycle_loss'], label='Cycle Loss', color='red')
    plt.title('Cycle Consistency Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # 语义一致性损失
    plt.subplot(2, 2, 4)
    plt.plot(train_loss['sem_loss'], label='Semantic Loss', color='green')
    plt.title('Semantic Consistency Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def analyze_acdc_dataset(data_dir):
    """
    分析ACDC数据集的结构和语义分割图格式
    """
    print("=" * 60)
    print("ACDC数据集分析")
    print("=" * 60)

    # 检查数据集目录结构
    for phase in ['train', 'val']:
        phase_dir = os.path.join(data_dir, phase)
        if os.path.exists(phase_dir):
            print(f"\n{phase.upper()} 阶段目录结构:")
            for domain in ['A', 'B']:
                domain_dir = os.path.join(phase_dir, domain)
                if os.path.exists(domain_dir):
                    files = os.listdir(domain_dir)
                    image_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg')) and '_semantic' not in f]
                    semantic_files = [f for f in files if '_semantic' in f]

                    print(f"  {domain}: {len(image_files)} 图像, {len(semantic_files)} 语义图")

                    # 分析第一个图像和语义图
                    if image_files and semantic_files:
                        img_path = os.path.join(domain_dir, image_files[0])
                        sem_path = os.path.join(domain_dir, semantic_files[0])

                        # 分析图像
                        img = Image.open(img_path)
                        print(f"    图像: {img_path}")
                        print(f"      尺寸: {img.size}, 模式: {img.mode}")

                        # 分析语义图
                        sem = Image.open(sem_path)
                        print(f"    语义图: {sem_path}")
                        print(f"      尺寸: {sem.size}, 模式: {sem.mode}")

                        # 分析语义图的值
                        sem_array = np.array(sem)
                        print(f"      值范围: [{sem_array.min()}, {sem_array.max()}]")
                        unique_vals = np.unique(sem_array)
                        print(f"      唯一值: {unique_vals}")
                        print(f"      值分布: {[(val, np.sum(sem_array == val)) for val in unique_vals]}")

                        # 可视化示例
                        visualize_sample(img, sem_array, f"{phase}_{domain}_sample.png")

    print("=" * 60)


def visualize_sample(image, semantic_array, save_path):
    """
    可视化图像和语义分割图
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # 原始图像
    ax1.imshow(image)
    ax1.set_title("原始图像")
    ax1.axis('off')

    # 语义分割图（灰度）
    ax2.imshow(semantic_array, cmap='gray')
    ax2.set_title("语义图（灰度）")
    ax2.axis('off')

    # 语义分割图（彩色）
    # 尝试使用标准ACDC颜色映射
    try:
        from cityscapesscripts.helpers.labels import labels as cs_labels
        color_map = {label.id: label.color for label in cs_labels if label.id != -1}

        colored_sem = np.zeros((semantic_array.shape[0], semantic_array.shape[1], 3), dtype=np.uint8)
        for val in np.unique(semantic_array):
            if val in color_map:
                colored_sem[semantic_array == val] = color_map[val]
            else:
                colored_sem[semantic_array == val] = [255, 0, 0]  # 红色表示未知类别

        ax3.imshow(colored_sem)
        ax3.set_title("语义图（彩色）")
    except ImportError:
        # 如果没有安装cityscapesscripts，使用随机颜色
        unique_vals = np.unique(semantic_array)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_vals)))

        colored_sem = np.zeros((semantic_array.shape[0], semantic_array.shape[1], 3), dtype=np.uint8)
        for i, val in enumerate(unique_vals):
            colored_sem[semantic_array == val] = colors[i][:3] * 255

        ax3.imshow(colored_sem)
        ax3.set_title("语义图（随机颜色）")

    ax3.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()

    print(f"    示例已保存到: {save_path}")


# 运行分析
def validate_acdc_mapping(data_dir):
    """
    验证ACDC数据集的映射是否正确
    """
    # 加载一个样本
    train_A_dir = os.path.join(data_dir, "train", "A")
    image_files = [f for f in os.listdir(train_A_dir)
                   if f.endswith(('.png', '.jpg', '.jpeg')) and '_semantic' not in f]

    if not image_files:
        print("未找到图像文件")
        return

    img_path = os.path.join(train_A_dir, image_files[0])
    base_name = os.path.splitext(image_files[0])[0]
    semantic_path = os.path.join(train_A_dir, f"{base_name}_semantic.png")

    # 加载语义图
    semantic = Image.open(semantic_path).convert('L')
    semantic_np = np.array(semantic)

    print("原始语义图分析:")
    print(f"  值范围: [{semantic_np.min()}, {semantic_np.max()}]")
    unique_vals = np.unique(semantic_np)
    print(f"  唯一值: {unique_vals}")

    # 应用您的映射
    mapped = map_acdc_to_custom(semantic_np)

    print("\n映射后语义图分析:")
    print(f"  值范围: [{mapped.min()}, {mapped.max()}]")
    mapped_unique = np.unique(mapped)
    print(f"  唯一值: {mapped_unique}")

    # 统计每个类别的像素数量
    class_names = ["背景", "道路", "标志", "植被", "天空", "动态物体"]
    for class_id in range(6):
        count = np.sum(mapped == class_id)
        percentage = count / mapped.size * 100
        print(f"  类别 {class_id} ({class_names[class_id]}): {count} 像素 ({percentage:.2f}%)")

    # 特别检查动态物体
    dynamic_pixels = np.sum(mapped == 5)
    print(f"\n动态物体像素数量: {dynamic_pixels} ({dynamic_pixels / mapped.size * 100:.2f}%)")

    # 可视化原始语义图和映射后的语义图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 原始语义图
    ax1.imshow(semantic_np, cmap='tab20')
    ax1.set_title("原始语义图 (Cityscapes ID)")
    ax1.axis('off')

    # 映射后语义图
    ax2.imshow(mapped, cmap='tab10', vmin=0, vmax=5)
    ax2.set_title("映射后语义图 (自定义类别)")
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "mapping_validation.png"), dpi=120, bbox_inches='tight')
    plt.close()

    print(f"\n验证结果已保存到: {os.path.join(data_dir, 'mapping_validation.png')}")


def check_semantic_format(data_dir):
    """检查语义图的格式"""
    train_A_dir = os.path.join(data_dir, "train", "A")

    # 查找语义图文件
    sem_files = [f for f in os.listdir(train_A_dir) if '_semantic' in f]

    if not sem_files:
        print("未找到语义图文件")
        return

    # 检查文件后缀
    for sem_file in sem_files[:3]:  # 检查前3个文件
        print(f"语义图文件: {sem_file}")

        # 加载并分析
        sem_path = os.path.join(train_A_dir, sem_file)
        semantic = np.array(Image.open(sem_path))

        print(f"  值范围: [{semantic.min()}, {semantic.max()}]")
        unique_vals = np.unique(semantic)
        print(f"  唯一值: {unique_vals}")
        print(f"  形状: {semantic.shape}")

        # 检查是否是labelTrainIds格式（值范围0-18）
        if semantic.max() <= 18:
            print("  格式: 可能是labelTrainIds")
        else:
            print("  格式: 可能是labelIds")


def generate_paper_figures(model, real_A, semantic_A, outputs_dict, save_dir, step=0):
    """
    生成论文所需的机制图 - 修复版本
    特点：无标题、确保输出最终图像、数据类型安全
    """
    os.makedirs(save_dir, exist_ok=True)

    # 获取第一个样本
    idx = 0

    def safe_convert_to_uint8(img_array):
        """安全地将图像数组转换为uint8格式"""
        if img_array.dtype != np.uint8:
            # 确保值在[0, 1]范围内
            img_array = np.clip(img_array, 0, 1)
            # 转换为uint8
            img_array = (img_array * 255).astype(np.uint8)
        return img_array

    def safe_denorm_to_numpy(tensor):
        """安全地将张量转换为numpy数组"""
        if tensor.dim() == 3:  # [C, H, W]
            # 反归一化
            denormed = tensor.clone()
            for c in range(min(3, denormed.shape[0])):
                denormed[c] = denormed[c] * 0.5 + 0.5
            # 转换为numpy并调整维度
            if denormed.shape[0] == 3:  # RGB图像
                img_array = denormed.permute(1, 2, 0).detach().cpu().numpy()
            else:  # 单通道图像
                img_array = denormed.squeeze(0).detach().cpu().numpy()
            return safe_convert_to_uint8(img_array)
        else:
            # 处理其他维度情况
            return np.zeros((256, 256, 3), dtype=np.uint8)

    # 确保最终图像一定会输出
    def ensure_final_output():
        """确保有最终输出图像"""
        final_img_path = os.path.join(save_dir, f"05_final_result_{step}.png")

        # 检查是否已经有最终图像
        if os.path.exists(final_img_path):
            return

        # 如果没有最终图像，尝试从不同来源生成
        try:
            if 'fake_B' in outputs_dict and outputs_dict['fake_B'] is not None:
                final_img = safe_denorm_to_numpy(outputs_dict['fake_B'][idx])
            elif 'base_scene' in outputs_dict and outputs_dict['base_scene'] is not None:
                final_img = safe_denorm_to_numpy(outputs_dict['base_scene'][idx])
            else:
                # 使用原始图像作为后备
                final_img = safe_denorm_to_numpy(real_A[idx])

            plt.figure(figsize=(10, 8))
            plt.imshow(final_img)
            plt.axis('off')  # 去除坐标轴
            plt.savefig(final_img_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            print(f"✓ 最终图像已生成: {final_img_path}")
        except Exception as e:
            print(f"✗ 生成最终图像失败: {e}")

    # 1. 原始输入图像（无标题）
    try:
        original_img = safe_denorm_to_numpy(real_A[idx])

        plt.figure(figsize=(10, 8))
        plt.imshow(original_img)
        plt.axis('off')  # 去除坐标轴和标题
        plt.savefig(os.path.join(save_dir, f"01_input_image_{step}.png"),
                    dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"✓ 原始图像保存成功")
    except Exception as e:
        print(f"✗ 原始图像保存失败: {e}")

    # 2. 语义分割掩码图（无标题）
    try:
        sem_map = semantic_A[idx].cpu().numpy()

        # 创建彩色语义图
        colors = [
            [0, 0, 0],  # 0: 背景 - 黑色
            [128, 64, 128],  # 1: 道路 - 深紫色
            [244, 35, 232],  # 2: 标志 - 粉色
            [107, 142, 35],  # 3: 植被 - 橄榄绿
            [70, 130, 180],  # 4: 天空 - 钢蓝色
            [220, 20, 60]  # 5: 动态物体 - 红色
        ]

        semantic_color = np.zeros((sem_map.shape[0], sem_map.shape[1], 3), dtype=np.uint8)
        for class_id in range(6):
            mask = sem_map == class_id
            semantic_color[mask] = colors[class_id]

        plt.figure(figsize=(10, 8))
        plt.imshow(semantic_color)
        plt.axis('off')  # 去除坐标轴和标题
        plt.savefig(os.path.join(save_dir, f"02_semantic_mask_{step}.png"),
                    dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"✓ 语义掩码图保存成功")
    except Exception as e:
        print(f"✗ 语义掩码图保存失败: {e}")

    # 3. 基础场景生成结果（无标题）
    try:
        if 'base_scene' in outputs_dict and outputs_dict['base_scene'] is not None:
            base_scene = outputs_dict['base_scene'][idx]
            base_img = safe_denorm_to_numpy(base_scene)

            plt.figure(figsize=(10, 8))
            plt.imshow(base_img)
            plt.axis('off')  # 去除坐标轴和标题
            plt.savefig(os.path.join(save_dir, f"03_base_scene_{step}.png"),
                        dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            print(f"✓ 基础场景保存成功")
    except Exception as e:
        print(f"✗ 基础场景保存失败: {e}")

    # 4. 天气特效可视化（无标题）
    try:
        if 'weather_effect' in outputs_dict and outputs_dict['weather_effect'] is not None:
            weather_effect = outputs_dict['weather_effect'][idx].cpu()

            # 归一化天气特效
            weather_img = (weather_effect - weather_effect.min()) / \
                          (weather_effect.max() - weather_effect.min() + 1e-6)

            # 处理不同维度情况
            if weather_img.dim() == 3 and weather_img.shape[0] in [1, 3]:
                if weather_img.shape[0] == 3:  # RGB
                    weather_img = weather_img.permute(1, 2, 0).numpy()
                else:  # 单通道
                    weather_img = weather_img.squeeze(0).numpy()

            # 确保数据类型正确
            weather_img = np.clip(weather_img, 0, 1)
            if weather_img.dtype != np.uint8:
                weather_img = (weather_img * 255).astype(np.uint8)

            plt.figure(figsize=(10, 8))
            plt.imshow(weather_img, cmap='viridis' if weather_img.ndim == 2 else None)
            plt.axis('off')  # 去除坐标轴和标题
            plt.savefig(os.path.join(save_dir, f"04_weather_effect_{step}.png"),
                        dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            print(f"✓ 天气特效保存成功")
    except Exception as e:
        print(f"✗ 天气特效保存失败: {e}")

    # 5. 最终天气效果（无标题）- 确保一定会生成
    ensure_final_output()

    # 6. 简化组合对比图（无标题）
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 第一行：输入和语义
        axes[0, 0].imshow(safe_convert_to_uint8(safe_denorm_to_numpy(real_A[idx])))
        axes[0, 0].axis('off')

        try:
            sem_map_plot = semantic_A[idx].cpu().numpy()
            semantic_color_plot = np.zeros((sem_map_plot.shape[0], sem_map_plot.shape[1], 3), dtype=np.uint8)
            colors_plot = [[0, 0, 0], [128, 64, 128], [244, 35, 232], [107, 142, 35], [70, 130, 180], [220, 20, 60]]
            for class_id in range(6):
                mask = sem_map_plot == class_id
                semantic_color_plot[mask] = colors_plot[class_id]
            axes[0, 1].imshow(semantic_color_plot)
        except:
            axes[0, 1].axis('off')
        axes[0, 1].axis('off')

        # 第二行：基础场景和最终结果
        if 'base_scene' in outputs_dict and outputs_dict['base_scene'] is not None:
            base_img_plot = safe_denorm_to_numpy(outputs_dict['base_scene'][idx])
            axes[1, 0].imshow(base_img_plot)
        axes[1, 0].axis('off')

        # 确保最终图像一定显示
        final_img_path = os.path.join(save_dir, f"05_final_result_{step}.png")
        if os.path.exists(final_img_path):
            final_img_plot = plt.imread(final_img_path)
            axes[1, 1].imshow(final_img_plot)
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"06_comparison_grid_{step}.png"),
                    dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"✓ 组合对比图保存成功")
    except Exception as e:
        print(f"✗ 组合对比图保存失败: {e}")

    print(f"所有图表已保存到: {save_dir}")


def create_workflow_diagram(save_path):
    """
    创建工作流程图 - 无标题版本
    """
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # 设置坐标轴
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4)
        ax.axis('off')

        # 定义框的位置和大小
        boxes = [
            (1, 3, 1.5, 0.6, "Input\nImage", "#4CAF50"),
            (3, 3, 1.5, 0.6, "Semantic\nEncoder", "#2196F3"),
            (5, 3, 1.5, 0.6, "Base Scene\nGenerator", "#FF9800"),
            (7, 3, 1.5, 0.6, "Weather\nLayer", "#F44336"),
            (9, 3, 1.5, 0.6, "Final\nOutput", "#9C27B0"),
        ]

        # 绘制框图
        for x, y, w, h, text, color in boxes:
            rect = plt.Rectangle((x - w / 2, y - h / 2), w, h,
                                 facecolor=color, alpha=0.7,
                                 edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            ax.text(x, y, text, ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white')

        # 绘制箭头
        arrows = [
            (1.75, 3, 3.25, 3),  # Input -> Semantic Encoder
            (4.5, 3, 5.5, 3),  # Semantic Encoder -> Base Generator
            (6.5, 3, 7.5, 3),  # Base Generator -> Weather Layer
            (8.5, 3, 9.25, 3),  # Weather Layer -> Final Output
        ]

        for x1, y1, x2, y2 in arrows:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->',
                                        color='black',
                                        lw=1,
                                        shrinkA=5, shrinkB=5))

        # 无标题
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"✓ 工作流程图表已保存到: {save_path}")
    except Exception as e:
        print(f"✗ 工作流程图保存失败: {e}")


