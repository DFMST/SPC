import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ==================== 辅助工具函数 ====================

def dark_channel(x, patch_size=15):
    """
    计算图像的暗通道
    x: (B,3,H,W) 值域 [0,1]
    patch_size: 窗口大小
    返回: (B,1,H,W) 暗通道图
    """
    # 取每个像素三个通道的最小值
    min_rgb = torch.min(x, dim=1, keepdim=True)[0]  # (B,1,H,W)
    # 在patch内取最小值（相当于最小值池化）
    padding = patch_size // 2
    dark = -F.max_pool2d(-min_rgb, kernel_size=patch_size, stride=1, padding=padding)
    return dark


def compute_edge_weights(label, gamma=5.0):
    """
    从语义标签计算边界权重
    label: (B,H,W) 整数标签 (0~5)
    gamma: 边界放大系数
    返回: (B,1,H,W) 边界权重图
    """
    # 使用Sobel算子检测标签的边缘
    sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32).view(1, 1, 3, 3)
    device = label.device
    sobel_x = sobel_x.to(device)
    sobel_y = sobel_y.to(device)

    # 将标签转为one-hot或直接用浮点数表示类别（这里直接对标签图求梯度）
    label_float = label.float().unsqueeze(1)  # (B,1,H,W)
    grad_x = F.conv2d(label_float, sobel_x, padding=1)
    grad_y = F.conv2d(label_float, sobel_y, padding=1)
    edge_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)  # (B,1,H,W)
    # 二值化边缘（阈值>0表示有边缘）
    edge_mask = (edge_mag > 1e-4).float()
    edge_weights = 1 + gamma * edge_mask
    return edge_weights


# ==================== 1. 对抗损失 ====================

class GANLoss(nn.Module):
    """
    对抗损失，使用LSGAN (Least Squares GAN)
    真实标签: 1.0, 生成标签: 0.0
    """
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)


# ==================== 2. 语义一致性损失 ====================

class SemanticConsistencyLoss(nn.Module):
    """
    边界感知交叉熵损失
    需要从F_sem解码出分割预测（6类）
    """
    def __init__(self, num_classes=6, gamma=5.0):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        # 语义解码器：将F_sem (128,32,32) 映射到分割预测 (num_classes,32,32)
        self.sem_decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1, bias=True)
        )

    def forward(self, F_sem, label):
        """
        F_sem: (B,128,32,32)
        label: (B, H, W) 整数标签，可能为原始尺寸 256x256
        返回: 标量损失
        """
        # 如果标签空间尺寸与 F_sem 不一致，则下采样到 32x32
        if label.shape[-2:] != F_sem.shape[-2:]:
            # 使用最近邻插值保持离散标签值
            label = F.interpolate(label.float().unsqueeze(1), size=F_sem.shape[-2:], mode='nearest').squeeze(1).long()

        # 解码得到分割logits
        logits = self.sem_decoder(F_sem)  # (B,num_classes,32,32)
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(logits, label, reduction='none')  # (B,32,32)
        # 计算边界权重
        edge_weights = compute_edge_weights(label, gamma=self.gamma)  # (B,1,32,32)
        edge_weights = edge_weights.squeeze(1)  # (B,32,32)
        # 加权平均
        weighted_loss = (ce_loss * edge_weights).mean()
        return weighted_loss


# ==================== 3. 物理先验损失 ====================

class PhysicsPriorLoss(nn.Module):
    """
    物理先验损失，根据天气类型调用不同的损失
    """
    def __init__(self, device='cuda'):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, weather_type, phys_info, I_scene=None, label=None,
                fake_image=None, pseudo_depth=None):
        """
        weather_type: str
        phys_info: dict 来自物理引擎的输出
        I_scene: (B,3,256,256) 基础场景（某些损失需要，雾天不用）
        label: (B,32,32) 语义标签（雪天需要）
        fake_image: (B,3,256,256) 最终生成的天气图像（雾天必须传入）
        pseudo_depth: (B,1,256,256) 伪深度标签（雾天必须传入）
        """
        if weather_type == 'fog':
            return self._fog_loss(phys_info, fake_image, pseudo_depth)
        elif weather_type == 'rain':
            return self._rain_loss(phys_info)
        elif weather_type == 'snow':
            return self._snow_loss(phys_info, label)
        elif weather_type == 'night':
            return self._night_loss(phys_info)
        else:
            raise ValueError(f"Unknown weather type: {weather_type}")

    def _fog_loss(self, info, fake_image, pseudo_depth):
        """
        雾天物理先验损失（修正版）:
        - 深度辅助损失 L_depth: L1 between predicted depth (from physics engine) and pseudo depth
        - 暗通道约束损失 L_dark: L1 between dark channel of generated fog image and atmospheric light A's luminance
        """
        if fake_image is None or pseudo_depth is None:
            raise ValueError("Fog loss requires fake_image and pseudo_depth.")

        depth = info['depth']           # (B,1,256,256) 物理引擎输出的深度图
        A = info['A']                   # (B,3,1,1)     大气光 RGB 值

        # === 1. 深度辅助损失 ===
        # 论文：L_depth = ||D_pred - D_pseudo||_1
        # 这里 D_pred 来自物理引擎的 depth，D_pseudo 是 MiDaS 伪标签
        loss_depth = self.l1_loss(depth, pseudo_depth)

        # === 2. 暗通道约束损失 ===
        # 论文：L_dark = ||J_dark - A_lum||_1
        # J_dark: 生成雾图的暗通道
        # A_lum: 大气光的亮度（取RGB均值作为亮度）
        dark = dark_channel(fake_image, patch_size=15)   # (B,1,256,256)
        # 计算大气光亮度：A 的形状 (B,3,1,1)，取三个通道均值 -> (B,1,1,1)
        A_lum = A.mean(dim=1, keepdim=True)              # (B,1,1,1)
        # 将 A_lum 扩展到与 dark 相同尺寸
        A_lum_expanded = A_lum.expand_as(dark)           # (B,1,256,256)
        loss_dark = self.l1_loss(dark, A_lum_expanded)

        # 两个损失的权重可调节，这里按论文默认比例（各0.5）
        lambda_depth = 0.5
        lambda_dark = 0.5
        return lambda_depth * loss_depth + lambda_dark * loss_dark

    def _rain_loss(self, info):
        """
        雨天物理先验损失:
        - 稀疏性损失 L_sparse: 雨效果层中非零像素的比例
        """
        M_rain = info['M_rain']  # (B,1,256,256)
        # 稀疏性：鼓励M_rain大部分为零
        sparsity = torch.mean(torch.abs(M_rain))  # L1范数
        return sparsity * 0.1

    def _snow_loss(self, info, label):
        """
        雪天物理先验损失:
        - 垂直表面抑制损失 L_vertical: 在垂直表面区域对积雪厚度图施加L1惩罚
        """
        S_ground = info['S_ground']  # (B,1,256,256)
        if label is not None:
            # 从语义标签中提取垂直表面区域（假设类别2为建筑物/墙壁等垂直表面）
            vertical_mask = (label == 2).float().unsqueeze(1)  # (B,1,32,32)
            vertical_mask = F.interpolate(vertical_mask, size=(256, 256), mode='nearest')
            loss_vertical = (S_ground * vertical_mask).mean() * 0.1
        else:
            loss_vertical = torch.tensor(0.0, device=S_ground.device)
        return loss_vertical

    def _night_loss(self, info):
        """
        夜晚物理先验损失:
        - 光源稀疏性损失 L_sparse: L1 on M_light
        - 亮度平滑性损失 L_smooth: gradient norm on L
        """
        M_light = info['M_light']  # (B,1,256,256)
        L = info['L']  # (B,1,256,256)
        loss_sparse = torch.mean(torch.abs(M_light)) * 0.05
        loss_smooth = self._tv_loss(L) * 0.05
        return loss_sparse + loss_smooth

    def _tv_loss(self, x):
        """总变差损失，用于平滑"""
        batch_size = x.size(0)
        h_x = x.size(2)
        w_x = x.size(3)
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


# ==================== 4. 感知损失 ====================

class PerceptualLoss(nn.Module):
    """
    感知损失，使用预训练VGG19的特征
    提取 relu1_2, relu2_2, relu3_3, relu4_2 层的特征
    """
    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_2']):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.layers = layers
        self.layer_mapping = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 17,
            'relu4_2': 26
        }
        # 提取所需层
        self.slices = nn.ModuleList()
        prev_idx = 0
        for layer_name in layers:
            idx = self.layer_mapping[layer_name]
            self.slices.append(nn.Sequential(*vgg[prev_idx:idx+1]))
            prev_idx = idx + 1
        # 冻结参数
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        """
        input, target: (B,3,256,256) 值域应为[-1,1]或[0,1]（VGG期望[0,1]）
        注意：VGG要求输入归一化到ImageNet均值方差，这里假设输入已在[0,1]范围
        """
        # 如果输入在[-1,1]，转换到[0,1]
        if input.min() < 0:
            input = (input + 1) / 2
            target = (target + 1) / 2
        loss = 0.0
        x = input
        y = target
        for slice in self.slices:
            x = slice(x)
            y = slice(y)
            loss += F.l1_loss(x, y)
        return loss


# ==================== 5. 总损失计算器 ====================

class SPCLossCalculator(nn.Module):
    """
    整合所有损失，支持动态权重调度
    """
    def __init__(self, weather_type, num_classes=6):
        super().__init__()
        self.weather_type = weather_type
        self.gan_loss = GANLoss()
        self.sem_loss = SemanticConsistencyLoss(num_classes=num_classes)
        self.phys_loss = PhysicsPriorLoss()
        self.perc_loss = PerceptualLoss()

        # 损失权重（默认值，训练时会根据epoch动态调整）
        self.lambda_adv = 1.0
        self.lambda_sem = 2.0   # 初始值，线性衰减到1.0
        self.lambda_phys = 0.1  # 初始值，线性增长到0.2
        self.lambda_perc = 0.3  # 雾天0.3，雨天0.5，雪天0.4，夜晚0.5

        # 设置感知损失权重
        perc_weights = {'fog': 0.3, 'rain': 0.5, 'snow': 0.4, 'night': 0.5}
        self.lambda_perc = perc_weights.get(weather_type, 0.3)

    def set_stage(self, stage_config):
        """
        设置当前阶段的损失权重
        stage_config: dict with keys 'adv', 'perc', 'sem', 'phys'
        """
        self.lambda_adv = stage_config.get('adv', 1.0)
        self.lambda_perc = stage_config.get('perc', 0.0)
        self.lambda_sem = stage_config.get('sem', 0.0)
        self.lambda_phys = stage_config.get('phys', 0.0)

    def update_weights(self, epoch, total_epochs):
        """
        动态权重调度
        lambda_sem: 2.0 -> 1.0 线性衰减
        lambda_phys: 0.1 -> 0.2 线性增长
        """
        progress = epoch / total_epochs
        self.lambda_sem = 2.0 - 1.0 * progress
        self.lambda_phys = 0.1 + 0.1 * progress

    def forward(self, generator_output, discriminator_output, input_image,
                real_image, label, epoch, total_epochs, pseudo_depth=None):
        """
        generator_output: tuple (fake_image, info_dict)
        discriminator_output: list of multi-scale predictions for fake images
        input_image: (B,3,256,256) 输入晴天图像（用于感知损失）
        real_image: (B,3,256,256) 真实天气图像（当前未使用，保留接口）
        label: (B,32,32) 语义标签
        pseudo_depth: (B,1,256,256) 伪深度标签（雾天必需，其他天气可传None）
        """
        fake_image, info = generator_output

        # 1. 对抗损失（始终计算）
        loss_adv = 0
        for pred in discriminator_output:
            loss_adv += self.gan_loss(pred, True)
        loss_adv = loss_adv * self.lambda_adv

        # 2. 感知损失（如果权重大于0）
        if self.lambda_perc > 0:
            loss_perc = self.perc_loss(fake_image, input_image) * self.lambda_perc
        else:
            loss_perc = torch.tensor(0.0, device=fake_image.device)

        # 3. 语义一致性损失（如果权重大于0）
        if self.lambda_sem > 0:
            F_sem = info['F_sem']
            loss_sem = self.sem_loss(F_sem, label) * self.lambda_sem
        else:
            loss_sem = torch.tensor(0.0, device=fake_image.device)

        # 4. 物理先验损失（如果权重大于0）
        if self.lambda_phys > 0:
            phys_info = info['phys_info']
            I_scene = info['I_scene']
            # 传入 fake_image 和 pseudo_depth（雾天需要）
            loss_phys = self.phys_loss(
                self.weather_type, phys_info, I_scene, label,
                fake_image=fake_image, pseudo_depth=pseudo_depth
            ) * self.lambda_phys
        else:
            loss_phys = torch.tensor(0.0, device=fake_image.device)

        total_loss = loss_adv + loss_perc + loss_sem + loss_phys
        return total_loss, {
            'loss_adv': loss_adv.item(),
            'loss_perc': loss_perc.item(),
            'loss_sem': loss_sem.item(),
            'loss_phys': loss_phys.item()
        }