import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ==================== 1. ChannelAttention ====================
class ChannelAttention(nn.Module):
    """
    通道注意力模块（类似SE Block）
    输入: (B, C, H, W)
    输出: (B, C, H, W) —— 与输入逐元素相乘后的结果
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# ==================== 2. DetailEnhanceBlock ====================
class DetailEnhanceBlock(nn.Module):
    """
    细节增强模块
    输入: (B, C, H, W)
    输出: (B, C, H, W)
    结构: Conv(C→C/2) → LeakyReLU → Conv(C/2→C) → ChannelAttention → Sigmoid
          然后与输入相乘: x * (1 + branch(x))
    """
    def __init__(self, channels):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1, bias=False),
            ChannelAttention(channels, reduction=16),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * (1 + self.branch(x))


# ==================== 3. EdgeAwareBlock ====================
class EdgeAwareBlock(nn.Module):
    """
    边缘感知模块
    输入: (B, C, H, W)
    输出: (B, C, H, W)
    使用固定的Sobel滤波器计算梯度幅度，然后通过小型CNN生成边缘增强权重
    """
    def __init__(self, channels):
        super().__init__()
        # 固定Sobel核（不可学习）
        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # 小型CNN生成边缘权重
        self.weight_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 计算灰度图上的梯度幅度（对每个通道独立计算，然后取平均）
        gray = x.mean(dim=1, keepdim=True)  # (B,1,H,W)
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)  # (B,1,H,W)

        # 通过CNN得到边缘增强权重
        edge_weight = self.weight_net(magnitude)  # (B,1,H,W)
        return x * (1 + edge_weight)


# ==================== 4. ResidualBlock ====================
class ResidualBlock(nn.Module):
    """
    残差块
    输入: (B, C, H, W)
    输出: (B, C, H, W)
    结构: ReflectionPad → Conv → InstanceNorm → ReLU → ReflectionPad → Conv → InstanceNorm → 残差连接
    可选: use_norm=False时跳过InstanceNorm
    """
    def __init__(self, in_features, use_norm=True):
        super().__init__()
        self.use_norm = use_norm
        layers = []
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, kernel_size=3, bias=False),
        ]
        if use_norm:
            layers.append(nn.InstanceNorm2d(in_features))
        layers.append(nn.ReLU(inplace=True))

        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, kernel_size=3, bias=False),
        ]
        if use_norm:
            layers.append(nn.InstanceNorm2d(in_features))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)


# ==================== 5. UpBlock ====================
class UpBlock(nn.Module):
    """
    上采样块（亚像素卷积 / PixelShuffle）
    输入: (B, C_in, H, W)
    输出: (B, C_out, H*scale, W*scale)
    结构: Conv(C_in → C_out * scale^2, 3, padding=1) → PixelShuffle → InstanceNorm → ReLU
    """
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2),
                              kernel_size=3, padding=1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


# ==================== 6. NoiseSuppression ====================
class NoiseSuppression(nn.Module):
    """
    噪声抑制模块
    输入: (B, C, H, W)
    输出: (B, C, H, W)
    结构: Conv(C→C,3,padding=1) → InstanceNorm → LeakyReLU → Conv(C→C,1)
          然后 signal + noise * 0.3
    """
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        )

    def forward(self, x):
        residual = self.net(x)
        return x + residual * 0.3

# ==================== 结构化语义编码器 (Sec 3.1) ====================
class StructuredSemanticEncoder(nn.Module):
    """
    结构化语义编码器
    输入: I_in (B,3,256,256), enable_aux (bool)
    输出: F_sem (B,128,32,32)
          若 enable_aux=True，额外返回 aux_dict = {
              'depth': (B,1,32,32),
              'normal': (B,3,32,32),
              'boundary': (B,1,32,32)
          }
    """
    def __init__(self, output_dim=128, pretrained=True):
        super().__init__()
        # ---------- 骨干网络：ResNet-50 前三层 ----------
        resnet = models.resnet50(pretrained=pretrained)
        # conv1: 输入3->64, kernel=7, stride=2, padding=3 -> 输出 (B,64,128,128)
        self.conv1 = nn.Sequential(
            resnet.conv1,   # Conv2d(3,64,7,2,3)
            resnet.bn1,     # BatchNorm2d(64)
            resnet.relu     # ReLU
        )
        self.maxpool = resnet.maxpool  # MaxPool2d(kernel_size=3, stride=2, padding=1) -> (B,64,64,64)
        # layer1: 输出 (B,256,64,64)
        self.layer1 = resnet.layer1
        # layer2: 输出 (B,512,32,32)
        self.layer2 = resnet.layer2
        # layer3: 输出 (B,1024,16,16)
        self.layer3 = resnet.layer3

        # ---------- 特征融合模块 ----------
        # 拼接后通道数: 256(layer1) + 512(layer2) + 1024(layer3) = 1792
        self.fusion = nn.Sequential(
            nn.Conv2d(256 + 512 + 1024, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, output_dim, kernel_size=1, bias=False)
        )

        # ---------- 通道注意力重加权 ----------
        self.channel_attention = ChannelAttention(output_dim, reduction=16)

        # ---------- 辅助预测头（仅在训练时启用） ----------
        # 深度头：输出1通道，Sigmoid归一化到(0,1)
        self.aux_depth_head = nn.Sequential(
            nn.Conv2d(output_dim, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        # 法线头：输出3通道，Tanh归一化到[-1,1]
        self.aux_normal_head = nn.Sequential(
            nn.Conv2d(output_dim, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1, bias=True),
            nn.Tanh()
        )
        # 边界头：输出1通道，Sigmoid
        self.aux_boundary_head = nn.Sequential(
            nn.Conv2d(output_dim, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x, enable_aux=False):
        """
        x: (B,3,256,256)
        enable_aux: bool，训练时为True，推理时为False
        返回:
            F_sem: (B,128,32,32)
            aux_dict: dict 或 None
        """
        # ---- 1. 骨干特征提取 ----
        x1 = self.conv1(x)          # (B,64,128,128)
        x2 = self.maxpool(x1)       # (B,64,64,64)
        x3 = self.layer1(x2)        # (B,256,64,64)
        x4 = self.layer2(x3)        # (B,512,32,32)
        x5 = self.layer3(x4)        # (B,1024,16,16)

        # ---- 2. 上采样到统一尺寸 32x32 ----
        x3_up = F.interpolate(x3, size=(32, 32), mode='bilinear', align_corners=False)   # (B,256,32,32)
        x5_up = F.interpolate(x5, size=(32, 32), mode='bilinear', align_corners=False)   # (B,1024,32,32)

        # ---- 3. 拼接并融合 ----
        fused = torch.cat([x3_up, x4, x5_up], dim=1)  # (B,1792,32,32)
        F_sem_raw = self.fusion(fused)                 # (B,128,32,32)

        # ---- 4. 通道注意力重加权 ----
        F_sem = self.channel_attention(F_sem_raw)      # (B,128,32,32)

        # ---- 5. 辅助预测（仅训练时） ----
        if enable_aux:
            aux_depth = self.aux_depth_head(F_sem)      # (B,1,32,32)
            aux_normal = self.aux_normal_head(F_sem)    # (B,3,32,32)
            aux_boundary = self.aux_boundary_head(F_sem) # (B,1,32,32)
            aux_dict = {
                'depth': aux_depth,
                'normal': aux_normal,
                'boundary': aux_boundary
            }
            return F_sem, aux_dict
        else:
            return F_sem, None

# ==================== 基础场景生成器 (Sec 3.2) ====================
class BaseSceneGenerator(nn.Module):
    """
    基础场景生成器
    输入: I_in (B,3,256,256), F_sem (B,128,32,32)
    输出: I_scene (B,3,256,256) — 初步保持结构的基础天气场景
    """
    def __init__(self):
        super().__init__()

        # ---------- 天气特征适配器 ----------
        # 将F_sem压缩为天气条件向量 weather_vec (B,256,1,1)
        self.weather_adapter = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        # ---------- 编码器 ----------
        # Initial: 反射填充 + 卷积 + InstanceNorm + ReLU
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Down1: 卷积 stride=2
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # Down1后条件注入: 拼接weather_vec并降维
        self.cond_inject_down1 = nn.Sequential(
            nn.Conv2d(128 + 256, 128, kernel_size=1, bias=False)
        )

        # Down2
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.cond_inject_down2 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=1, bias=False)
        )

        # Down3
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.cond_inject_down3 = nn.Sequential(
            nn.Conv2d(512 + 256, 512, kernel_size=1, bias=False)
        )

        # ---------- 残差块 (9个) ----------
        self.resblocks = nn.Sequential(*[
            ResidualBlock(512, use_norm=True) for _ in range(3)
        ])

        # ---------- 解码器 ----------
        # Up1: 转置卷积 512→256, stride=2, output_padding=1
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2,
                               output_padding=1, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # Up1后条件注入: 拼接weather_vec并降维
        self.cond_inject_up1 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=1, bias=False)
        )

        # Up2: 256→128
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2,
                               output_padding=1, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.cond_inject_up2 = nn.Sequential(
            nn.Conv2d(128 + 256, 128, kernel_size=1, bias=False)
        )

        # Up3: 128→64
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,
                               output_padding=1, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.cond_inject_up3 = nn.Sequential(
            nn.Conv2d(64 + 256, 64, kernel_size=1, bias=False)
        )

        # ---------- 输出层 ----------
        self.out_layer = nn.Sequential(
            DetailEnhanceBlock(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            NoiseSuppression(64),
            nn.Conv2d(64, 3, kernel_size=3, padding=1, bias=True),
            nn.Tanh()   # 输出值域 [-1,1]
        )

    def forward(self, I_in, F_sem):
        """
        I_in: (B,3,256,256)
        F_sem: (B,128,32,32)
        Returns: I_scene (B,3,256,256)
        """
        # ---- 1. 天气特征适配 ----
        weather_vec = self.weather_adapter(F_sem)  # (B,256,1,1)

        # ---- 2. 编码器 ----
        # Initial
        x = self.initial(I_in)                     # (B,64,256,256)
        skip_init = x                              # 保存用于跳跃连接

        # Down1
        x = self.down1(x)                          # (B,128,128,128)
        # 条件注入: 扩展weather_vec到空间尺寸128x128
        w = weather_vec.expand(-1, -1, 128, 128)   # (B,256,128,128)
        x = torch.cat([x, w], dim=1)               # (B,128+256,128,128)
        x = self.cond_inject_down1(x)              # (B,128,128,128)
        skip_down1 = x                             # 保存用于跳跃连接

        # Down2
        x = self.down2(x)                          # (B,256,64,64)
        w = weather_vec.expand(-1, -1, 64, 64)
        x = torch.cat([x, w], dim=1)               # (B,256+256,64,64)
        x = self.cond_inject_down2(x)              # (B,256,64,64)
        skip_down2 = x                             # 保存用于跳跃连接

        # Down3
        x = self.down3(x)                          # (B,512,32,32)
        w = weather_vec.expand(-1, -1, 32, 32)
        x = torch.cat([x, w], dim=1)               # (B,512+256,32,32)
        x = self.cond_inject_down3(x)              # (B,512,32,32)

        # ---- 3. 残差块 ----
        x = self.resblocks(x)                      # (B,512,32,32)

        # ---- 4. 解码器 ----
        # Up1
        x = self.up1(x)                            # (B,256,64,64)
        x = x + skip_down2                         # 跳跃连接相加
        w = weather_vec.expand(-1, -1, 64, 64)
        x = torch.cat([x, w], dim=1)               # (B,256+256,64,64)
        x = self.cond_inject_up1(x)                # (B,256,64,64)

        # Up2
        x = self.up2(x)                            # (B,128,128,128)
        x = x + skip_down1                         # 跳跃连接相加
        w = weather_vec.expand(-1, -1, 128, 128)
        x = torch.cat([x, w], dim=1)               # (B,128+256,128,128)
        x = self.cond_inject_up2(x)                # (B,128,128,128)

        # Up3
        x = self.up3(x)                            # (B,64,256,256)
        x = x + skip_init                          # 跳跃连接相加
        w = weather_vec.expand(-1, -1, 256, 256)
        x = torch.cat([x, w], dim=1)               # (B,64+256,256,256)
        x = self.cond_inject_up3(x)                # (B,64,256,256)

        # ---- 5. 输出层 ----
        I_scene = self.out_layer(x)                # (B,3,256,256)

        return I_scene

# ==================== 可微物理引擎 (Sec 3.3) ====================
class DifferentiablePhysicsEngine(nn.Module):
    """
    输入: I_scene (B,3,256,256), F_sem (B,128,32,32), weather_type str
    输出: I_effect (B,3,256,256) — 纯天气效果层
          附加信息: 用于计算物理先验损失的中间变量
    """
    def __init__(self):
        super().__init__()
        self.fog_branch = FogBranch()
        self.rain_branch = RainBranch()
        self.snow_branch = SnowBranch()
        self.night_branch = NightBranch()

    def forward(self, I_scene, F_sem, weather_type):
        if weather_type == 'fog':
            return self.fog_branch(I_scene, F_sem)
        elif weather_type == 'rain':
            return self.rain_branch(I_scene, F_sem)
        elif weather_type == 'snow':
            return self.snow_branch(I_scene, F_sem)
        elif weather_type == 'night':
            return self.night_branch(I_scene, F_sem)
        else:
            raise ValueError(f"Unknown weather type: {weather_type}")

# --- 各天气分支的具体实现（仅给出接口，细节需按论文公式实现）---
class FogBranch(nn.Module):
    """
    雾天分支
    输入: I_scene (B,3,256,256), F_sem (B,128,32,32)
    输出: I_effect (B,3,256,256)
          phys_info = {
              'depth': (B,1,256,256),
              'beta': (B,1,256,256),
              't': (B,1,256,256),
              'A': (B,3,1,1)
          }
    """
    def __init__(self):
        super().__init__()

        # ---------- 深度估计网络 (depth_predictor) ----------
        # 输入: I_scene (B,3,256,256)
        # 5层卷积，通道变化: 3→64→128→256→128→1，带跳跃连接
        self.depth_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.depth_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.depth_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.depth_conv4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.depth_conv5 = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()   # 深度值归一化到 (0,1)
        )
        # 跳跃连接: 将conv2的输出与conv4的输出相加
        self.skip_conv = nn.Conv2d(128, 128, kernel_size=1, bias=False)  # 可选，用于对齐通道

        # ---------- 散射系数预测网络 (beta_predictor) ----------
        # 输入: F_sem (B,128,32,32)，先上采样到256x256
        self.beta_conv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.beta_conv2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, bias=True),
            nn.Sigmoid()   # I_fog ∈ (0,1)
        )
        # 可学习参数: beta_min 和 alpha
        self.beta_min = nn.Parameter(torch.tensor(0.03))
        self.alpha = nn.Parameter(torch.tensor(0.25))

        # ---------- 大气光预测网络 (atmospheric_light) ----------
        # 输入: F_sem (B,128,32,32)
        self.atmos_pool = nn.AdaptiveAvgPool2d(1)
        self.atmos_fc = nn.Sequential(
            nn.Linear(128, 3, bias=True),
            nn.Sigmoid()
        )

    def forward(self, I_scene, F_sem):
        """
        I_scene: (B,3,256,256)
        F_sem: (B,128,32,32)
        Returns: I_effect (B,3,256,256), phys_info (dict)
        """
        # ---- 1. 深度估计 ----
        d1 = self.depth_conv1(I_scene)   # (B,64,256,256)
        d2 = self.depth_conv2(d1)        # (B,128,256,256)
        d3 = self.depth_conv3(d2)        # (B,256,256,256)
        d4 = self.depth_conv4(d3)        # (B,128,256,256)
        # 跳跃连接: d2 与 d4 相加（通道数均为128，无需特殊处理）
        d4 = d4 + d2                     # 跳跃连接
        depth = self.depth_conv5(d4)     # (B,1,256,256)

        # ---- 2. 散射系数预测 ----
        # 上采样 F_sem 到 256x256
        F_sem_up = F.interpolate(F_sem, size=(256, 256), mode='bilinear', align_corners=False)  # (B,128,256,256)
        I_fog = self.beta_conv1(F_sem_up)   # (B,64,256,256)
        I_fog = self.beta_conv2(I_fog)      # (B,1,256,256)
        beta = self.beta_min + self.alpha * I_fog  # (B,1,256,256)

        # ---- 3. 大气光预测 ----
        pooled = self.atmos_pool(F_sem).view(F_sem.size(0), -1)  # (B,128)
        A = self.atmos_fc(pooled).view(-1, 3, 1, 1)             # (B,3,1,1)

        # ---- 4. 透射率 ----
        t = torch.exp(-beta * depth)   # (B,1,256,256)

        # ---- 5. 雾效果层 ----
        # I_effect = (A - I_scene) * (1 - t)
        I_effect = (A - I_scene) * (1 - t)   # 广播机制自动匹配尺寸

        # 组装 phys_info
        phys_info = {
            'depth': depth,
            'beta': beta,
            't': t,
            'A': A
        }

        return I_effect, phys_info

class RainBranch(nn.Module):
    """
    雨天分支
    输入: I_scene (B,3,256,256), F_sem (B,128,32,32)
    输出: I_effect (B,3,256,256)
          phys_info = {
              'M_rain': (B,1,256,256),
              'theta': (B,),
              'alpha': (B,1,1,1)
          }
    """
    def __init__(self):
        super().__init__()

        # ---------- 雨滴密度图预测网络 (density_predictor) ----------
        # 输入: F_sem (B,128,32,32)，先上采样到256x256
        self.density_conv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.density_conv2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, bias=True),
            nn.Sigmoid()   # M_rain ∈ (0,1)
        )

        # ---------- 运动方向预测 (angle_predictor) ----------
        # 输入: F_sem (B,128,32,32) -> 全局平均池化 -> FC -> 乘以π
        self.angle_pool = nn.AdaptiveAvgPool2d(1)
        self.angle_fc = nn.Sequential(
            nn.Linear(128, 1, bias=True),
            nn.Sigmoid()   # 输出在(0,1)，乘以π得到(0,π)
        )

        # ---------- 模糊核软选择 ----------
        # 预定义8个方向的运动模糊核 (15x15)
        # 方向角度: 0°, 22.5°, 45°, ..., 157.5°
        self.register_buffer('kernel_bank', self._generate_kernel_bank())

        # ---------- 透明度预测 (alpha_predictor) ----------
        # 输入: F_sem (B,128,32,32) -> 全局平均池化 -> FC -> Sigmoid
        self.alpha_pool = nn.AdaptiveAvgPool2d(1)
        self.alpha_fc = nn.Sequential(
            nn.Linear(128, 1, bias=True),
            nn.Sigmoid()
        )

    def _generate_kernel_bank(self, kernel_size=15):
        """
        生成8个方向的运动模糊核，每个核大小为 (1, kernel_size, kernel_size)
        方向: 0°, 22.5°, 45°, 67.5°, 90°, 112.5°, 135°, 157.5°
        返回: (8, 1, kernel_size, kernel_size)
        """
        num_kernels = 8
        angles = torch.linspace(0, 157.5, num_kernels) * (torch.pi / 180.0)  # 弧度
        kernels = []
        center = kernel_size // 2
        for angle in angles:
            kernel = torch.zeros((kernel_size, kernel_size))
            cos_a = torch.cos(angle).item()
            sin_a = torch.sin(angle).item()
            # 沿运动方向画一条线，长度约为 kernel_size
            for i in range(kernel_size):
                offset = i - center
                x = int(round(center + offset * cos_a))
                y = int(round(center + offset * sin_a))
                if 0 <= x < kernel_size and 0 <= y < kernel_size:
                    kernel[y, x] = 1.0
            # 归一化使和为1
            kernel_sum = kernel.sum()
            if kernel_sum > 0:
                kernel /= kernel_sum
            kernels.append(kernel.unsqueeze(0))  # (1, K, K)
        bank = torch.stack(kernels, dim=0)  # (8, 1, K, K)
        return bank

    def soft_select_kernel(self, theta):
        """
        根据角度theta软选择模糊核
        theta: (B,) 弧度值，范围[0, π)
        返回: (B, 1, K, K) 加权求和后的模糊核
        """
        B = theta.shape[0]
        K = self.kernel_bank.shape[-1]
        device = theta.device

        # 将theta归一化到[0, 8)区间，对应8个核的索引
        indices = theta / (torch.pi) * 8.0  # (B,)
        # 对每个样本，计算与8个核的权重（使用高斯距离）
        # 生成8个锚点角度
        anchor_angles = torch.linspace(0, 157.5, 8, device=device) * (torch.pi / 180.0)  # (8,)
        # 计算角度差，考虑圆周对称性（雨方向无方向性，θ和θ+π等价，但这里θ∈[0,π)）
        diff = indices.unsqueeze(1) - torch.arange(8, device=device).float().unsqueeze(0)  # (B,8)
        # 使用softmax生成权重，温度参数设为1
        weights = F.softmax(-diff.abs() * 2.0, dim=1)  # (B,8)
        # 加权求和得到每个样本的模糊核
        # kernel_bank: (8,1,K,K) -> (1,8,1,K,K)
        bank = self.kernel_bank.unsqueeze(0)  # (1,8,1,K,K)
        weights = weights.view(B, 8, 1, 1, 1)  # (B,8,1,1,1)
        combined_kernel = (bank * weights).sum(dim=1)  # (B,1,K,K)
        return combined_kernel

    def forward(self, I_scene, F_sem):
        """
        I_scene: (B,3,256,256)
        F_sem: (B,128,32,32)
        Returns: I_effect (B,3,256,256), phys_info (dict)
        """
        # ---- 1. 雨滴密度图 M_rain ----
        F_sem_up = F.interpolate(F_sem, size=(256, 256), mode='bilinear', align_corners=False)  # (B,128,256,256)
        M_rain = self.density_conv1(F_sem_up)   # (B,64,256,256)
        M_rain = self.density_conv2(M_rain)     # (B,1,256,256)

        # ---- 2. 运动方向 theta ----
        pooled = self.angle_pool(F_sem).view(F_sem.size(0), -1)  # (B,128)
        theta = self.angle_fc(pooled).squeeze(-1) * torch.pi      # (B,)，值域[0,π)

        # ---- 3. 模糊核软选择 ----
        K_motion = self.soft_select_kernel(theta)  # (B,1,15,15)

        # ---- 4. 透明度 alpha ----
        pooled_alpha = self.alpha_pool(F_sem).view(F_sem.size(0), -1)  # (B,128)
        alpha = self.alpha_fc(pooled_alpha).view(-1, 1, 1, 1)          # (B,1,1,1)

        # ---- 5. 雨效果层 ----
        # 对M_rain进行分组卷积（每组一个核），padding='same'
        # 由于每个样本的核不同，需要逐样本处理或使用group=B的方式
        B = M_rain.shape[0]
        rain_maps = []
        for i in range(B):
            # M_rain[i]: (1,1,256,256), K_motion[i]: (1,1,15,15)
            rain_map_i = F.conv2d(
                M_rain[i:i+1],           # (1,1,256,256)
                K_motion[i:i+1],         # (1,1,15,15)
                padding=7,               # 15//2 = 7
                groups=1
            )  # (1,1,256,256)
            rain_maps.append(rain_map_i)
        rain_map = torch.cat(rain_maps, dim=0)  # (B,1,256,256)

        # 效果层: I_effect = alpha * rain_map，然后通过tanh映射到[-1,1]
        I_effect = alpha * rain_map               # (B,1,256,256)
        # 扩展到3通道（雨效果通常作用于亮度通道，这里简单复制到RGB）
        I_effect = I_effect.repeat(1, 3, 1, 1)    # (B,3,256,256)
        I_effect = torch.tanh(I_effect)           # 值域[-1,1]

        # 组装phys_info
        phys_info = {
            'M_rain': M_rain,
            'theta': theta,
            'alpha': alpha
        }

        return I_effect, phys_info

class GaussianBlur(nn.Module):
    """
    可微高斯模糊模块
    使用固定高斯核进行卷积，sigma可指定
    """
    def __init__(self, kernel_size=5, sigma=2.0):
        super().__init__()
        self.kernel_size = kernel_size
        # 创建高斯核
        kernel = self._gaussian_kernel(kernel_size, sigma)
        # 注册为buffer（不可学习参数）
        self.register_buffer('kernel', kernel)

    def _gaussian_kernel(self, size, sigma):
        """生成二维高斯核 (1,1,size,size)"""
        coords = torch.arange(size).float() - size // 2
        x = coords.view(1, -1).repeat(size, 1)
        y = coords.view(-1, 1).repeat(1, size)
        g = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        g = g / g.sum()
        return g.view(1, 1, size, size)

    def forward(self, x):
        """
        x: (B, C, H, W)
        返回: (B, C, H, W) 模糊后的图像
        """
        # 对每个通道分别卷积（groups=C）
        padding = self.kernel_size // 2
        return F.conv2d(x, self.kernel.expand(x.size(1), -1, -1, -1),
                        padding=padding, groups=x.size(1))


class SnowBranch(nn.Module):
    """
    雪天分支
    输入: I_scene (B,3,256,256), F_sem (B,128,32,32)
    输出: I_effect (B,3,256,256)
          phys_info = {
              'S_ground': (B,1,256,256),
              'S_flake': (B,1,256,256),
              'gamma': (B,1,1,1)
          }
    """
    def __init__(self):
        super().__init__()

        # ---------- 积雪厚度图预测网络 (ground_snow_predictor) ----------
        # 与M_rain结构相同: 上采样F_sem → Conv(128→64,3) → ReLU → Conv(64→1,1) → Sigmoid
        self.ground_conv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.ground_conv2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, bias=True),
            nn.Sigmoid()   # S_ground ∈ (0,1)
        )

        # ---------- 飘雪层高斯模糊 (固定sigma=2) ----------
        self.gaussian_blur = GaussianBlur(kernel_size=5, sigma=2.0)

        # ---------- 全局强度预测 (intensity_predictor) ----------
        # 从F_sem通过全局平均池化 + FC得到
        self.gamma_pool = nn.AdaptiveAvgPool2d(1)
        self.gamma_fc = nn.Sequential(
            nn.Linear(128, 1, bias=True),
            nn.Sigmoid()   # gamma ∈ (0,1)
        )

    def forward(self, I_scene, F_sem):
        """
        I_scene: (B,3,256,256)
        F_sem: (B,128,32,32)
        Returns: I_effect (B,3,256,256), phys_info (dict)
        """
        # ---- 1. 积雪厚度图 S_ground ----
        F_sem_up = F.interpolate(F_sem, size=(256, 256), mode='bilinear', align_corners=False)  # (B,128,256,256)
        S_ground = self.ground_conv1(F_sem_up)   # (B,64,256,256)
        S_ground = self.ground_conv2(S_ground)   # (B,1,256,256)

        # ---- 2. 飘雪层 S_flake ----
        # 固定种子以保证可复现
        torch.manual_seed(42)
        # 生成均匀分布噪声 U(0,1)
        noise = torch.rand(S_ground.shape, device=S_ground.device)  # (B,1,256,256)
        # 高斯模糊
        noise_blur = self.gaussian_blur(noise)    # (B,1,256,256)
        # 飘雪层: 只在非积雪区域出现
        S_flake = noise_blur * (1 - S_ground)     # (B,1,256,256)

        # ---- 3. 全局强度 gamma ----
        pooled = self.gamma_pool(F_sem).view(F_sem.size(0), -1)  # (B,128)
        gamma = self.gamma_fc(pooled).view(-1, 1, 1, 1)          # (B,1,1,1)

        # ---- 4. 雪效果层 ----
        # I_effect = gamma * (S_flake + S_ground)
        I_effect = gamma * (S_flake + S_ground)   # (B,1,256,256)
        # clip到[0,1]（规格要求）
        I_effect = torch.clamp(I_effect, 0, 1)
        # 扩展到3通道（雪效果同样作用于亮度，复制到RGB）
        I_effect = I_effect.repeat(1, 3, 1, 1)    # (B,3,256,256)

        # 组装phys_info
        phys_info = {
            'S_ground': S_ground,
            'S_flake': S_flake,
            'gamma': gamma
        }

        return I_effect, phys_info

class LearnableGaussianBlur(nn.Module):
    def __init__(self, max_kernel_size=31):
        super().__init__()
        self.max_kernel_size = max_kernel_size

    def forward(self, x, sigma):
        B, C, H, W = x.shape
        device = x.device
        # 确保 sigma 为一维 (B,)
        sigma = sigma.view(-1)
        # 根据所有 sigma 的最大值确定核大小（所有样本共用同一核大小）
        max_sigma = sigma.max().item()
        k = int(2 * torch.ceil(torch.tensor(3 * max_sigma)).item() + 1)
        k = max(3, min(k, self.max_kernel_size))
        if k % 2 == 0:
            k += 1
        half = k // 2
        # 生成坐标网格
        coords = torch.arange(-half, half + 1, device=device).float()
        x_grid = coords.view(1, -1).repeat(k, 1)   # (k, k)
        y_grid = coords.view(-1, 1).repeat(1, k)   # (k, k)
        out_list = []
        for i in range(B):
            s = sigma[i].clamp(min=1e-6)           # 防止除以零
            g = torch.exp(-(x_grid**2 + y_grid**2) / (2 * s**2))
            g = g / g.sum()
            kernel = g.view(1, 1, k, k).repeat(C, 1, 1, 1)  # (C, 1, k, k)
            xi = x[i].unsqueeze(0)                           # (1, C, H, W)
            oi = F.conv2d(xi, kernel, padding=half, groups=C)
            out_list.append(oi)
        return torch.cat(out_list, dim=0)  # (B, C, H, W)


class NightBranch(nn.Module):
    """
    夜间分支
    输入: I_scene (B,3,256,256), F_sem (B,128,32,32)
    输出: I_effect (B,3,256,256)
          phys_info = {
              'L': (B,1,256,256),
              'M_light': (B,1,256,256),
              'E': (B,3,256,256)
          }
    """
    def __init__(self):
        super().__init__()

        # ---------- 亮度衰减因子预测网络 (luminance_predictor) ----------
        # 结构同 S_ground: 上采样 F_sem → Conv(128→64,3) → ReLU → Conv(64→1,1) → Sigmoid
        self.lum_conv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.lum_conv2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, bias=True),
            nn.Sigmoid()   # L' ∈ (0,1)
        )

        # ---------- 光源掩膜预测网络 (light_mask_predictor) ----------
        # 结构同上
        self.mask_conv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.mask_conv2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, bias=True),
            nn.Sigmoid()   # M_light ∈ (0,1)
        )

        # ---------- 光源颜色预测网络 (light_color_predictor) ----------
        # 输出3通道，Sigmoid
        self.color_conv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.color_conv2 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1, bias=True),
            nn.Sigmoid()   # C ∈ (0,1)
        )

        # ---------- 可学习模糊标准差预测 (blur_sigma_predictor) ----------
        # 从 F_sem 通过全局平均池化 + FC 得到，Softplus 确保正数
        self.sigma_pool = nn.AdaptiveAvgPool2d(1)
        self.sigma_fc = nn.Sequential(
            nn.Linear(128, 1, bias=True),
            nn.Softplus()   # sigma > 0
        )

        # ---------- 可学习高斯模糊 ----------
        self.gaussian_blur = LearnableGaussianBlur(max_kernel_size=31)

    def forward(self, I_scene, F_sem):
        """
        I_scene: (B,3,256,256)
        F_sem: (B,128,32,32)
        Returns: I_effect (B,3,256,256), phys_info (dict)
        """
        # ---- 1. 亮度衰减因子 L ----
        F_sem_up = F.interpolate(F_sem, size=(256, 256), mode='bilinear', align_corners=False)  # (B,128,256,256)
        L_prime = self.lum_conv1(F_sem_up)   # (B,64,256,256)
        L_prime = self.lum_conv2(L_prime)    # (B,1,256,256)
        L = 0.1 + 0.9 * L_prime              # 值域 [0.1, 1.0]

        # ---- 2. 光源掩膜 M_light ----
        M_light = self.mask_conv1(F_sem_up)  # (B,64,256,256)
        M_light = self.mask_conv2(M_light)   # (B,1,256,256)

        # ---- 3. 光源颜色图 C ----
        C = self.color_conv1(F_sem_up)       # (B,64,256,256)
        C = self.color_conv2(C)              # (B,3,256,256)

        # ---- 4. 光晕模糊 sigma ----
        pooled = self.sigma_pool(F_sem).view(F_sem.size(0), -1)  # (B,128)
        sigma = self.sigma_fc(pooled).squeeze(-1)                # (B,)

        # ---- 5. 光源发光层 E ----
        E_base = M_light * C                  # (B,3,256,256)
        E = self.gaussian_blur(E_base, sigma) # (B,3,256,256)

        # ---- 6. 夜间效果层 ----
        # I_effect = I_scene * (L - 1) + E
        I_effect = I_scene * (L - 1) + E      # (B,3,256,256)
        # 通过 tanh 映射到 [-1,1]（与场景生成器输出范围一致）
        I_effect = torch.tanh(I_effect)

        # 组装 phys_info
        phys_info = {
            'L': L,
            'M_light': M_light,
            'E': E
        }

        return I_effect, phys_info

# ==================== 条件融合模块 (Sec 3.4) ====================
class ConditionalFusionModule(nn.Module):
    """
    条件融合模块
    输入:
        I_scene: (B,3,256,256)
        I_effect: (B,3,256,256)
        F_sem: (B,128,32,32)
        t: (B,) 天气强度编码（标量）
    输出:
        I_out: (B,3,256,256)
    """
    def __init__(self):
        super().__init__()
        # 三个卷积层：拼接特征后预测空间权重图 w(x)
        # 输入通道: 128(F_sem上采样后) + 3(I_effect) + 1(t_map) = 132
        self.conv = nn.Sequential(
            nn.Conv2d(132, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()   # w ∈ (0,1)
        )

    def forward(self, I_scene, I_effect, F_sem, t):
        """
            I_scene: (B,3,256,256)
            I_effect: (B,3,256,256)
            F_sem: (B,128,32,32)
            t: (B,) 或 (B,1) 天气强度（可以是标量、列表或张量）
            """
        # ---- 0. 确保 t 为张量且至少是1D ----
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=I_scene.device, dtype=I_scene.dtype)
        elif t.dim() == 0:
            t = t.unsqueeze(0)  # scalar -> (1,)

        # ---- 关键修复：将 t 扩展到与批次大小一致 ----
        if t.size(0) != I_scene.size(0):
            t = t.expand(I_scene.size(0))  # (1,) -> (B,) 或 (B,)保持不变

        # ---- 1. 上采样 F_sem 到 256x256 ----
        F_sem_up = F.interpolate(F_sem, size=(256, 256), mode='bilinear', align_corners=False)  # (B,128,256,256)

        # ---- 2. t 扩展为全图常量图 ----
        t = t.view(-1, 1, 1, 1)  # (B,1,1,1)
        t_map = t.expand(-1, 1, 256, 256)  # (B,1,256,256)

        # ---- 3. 拼接输入 ----
        fusion_input = torch.cat([F_sem_up, I_effect, t_map], dim=1)  # (B,132,256,256)

        # ---- 4. 预测权重图 w ----
        w = self.conv(fusion_input)  # (B,1,256,256)

        # ---- 5. 融合 ----
        I_out = torch.tanh(I_scene + I_effect * w)  # 保持 [-1,1]

        return I_out

# ==================== SPC生成器（整体框架） ====================
class SPCGenerator(nn.Module):
    """
    SPC生成器（完整框架）
    输入:
        I_in: (B,3,256,256) 晴天图像
        weather_type: str ('fog','rain','snow','night')
        weather_intensity: (B,) 或 scalar，默认为1.0
        enable_aux: bool 是否启用辅助预测头（训练时True）
    输出:
        I_out: (B,3,256,256) 生成的天气图像
        info_dict: dict 包含所有中间变量，用于损失计算
    """
    def __init__(self):
        super().__init__()
        # 子模块实例化
        self.semantic_encoder = StructuredSemanticEncoder(output_dim=128, pretrained=True)
        self.base_scene_gen = BaseSceneGenerator()
        self.physics_engine = DifferentiablePhysicsEngine()
        self.fusion_module = ConditionalFusionModule()

    def forward(self, I_in, weather_type, weather_intensity=1.0, enable_aux=False):
        """
        I_in: (B,3,256,256)
        weather_type: str
        weather_intensity: (B,) 或 scalar
        enable_aux: bool
        Returns: I_out (B,3,256,256), info_dict
        """
        # ---- 1. 结构化语义编码 ----
        F_sem, aux_info = self.semantic_encoder(I_in, enable_aux=enable_aux)  # F_sem: (B,128,32,32)

        # ---- 2. 基础场景生成 ----
        I_scene = self.base_scene_gen(I_in, F_sem)  # (B,3,256,256)

        # ---- 3. 可微物理引擎 ----
        I_effect, phys_info = self.physics_engine(I_scene, F_sem, weather_type)  # (B,3,256,256)

        # ---- 4. 条件融合 ----
        I_out = self.fusion_module(I_scene, I_effect, F_sem, weather_intensity)  # (B,3,256,256)

        # ---- 组装info_dict ----
        info_dict = {
            'I_scene': I_scene,
            'I_effect': I_effect,
            'F_sem': F_sem,
            'phys_info': phys_info,
            'aux_info': aux_info
        }

        return I_out, info_dict

# ==================== 多尺度判别器 ====================
class MultiScaleDiscriminator(nn.Module):
    """
    多尺度判别器
    输入: x (B,3,256,256)
    输出: 列表 [out1, out2, out3]
          out1: (B,1,32,32)   — 原始尺度
          out2: (B,1,16,16)   — 2倍下采样
          out3: (B,1,8,8)     — 4倍下采样
    """
    def __init__(self):
        super().__init__()
        # 三个尺度子网络，结构完全相同
        self.scale1 = self._make_scale()
        self.scale2 = self._make_scale()
        self.scale3 = self._make_scale()

    def _make_scale(self):
        """构建单个尺度的子网络"""
        return nn.Sequential(
            # 第一层: 无归一化
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 第二层: InstanceNorm
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 第三层: InstanceNorm
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 输出层: 无归一化，无激活
            nn.Conv2d(256, 1, kernel_size=4, padding=1, bias=True)
        )

    def forward(self, x):
        """
        x: (B,3,256,256)
        Returns: list of three tensors
        """
        # 原始尺度
        out1 = self.scale1(x)                     # (B,1,32,32)

        # 2倍下采样
        x2 = F.avg_pool2d(x, kernel_size=2, stride=2)  # (B,3,128,128)
        out2 = self.scale2(x2)                    # (B,1,16,16)

        # 4倍下采样
        x4 = F.avg_pool2d(x, kernel_size=4, stride=4)  # (B,3,64,64)
        out3 = self.scale3(x4)                    # (B,1,8,8)

        return [out1, out2, out3]

# ==================== SPC完整模型（包含生成器和判别器） ====================
class SPCModel(nn.Module):
    def __init__(self, weather_type='fog'):
        super().__init__()
        self.generator = SPCGenerator()
        self.discriminator = MultiScaleDiscriminator()
        self.weather_type = weather_type
        self.semantic_encoder = self.generator.semantic_encoder

    def forward(self, I_in, weather_intensity=1.0, enable_aux=False):
        return self.generator(I_in, self.weather_type, weather_intensity, enable_aux)