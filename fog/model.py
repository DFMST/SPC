import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
import torch


class DetailEnhanceBlock(nn.Module):
    """细节增强模块 - 优化高分辨率细节"""

    def __init__(self, channels):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels // 2, channels, 3, padding=1),
            ChannelAttention(channels),  # 添加通道注意力
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * (1 + self.branch(x))


class EdgeAwareBlock(nn.Module):
    """边缘感知模块"""

    def __init__(self, channels):
        super().__init__()
        # Sobel算子初始化
        self.register_buffer('sobel_x', torch.tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]]).float())
        self.register_buffer('sobel_y', torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]]).float())

        # 边缘增强分支
        self.edge_branch = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 计算边缘
        gray = torch.mean(x, dim=1, keepdim=True)
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        edge_map = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

        # 增强边缘响应
        edge_enhanced = self.edge_branch(edge_map)

        # 应用边缘增强
        return x * (1 + edge_enhanced)

class ResidualBlock(nn.Module):
    """增强型残差块，支持可选归一化"""
    def __init__(self, in_features, use_norm=True):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features) if use_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features) if use_norm else nn.Identity()
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x)


class GradientReweighting(nn.Module):
    """梯度重加权模块"""

    def __init__(self, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 128 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128 // reduction, 128, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SemanticEncoder(nn.Module):
    def __init__(self, input_nc=3, output_dim=128, pretrained=True):
        super().__init__()
        # 使用ResNet50作为主干网络
        resnet = models.resnet50(pretrained=pretrained)

        # 特征提取层(保留多尺度特征)
        self.conv1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 256通道
        self.layer2 = resnet.layer2  # 512通道
        self.layer3 = resnet.layer3  # 1024通道

        # 深度监督分支
        self.auxiliary_heads = nn.ModuleList([
            self._make_aux_head(256, output_dim),  # layer1输出
            self._make_aux_head(512, output_dim),  # layer2输出
        ])

        # 特征融合模块
        self.fusion = nn.Sequential(
            nn.Conv2d(256 + 512 + 1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, output_dim, 1)
        )

        # 梯度重加权
        self.gradient_reweight = GradientReweighting(output_dim)

        # 冻结部分参数
        if pretrained:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.layer1.parameters():
                param.requires_grad = False

    def _make_aux_head(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, 1)
        )

    def forward(self, x):
        # 特征提取
        x1 = self.conv1(x)  # [B, 64, 128, 128]
        x2 = self.maxpool(x1)  # [B, 64, 64, 64]
        x3 = self.layer1(x2)  # [B, 256, 64, 64]
        x4 = self.layer2(x3)  # [B, 512, 32, 32]
        x5 = self.layer3(x4)  # [B, 1024, 16, 16]

        # 上采样到统一尺寸(32x32)
        x3_up = F.interpolate(x3, size=32, mode='bilinear', align_corners=False)
        x5_up = F.interpolate(x5, size=32, mode='bilinear', align_corners=False)

        # 特征融合
        fused = torch.cat([x3_up, x4, x5_up], dim=1)
        main_feat = self.fusion(fused)  # [B, output_dim, 32, 32]

        # 深度监督分支
        aux1 = self.auxiliary_heads[0](x3)  # [B, 128, 64, 64]
        aux2 = self.auxiliary_heads[1](x4)  # [B, 128, 32, 32]

        # 应用梯度重加权
        main_feat = self.gradient_reweight(main_feat)

        return main_feat, [aux1, aux2]


class SemanticDecoder(nn.Module):
    """增强型语义解码器，带特征精炼模块"""

    def __init__(self, input_dim=128, output_nc=6, feature_dim=256):
        super().__init__()
        # 特征精炼模块
        self.refinement = nn.Sequential(
            ResidualBlock(input_dim),
            ResidualBlock(input_dim),
            ResidualBlock(input_dim),
            ResidualBlock(input_dim)
        )

        # 渐进上采样路径
        self.upsample = nn.Sequential(
            # 32x32→64x64
            UpBlock(input_dim, 256, scale_factor=2),
            ResidualBlock(256),
            # 64x64→128x128
            UpBlock(256, 128, scale_factor=2),
            ResidualBlock(128),
            # 128x128→256x256
            UpBlock(128, 64, scale_factor=2),
            ResidualBlock(64),
        )

        # 输出层
        self.output = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_nc, 1)
        )

        # 创建并注册 Sobel滤波器作为缓冲区
        self._create_edge_detector()

    def forward(self, x):
        x = self.refinement(x)
        x = self.upsample(x)
        return self.output(x)

    def _create_edge_detector(self):
        """创建边界检测器(不可训练)"""
        sobel_x = torch.tensor([[[1, 0, -1], [2, 0, -2], [1, 0, -1]]],
                               dtype=torch.float32).repeat(1, 1, 1, 1)  # [1,1,3,3]
        sobel_y = torch.tensor([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]],
                               dtype=torch.float32).repeat(1, 1, 1, 1)  # [1,1,3,3]

        # 注册为缓冲区，不参与梯度更新
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def compute_loss(self, pred, target, weather_mask=None):
        """修正的语义损失计算-避免数值不稳定&天气区域降权"""
        # 添加 log_softmax确保数值稳定
        pred_log = F.log_softmax(pred / 0.5, dim=1)  # T=0.5

        # 基础交叉熵损失
        ce_loss = F.nll_loss(pred_log, target, reduction='none')

        # 应用天气区域降权
        if weather_mask is not None:
            # 确保掩码形状匹配[B,H,W]
            if weather_mask.dim() == 4:
                weather_mask = weather_mask.squeeze(1)
            weather_mask = weather_mask.to(pred_log.device)

            # 天气区域损失权重降为0.3，非天气区域保持1.0
            weight_map = torch.where(weather_mask > 0.5, 0.3, 1.0)
            ce_loss = ce_loss * weight_map

        ce_loss = ce_loss.mean()

        # 边界权重图
        with torch.no_grad():
            target_edges = self._compute_edges(target.unsqueeze(1).float())
            edge_weights = 1.0 + 3.0 * torch.clamp(target_edges, 0, 1)  # 权重 ∈ [1,2]

        # 边界加权损失
        weighted_loss_pixels = F.nll_loss(pred_log, target, reduction='none')

        # 确保权重形状匹配[B,H,W]
        if edge_weights.dim() == 4:
            edge_weights = edge_weights.squeeze(1)

        # 应用边界权重
        weighted_loss = (weighted_loss_pixels * edge_weights).mean()

        return 0.4 * ce_loss + 0.6 * weighted_loss

    def _compute_edges(self, sem_map):
        """计算语义边界"""
        grad_x = F.conv2d(sem_map, self.sobel_x, padding=1)
        grad_y = F.conv2d(sem_map, self.sobel_y, padding=1)
        edge_map = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        return edge_map


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        # 使用亚像素卷积提升细节
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class RaindropSimulator(nn.Module):
    """增强的雨滴模拟器-显著提升雨滴可见度"""

    def __init__(self):
        super().__init__()
        # 纹理增强器
        self.texture_enhancer = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, intensity):
        b, c, h, w = x.shape

        # 增强基础强度
        intensity = intensity.view(-1, 1, 1, 1) * 2.0  # 加倍强度

        # 生成更明显的雨滴参数
        params = torch.randn(b, 4, 1, 1, device=x.device) * 0.5
        density = torch.sigmoid(params[:, 0:1]) * 0.8 + 0.2  # 密度 [0.2, 1.0]
        drop_size = torch.sigmoid(params[:, 1:2]) * 3.0 + 2.0  # 雨滴大小 [2.0, 5.0]

        # 创建更明显的雨滴模式
        base_pattern = self._create_rain_pattern(b, 1, h, w, x.device)

        # 添加随机变化
        random_variation = torch.randn(b, 1, h, w, device=x.device) * 0.5
        rain_pattern = base_pattern + random_variation

        # 应用密度控制 - 增强对比度
        rain_drops = torch.sigmoid(rain_pattern * 8.0) * density

        # 纹理增强
        rain_drops_enhanced = self.texture_enhancer(rain_drops)

        # 创建方向性运动模糊核
        dir_x = torch.tanh(params[:, 2:3]) * 3.0  # 增强方向性 [-3, 3]
        dir_y = torch.tanh(params[:, 3:4]) * 3.0
        motion_kernel = self._create_directional_motion_kernel(drop_size, dir_x, dir_y)

        # 应用运动模糊
        rain_effect = F.conv2d(
            rain_drops_enhanced,
            motion_kernel,
            padding='same',
            groups=1
        )

        # 复制到所有通道
        rain_effect = rain_effect.repeat(1, 3, 1, 1)

        # 增强最终效果强度
        final_effect = rain_effect * intensity * 5.0  # 大幅增强强度

        return torch.clamp(final_effect, 0, 2.0)  # 允许超过1.0的亮度

    def _create_rain_pattern(self, batch_size, channels, height, width, device):
        """创建更明显的条纹状雨滴模式"""
        pattern = torch.zeros(batch_size, channels, height, width, device=device)

        for i in range(batch_size):
            # 更垂直的角度，减少倾斜
            angle = torch.rand(1, device=device) * 0.1 - 0.05  # [-0.05, 0.05]弧度

            # 创建更密集的雨滴条纹
            for y in range(height):
                for x in range(width):
                    stripe_pos = (x * torch.sin(angle) + y * torch.cos(angle)) * 0.08  # 减小间距
                    if torch.sin(stripe_pos) > 0.9:  # 更高的阈值，更细的条纹
                        pattern[i, 0, y, x] = torch.rand(1, device=device) * 4.0  # 更亮的雨滴

            # 增加随机雨滴点数量
            num_dots = int(height * width * 0.03)  # 增加到3%的像素点
            for _ in range(num_dots):
                dot_y = torch.randint(0, height, (1,), device=device)
                dot_x = torch.randint(0, width, (1,), device=device)
                pattern[i, 0, dot_y, dot_x] = torch.rand(1, device=device) * 5.0  # 更亮的点

        return pattern

    def _create_directional_motion_kernel(self, drop_size, dir_x, dir_y):
        """创建更明显的雨滴运动轨迹"""
        batch_size = drop_size.size(0)
        kernel_size = 15  # 增大核大小以获得更长的轨迹

        kernels = []
        for i in range(batch_size):
            kernel = torch.zeros(1, 1, kernel_size, kernel_size, device=drop_size.device)
            center = kernel_size // 2

            # 计算运动方向和长度（增强方向性）
            dx = dir_x[i].item() * 2.0  # 增强方向性
            dy = dir_y[i].item() * 2.0
            length = int(max(3, min(center - 1, drop_size[i].mean().item() * 2.0)))  # 更长的轨迹

            # 创建线性运动轨迹
            total_weight = 0
            for t in range(-length, length + 1):
                if t == 0:
                    weight = 1.0  # 中心点最大权重
                else:
                    dist = abs(t) / length
                    weight = math.exp(-dist * 1.5)  # 更平缓的衰减

                x_pos = center + int(t * dx)
                y_pos = center + int(t * dy)

                if 0 <= x_pos < kernel_size and 0 <= y_pos < kernel_size:
                    kernel[0, 0, y_pos, x_pos] = weight
                    total_weight += weight

            if total_weight > 0:
                kernel = kernel / total_weight
            else:
                kernel[0, 0, center, center] = 1.0

            kernels.append(kernel)

        return torch.cat(kernels, dim=0)


class FogGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # 增强深度预测网络（增加跳跃连接和感受野）
        self.depth_predictor = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 下采样1
            ResidualBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 下采样2
            ResidualBlock(128),
            # 跳跃连接融合
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.LeakyReLU(0.2),
            ChannelAttention(128),  # 加入通道注意力
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

        # 深度图后处理模块
        self.depth_refine = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            DetailEnhanceBlock(32),
            nn.Conv2d(32, 1, 1)
        )

        # 改进的大气光预测器
        self.light_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Sigmoid()  # 输出0-1范围
        )


    def forward(self, x, intensity_map):
        # 深度预测与精炼
        depth_map = self.depth_predictor(x)
        depth_map = self.depth_refine(depth_map)

        # 空间变化的beta计算（增强连续性）
        scene_luminance = 0.3 * x[:, 0] + 0.59 * x[:, 1] + 0.11 * x[:, 2]
        # 使用局部平均亮度
        local_brightness = F.avg_pool2d(scene_luminance.unsqueeze(1), kernel_size=7, padding=3, stride=1)
        beta = 0.03 + 0.25 * intensity_map * (1.0 + 0.7 * local_brightness)

        transmission = torch.exp(-beta * depth_map)

        # 大气光预测（增加空间变化）
        L = self.light_predictor(x)  # [B,3]
        L = L.view(-1, 3, 1, 1)  # 重塑为[B,3,1,1]
        L_spatial = F.interpolate(L, size=x.shape[2:], mode='bilinear')
        L_spatial = torch.clamp(L_spatial, 0.5, 0.95)  # 限制光强范围

        # 物理模型添加噪声扰动（模拟雾的不均匀性）
        noise_mask = torch.randn_like(x)[:, :1] * 0.1 * intensity_map
        transmission = transmission * (1 + noise_mask)

        # 物理校正的雾效公式
        fog_effect = x * transmission + L_spatial * (1 - transmission)
        return fog_effect





class SnowSimulator(nn.Module):
    """改进的雪天物理模型"""

    def __init__(self):
        super().__init__()
        # 增加参数预测能力
        self.param_predictor = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 1)  # [大小,密度,方向]
        )
        # 可学习风场参数
        self.wind_field = nn.Parameter(torch.tensor([0.7, 0.3]))

    def forward(self, x, intensity):
        # 确保intensity有正确的维度[B,1,1,1]
        intensity = intensity.view(-1, 1, 1, 1)

        # 生成雪片参数
        params = self.param_predictor(x)
        flake_size = torch.sigmoid(params[:, 0:1]) * 6 + 2  # 2-8像素
        density = torch.sigmoid(params[:, 1:2]) * 1.2 * intensity  # 增加密度系数

        # 改进的雪片生成：使用分形噪声替代随机噪声
        base_noise = torch.randn_like(x) * 0.5
        # 添加分形细节
        noise1 = F.interpolate(torch.randn_like(x), scale_factor=0.5, mode='bilinear')
        noise1 = F.interpolate(noise1, size=x.shape[2:], mode='bilinear')
        noise2 = F.interpolate(torch.randn_like(x), scale_factor=0.25, mode='bilinear')
        noise2 = F.interpolate(noise2, size=x.shape[2:], mode='bilinear')

        combined_noise = (base_noise + 0.5 * noise1 + 0.25 * noise2) / 1.75
        snow_flakes = torch.sigmoid(combined_noise * density)

        # 运动模糊(考虑风场方向)
        avg_flake_size = flake_size.mean(dim=[2, 3], keepdim=True)
        motion_kernel = self._create_directional_kernel(avg_flake_size)

        # 应用运动模糊
        snow_effect = F.conv2d(
            snow_flakes,
            motion_kernel.expand(3, 1, -1, -1),
            padding='same',
            groups=3
        )
        return snow_effect * intensity

    def _create_directional_kernel(self, flake_size):
        """创建方向性雪片模糊核"""
        flake_size_scalar = flake_size.mean().item()
        kernel_size = int(max(7, flake_size_scalar * 1.5))

        kernel = torch.zeros(1, 1, kernel_size, kernel_size,
                             device=flake_size.device)
        center = kernel_size // 2

        # 获取风场方向
        wind_x, wind_y = self.wind_field.data
        angle = torch.atan2(wind_y, wind_x)
        length = max(0.1, torch.sqrt(wind_x ** 2 + wind_y ** 2))

        # 创建椭圆型模糊核（沿风向拉伸）
        for i in range(kernel_size):
            for j in range(kernel_size):
                dx = j - center
                dy = i - center

                # 坐标旋转（使主轴沿风向）
                rot_x = dx * torch.cos(angle) + dy * torch.sin(angle)
                rot_y = -dx * torch.sin(angle) + dy * torch.cos(angle)

                # 椭圆方程 (a=沿风向，b=垂直风向)
                a = flake_size_scalar * 1.2
                b = flake_size_scalar * 0.6
                value = 1.0 - (rot_x ** 2 / a ** 2 + rot_y ** 2 / b ** 2)

                if value > 0:
                    # 增加中心权重
                    dist_center = math.sqrt(dx ** 2 + dy ** 2) / (kernel_size / 2)
                    value *= (1 - dist_center ** 2)
                    kernel[0, 0, i, j] = max(0, value)

        # 归一化
        kernel_sum = kernel.sum() + 1e-6
        return kernel / kernel_sum


class NightAdjuster(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用更轻量的深度预测器
        self.depth_predictor = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.UpsamplingBilinear2d(scale_factor=4),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

        # 简化光照预测器
        self.light_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3, 3, 1),
            nn.Tanh()
        )

    def forward(self, x, intensity):
        # 确保intensity有正确的维度[B, 1, 1, 1]
        intensity = intensity.view(-1, 1, 1, 1)

        # 深度预测
        depth = self.depth_predictor(x)
        depth = F.interpolate(depth, size=x.shape[2:], mode='bilinear')

        # 修复维度问题：确保所有张量有相同空间维度
        attenuation = 1.0 / (depth + 0.8)

        # 正确广播intensity到attenuation的形状
        night_factor = torch.exp(-intensity * attenuation * 1.5)

        # 光照效果
        L = self.light_predictor(x) * 0.5 + 0.8

        # 最终融合
        return torch.clamp(x * night_factor + L * (1 - night_factor) * intensity, 0, 1)

    def _detect_edges(self, img):
        if img.dim() == 3:
            img = img.unsqueeze(0)

        # 使用更精确的Sobel算子
        self.register_buffer('sobel_x', torch.tensor([
            [[[-1.0, 0.0, 1.0],
              [-2.0, 0.0, 2.0],
              [-1.0, 0.0, 1.0]]]
        ], device=img.device))

        self.register_buffer('sobel_y', torch.tensor([
            [[[-1.0, -2.0, -1.0],
              [0.0, 0.0, 0.0],
              [1.0, 2.0, 1.0]]]
        ], device=img.device))

        # 多通道边缘融合
        edge_maps = []
        for c in range(img.shape[1]):
            channel = img[:, c:c + 1]
            grad_x = F.conv2d(channel, self.sobel_x, padding=1)
            grad_y = F.conv2d(channel, self.sobel_y, padding=1)
            edge_map = torch.sqrt(grad_x ** 2 + grad_y ** 2)
            edge_maps.append(edge_map)

        # 取各通道最大值作为最终边缘
        combined = torch.max(torch.cat(edge_maps, dim=1), dim=1, keepdim=True)[0]
        return torch.sigmoid(combined * 8.0)  # 增强边缘响应





class EnhancedSceneParser(nn.Module):
    """增强型场景解析器（替换原LightweightSceneParser）"""

    def __init__(self, semantic_dim):
        super().__init__()
        self.region_classifier = nn.Sequential(
            nn.Conv2d(semantic_dim, 128, 3, padding=1),
            nn.ReLU(),
            EdgeAwareBlock(128),  # 新增边界感知模块
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, semantic_features):
        return self.region_classifier(semantic_features)


# 修改RegionAwareFusion的融合策略 - 添加边界约束
class RegionAwareFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # 建筑边缘检测器
        self.building_edge_detector = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        # 边缘白光抑制器
        self.edge_suppresser = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid()
        )

    def forward(self, base_scene, weather_effect, region_mask, weather_type=None):
        """区域感知融合-专门优化雨天效果"""
        # 如果是雨天，使用专门的雨天融合策略
        if weather_type == "rain" or weather_type == 0:
            return self._rain_specific_fusion(base_scene, weather_effect, region_mask)

        # 其他天气类型的原有逻辑保持不变
        # 建筑边缘抑制（通用逻辑）
        building_edges = self.building_edge_detector(base_scene)
        suppressed_edges = self.edge_suppresser(base_scene)
        weather_effect = weather_effect * (1 - building_edges) + weather_effect * building_edges * suppressed_edges

        # 原有通用融合逻辑
        sky_effect = weather_effect * region_mask[:, 0:1] * 0.8
        dynamic_effect = weather_effect * region_mask[:, 1:2] * 0.7
        ground_effect = weather_effect * region_mask[:, 2:3] * 1.0

        return base_scene + sky_effect + dynamic_effect + ground_effect

    def _rain_specific_fusion(self, base_scene, rain_effect, region_mask):
        """雨天专用融合策略-最大化雨滴可见度"""

        # 1. 大幅减少建筑边缘抑制
        building_edges = self.building_edge_detector(base_scene)
        edge_suppression_factor = 0.1  # 从0.3降到0.1
        rain_preserved = rain_effect * (1 - building_edges * edge_suppression_factor)

        # 2. 增强所有区域的雨滴效果
        sky_mask = region_mask[:, 0:1]
        dynamic_mask = region_mask[:, 1:2]
        ground_mask = region_mask[:, 2:3]

        # 大幅增强各区域雨滴强度
        enhanced_sky_rain = rain_preserved * sky_mask * 2.5  # 天空区域增强150%
        enhanced_ground_rain = rain_preserved * ground_mask * 2.0  # 地面增强100%
        enhanced_dynamic_rain = rain_preserved * dynamic_mask * 1.8  # 动态物体增强80%

        # 3. 直接叠加，减少透明度混合
        combined_rain = enhanced_sky_rain + enhanced_ground_rain + enhanced_dynamic_rain

        # 4. 使用更强的叠加方式
        result = base_scene + combined_rain * 1.2  # 增强叠加系数

        return torch.clamp(result, 0, 1.0)


class BaseSceneGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, semantic_dim=128):
        super().__init__()
        # 天气特征适配器
        self.weather_adapter = nn.Sequential(
            nn.Conv2d(semantic_dim, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1)
        )

        # 通道适配器 - 适配更高分辨率
        self.channel_adapter = nn.ModuleDict({
            'down1': nn.Conv2d(128 + 256, 128, 1),
            'down2': nn.Conv2d(256 + 256, 256, 1),
            'down3': nn.Conv2d(512 + 256, 512, 1),  # 新增512通道层
            'res_blocks': nn.Conv2d(512 + 256, 512, 1),
            'up1': nn.Conv2d(256 + 256, 256, 1),
            'up2': nn.Conv2d(128 + 256, 128, 1),
            'up3': nn.Conv2d(64 + 256, 64, 1)  # 新增上采样层
        })

        # 初始化层
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 下采样层 (3层下采样)
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.down3 = nn.Sequential(  # 新增下采样层
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # 残差块 (增加到9个)
        self.res_blocks = nn.Sequential(
            ResidualBlock(512, use_norm=True),
            ResidualBlock(512, use_norm=True),
            ResidualBlock(512, use_norm=True),
            ResidualBlock(512, use_norm=True),
            ResidualBlock(512, use_norm=True),
            ResidualBlock(512, use_norm=True),
            ResidualBlock(512, use_norm=True),
            ResidualBlock(512, use_norm=True),
            ResidualBlock(512, use_norm=True)
        )

        # 上采样层 (3层上采样)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(  # 新增上采样层
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 输出层 - 添加细节增强
        self.output = nn.Sequential(
            DetailEnhanceBlock(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            NoiseSuppression(64),  # 噪声抑制
            nn.Conv2d(64, output_nc, 3, padding=1),
            nn.Tanh()
        )

        # 残差连接
        self.skip1 = nn.Conv2d(64, 64, 1)
        self.skip2 = nn.Conv2d(128, 128, 1)
        self.skip3 = nn.Conv2d(256, 256, 1)

    def weather_injection(self, x, weather_vec, injection_point):
        b, c, h, w = x.shape
        adapted_vec = weather_vec.repeat(1, 1, h, w)
        combined = torch.cat([x, adapted_vec], dim=1)
        adapter = self.channel_adapter[injection_point]
        return adapter(combined)

    def forward(self, x, semantic):
        weather_vec = self.weather_adapter(semantic)

        # 初始层
        x0 = self.initial(x)
        skip0 = self.skip1(x0)

        # 下采样路径
        x1 = self.down1(x0)
        skip1 = self.skip2(x1)
        x1 = self.weather_injection(x1, weather_vec, 'down1')

        x2 = self.down2(x1)
        skip2 = self.skip3(x2)
        x2 = self.weather_injection(x2, weather_vec, 'down2')

        x3 = self.down3(x2)
        x3 = self.weather_injection(x3, weather_vec, 'down3')

        # 残差块
        x4 = self.res_blocks(x3)
        x4 = self.weather_injection(x4, weather_vec, 'res_blocks')

        # 上采样路径
        x5 = self.up1(x4) + skip2  # 残差连接
        x5 = self.weather_injection(x5, weather_vec, 'up1')

        x6 = self.up2(x5) + skip1  # 残差连接
        x6 = self.weather_injection(x6, weather_vec, 'up2')

        x7 = self.up3(x6) + skip0  # 残差连接
        x7 = self.weather_injection(x7, weather_vec, 'up3')

        return self.output(x7)


class DisentangledGenerator(nn.Module):
    """解耦天气生成器"""

    def __init__(self, input_nc=3, output_nc=3, semantic_dim=128, weather_type="rain"):
        super().__init__()
        self.weather_type = weather_type

        # 基础场景生成器不变
        self.base_generator = BaseSceneGenerator(input_nc, output_nc, semantic_dim)

        # 根据天气类型选择物理模型
        if weather_type == "rain":
            self.weather_layer = PhysicsInformedRainLayer(semantic_dim)
        elif weather_type == "snow":
            self.weather_layer = PhysicsInformedSnowLayer(semantic_dim)
        elif weather_type == "fog":
            self.weather_layer = PhysicsInformedFogLayer(semantic_dim)
        elif weather_type == "night":
            self.weather_layer = PhysicsInformedNightLayer(semantic_dim)

        # 区域感知融合模块不变
        self.fusion_module = RegionAwareFusion()
        self.scene_parser = EnhancedSceneParser(semantic_dim)
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)


    def forward(self, x, semantic_features):
        # 基础场景生成
        base_scene = self.base_generator(x, semantic_features)



        # 物理天气生成
        weather_effect, rain_intensity, fog_intensity = self.weather_layer(
            base_scene, semantic_features
        )  # [B, 3, 256, 256]



        # 物理天气生成 - 修改为处理字典返回值
        if self.weather_type == "fog":
            fog_output = self.weather_layer(base_scene, semantic_features)
            weather_effect = fog_output["output"]
            fog_intensity = fog_output["fog_intensity"]
            physics_loss = fog_output["physics_loss"]
            rain_intensity = torch.zeros_like(fog_intensity)

        # 场景解析 - 确保输出正确的区域掩码
        region_mask = self.scene_parser(semantic_features)  # [B, 3, H_low, W_low]

        # 确保区域掩码是浮点类型
        region_mask = region_mask.float()

        # 确保区域掩码有正确的通道数
        if region_mask.size(1) != 3:
            # 如果通道数不对，创建默认区域掩码
            print(f"Warning: Unexpected region_mask shape: {region_mask.shape}")
            region_mask = torch.zeros(
                region_mask.size(0), 3, region_mask.size(2), region_mask.size(3),
                device=region_mask.device
            )
            region_mask[:, 0] = 1.0  # 全部设为天空区域

        # 上采样区域掩码匹配天气效果分辨率
        region_mask_up = F.interpolate(
            region_mask,
            size=weather_effect.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        # 应用区域感知融合
        output = self.fusion_module(
            base_scene,
            weather_effect,
            region_mask_up  # 使用上采样后的区域掩码
        )

        return {
            "output": output,
            "base_scene": base_scene,
            "weather_effect": weather_effect,
            "region_mask": region_mask,  # 原始分辨率的区域掩码
            "physics_loss": physics_loss,
            "weather_type": self.weather_type
        }

# 原有雨天层保持不变
class PhysicsInformedRainLayer(nn.Module):
    """物理约束雨天生成层-显著增强雨滴可见度版本"""

    def __init__(self, input_dim=128):
        super().__init__()

        # 增强天气参数预测
        self.weather_param_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 输出雨强度、细节强度、空间变化
        )

        self.raindrop_simulator = RaindropSimulator()

        # 增强空间强度预测器
        self.spatial_intensity_predictor = nn.Sequential(
            nn.Conv2d(input_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, base_scene, semantic_features):
        # 预测更强的雨滴参数
        params = self.weather_param_predictor(semantic_features)
        rain_intensity = torch.sigmoid(params[:, 0:1]) * 2.5  # 增强到[0, 2.5]
        detail_intensity = 1.0 + torch.sigmoid(params[:, 1:2]) * 0.6  # [1.0, 1.6]
        spatial_variation = torch.sigmoid(params[:, 2:3])  # 空间变化因子

        # 预测更强的空间强度图
        h, w = base_scene.shape[2], base_scene.shape[3]
        sem_up = F.interpolate(semantic_features, size=(h, w), mode='bilinear')
        spatial_intensity_map = self.spatial_intensity_predictor(sem_up) * 1.5  # 增强

        # 组合强度：基础强度 + 空间变化 + 细节增强
        combined_intensity = (rain_intensity * 0.8 +
                              spatial_intensity_map * 0.4 * spatial_variation) * detail_intensity

        # 应用增强的雨滴效果
        rain_effect = self.raindrop_simulator(base_scene, combined_intensity)

        # 天空区域增强雨滴效果
        with torch.no_grad():
            # 简易天空检测（亮度较高的区域）
            gray = torch.mean(base_scene, dim=1, keepdim=True)
            sky_mask = (gray > 0.6).float()
            # 在天空区域增强雨滴
            rain_effect = rain_effect * (1 + sky_mask * 0.8)  # 增强天空区域效果

        return rain_effect, rain_intensity, torch.zeros_like(rain_intensity)


# 新增雾天层
class PhysicsInformedFogLayer(nn.Module):
    """物理约束的雾天生成层"""

    def __init__(self, input_dim=128):
        super().__init__()
        # 空间感知的雾强度预测器
        self.fog_intensity_predictor = nn.Sequential(
            nn.Conv2d(input_dim, 64, 3, padding=1),
            nn.ReLU(),
            DetailEnhanceBlock(64),  # 加入细节增强
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        self.fog_generator = FogGenerator()

        # 创建并注册Sobel滤波器用于边缘检测
        sobel_x = torch.tensor([[[[1.0, 0.0, -1.0],
                                  [2.0, 0.0, -2.0],
                                  [1.0, 0.0, -1.0]]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[[1.0, 2.0, 1.0],
                                  [0.0, 0.0, 0.0],
                                  [-1.0, -2.0, -1.0]]]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, base_scene, semantic_features):
        # 上采样语义特征匹配场景分辨率
        h, w = base_scene.shape[2], base_scene.shape[3]
        sem_up = F.interpolate(semantic_features, size=(h, w), mode='bilinear', align_corners=False)

        # 预测空间变化的强度图
        fog_intensity_map = self.fog_intensity_predictor(sem_up)

        # 应用场景亮度感知的强度调整
        scene_luminance = 0.3 * base_scene[:, 0] + 0.59 * base_scene[:, 1] + 0.11 * base_scene[:, 2]
        fog_intensity_map = fog_intensity_map * (0.7 + 0.3 * scene_luminance.unsqueeze(1))

        # 生成雾效
        fog_effect = self.fog_generator(base_scene, fog_intensity_map)

        # 边缘保留雾效（减少边缘处的白色块）
        edge_mask = self._detect_edges(base_scene)
        blended = base_scene * edge_mask + fog_effect * (1 - edge_mask)

        # 计算物理约束损失
        physics_loss = self._compute_physics_loss(fog_effect, base_scene, fog_intensity_map)

        return {
            "output": blended,
            "fog_intensity": fog_intensity_map.mean(),
            "physics_loss": physics_loss
        }

    def _detect_edges(self, img):
        """检测图像边缘"""
        # 转换为灰度图
        gray = 0.2989 * img[:, 0] + 0.5870 * img[:, 1] + 0.1140 * img[:, 2]
        gray = gray.unsqueeze(1)

        # 计算梯度
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        edge_map = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

        return torch.sigmoid(edge_map * 5.0)  # 增强边缘响应

    def _compute_physics_loss(self, fog_effect, base_scene, intensity_map):
        """稳定可靠的物理约束损失计算"""
        # 1. 雾效连续性损失（确保空间平滑）
        fog_density = 1.0 - fog_effect.mean(dim=1, keepdim=True)  # 雾浓度图

        # 使用Sobel算子计算梯度，避免使用绝对值导致不可导
        sobel_x = torch.tensor([[[[1.0, 0.0, -1.0],
                                  [2.0, 0.0, -2.0],
                                  [1.0, 0.0, -1.0]]]],
                               device=fog_effect.device)
        sobel_y = torch.tensor([[[[1.0, 2.0, 1.0],
                                  [0.0, 0.0, 0.0],
                                  [-1.0, -2.0, -1.0]]]],
                               device=fog_effect.device)

        grad_x = F.conv2d(fog_density, sobel_x, padding=1)
        grad_y = F.conv2d(fog_density, sobel_y, padding=1)

        # 使用L2范数计算梯度幅度，避免梯度爆炸
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        continuity_loss = torch.mean(gradient_magnitude) * 0.1

        # 2. 边缘区域雾效抑制（防止白色块）
        edge_mask = self._detect_edges(base_scene)

        # 使用软约束而非硬约束，避免梯度不连续
        edge_penalty = torch.mean(fog_effect * edge_mask) * 0.05

        # 3. 雾效可见性损失（确保雾效不过强也不过弱）
        visibility_loss = torch.abs(torch.mean(fog_effect) - 0.3 * torch.mean(intensity_map)) * 0.2

        # 4. 雾效范围约束（防止极端值）
        range_penalty = torch.mean(torch.relu(fog_effect - 0.9) + torch.relu(0.1 - fog_effect)) * 0.05

        # 总物理损失 - 所有项都是非负的
        physics_loss = (
                continuity_loss +
                edge_penalty +
                visibility_loss +
                range_penalty
        )

        # 确保损失不会变为负值
        physics_loss = torch.clamp(physics_loss, min=0.0, max=5.0)

        return physics_loss

# 新增雪天层
class PhysicsInformedSnowLayer(nn.Module):
    """物理约束雪天生成层"""

    def __init__(self, input_dim=128):
        super().__init__()
        # 天气参数预测 - 只预测雪强度
        self.weather_param_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 只输出雪强度
        )
        self.snow_simulator = SnowSimulator()

    def forward(self, base_scene, semantic_features):
        snow_intensity = torch.sigmoid(self.weather_param_predictor(semantic_features))
        snow_effect = self.snow_simulator(base_scene, snow_intensity)
        return snow_effect, torch.zeros_like(snow_intensity), torch.zeros_like(snow_intensity)


# 新增夜晚层
class PhysicsInformedNightLayer(nn.Module):
    """物理约束的夜晚效果生成层（完整实现）"""

    def __init__(self, input_dim=128):
        super().__init__()
        # 天气参数预测 - 只预测夜强度
        self.weather_param_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 只输出夜强度
        )
        self.night_adjuster = NightAdjuster()
        # 添加深度预测器（复用FogGenerator中的组件）
        self.depth_predictor = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, base_scene, semantic_features):
        # 预测夜强度
        night_intensity = torch.sigmoid(self.weather_param_predictor(semantic_features))
        # 生成夜晚效果
        night_effect = self.night_adjuster(base_scene, night_intensity)
        return night_effect, torch.zeros_like(night_intensity), torch.zeros_like(night_intensity)

# ======================= 修改SemanticCycleGAN类 =======================
class SemanticCycleGAN(nn.Module):
    def __init__(self, device='cuda', semantic_dim=128, weather_type="rain"):
        super().__init__()
        self.device = device
        self.weather_type = weather_type
        # 语义编解码器
        # 语义编解码器
        self.semantic_encoder = SemanticEncoder(output_dim=semantic_dim).to(device)
        self.semantic_decoder = SemanticDecoder(input_dim=semantic_dim).to(device)

        # 使用解耦生成器 - 传入天气类型
        self.G_A2B = DisentangledGenerator(
            semantic_dim=semantic_dim,
            weather_type=weather_type
        ).to(device)

        self.G_B2A = DisentangledGenerator(
            semantic_dim=semantic_dim,
            weather_type=weather_type
        ).to(device)
        # 使用解耦生成器

        # 判别器
        self.D_A = Discriminator().to(device)  # A域判别器
        self.D_B = Discriminator().to(device)  # B域判别器

        # 损失函数
        self.criterion_gan = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_semantic = nn.NLLLoss()

    def forward(self, real_A, real_B):
        """前向传播计算所有必要输出"""
        # 获取A域的语义特征
        sem_feature_A, _ = self.semantic_encoder(real_A)

        # 使用解耦生成器生成结果
        fake_B_dict = self.G_A2B(real_A, sem_feature_A)

        # 训练前期使用弱雾效（防止过早形成白色块）
        if self.training and self.weather_type == "fog":
            intensity_factor = 1.0
            base_scene = fake_B_dict.get('base_scene', real_A)
            weather_effect = fake_B_dict['weather_effect']
            fake_B_dict['weather_effect'] = weather_effect * intensity_factor
            fake_B_dict['output'] = base_scene * (1 - intensity_factor) + fake_B_dict['output'] * intensity_factor

        fake_B = fake_B_dict["output"]

        # 重建A域图像
        rec_A_dict = self.G_B2A(fake_B, sem_feature_A)
        rec_A = rec_A_dict["output"]

        # 获取B域的语义特征
        sem_feature_B, _ = self.semantic_encoder(real_B)

        # 生成B→A结果
        fake_A_dict = self.G_B2A(real_B, sem_feature_B)
        fake_A = fake_A_dict["output"]

        # 重建B域图像
        rec_B_dict = self.G_A2B(fake_A, sem_feature_B)
        rec_B = rec_B_dict["output"]

        # 语义重建
        sem_hat_A = self.semantic_decoder(sem_feature_A)
        sem_hat_B = self.semantic_decoder(sem_feature_B)

        # 判别器输出
        D_real_A = self.D_A(real_A)
        D_fake_A = self.D_A(fake_A.detach())
        D_real_B = self.D_B(real_B)
        D_fake_B = self.D_B(fake_B.detach())

        return {
            'fake_B': fake_B,
            'rec_A': rec_A,
            'fake_A': fake_A,

            'rec_B': rec_B,
            'sem_hat_A': sem_hat_A,
            'sem_hat_B': sem_hat_B,
            'D_real_A': D_real_A,
            'D_fake_A': D_fake_A,
            'D_real_B': D_real_B,
            'D_fake_B': D_fake_B,
            'sem_feature_A': sem_feature_A,
            'sem_feature_B': sem_feature_B,
            'fake_B_dict': fake_B_dict,
            'fake_A_dict': fake_A_dict  # 添加解耦生成器的输出
        }

    def compute_semantic_loss(self, features, sem_pred, targets, epoch):
        """计算语义损失（包含主损失、辅助损失和一致性损失）"""
        # 如果还没有初始化辅助解码器，创建它们
        if not hasattr(self, 'auxiliary_decoders'):
            self.auxiliary_decoders = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 6, 1)
                ).to(self.device),
                nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 6, 1)
                ).to(self.device)
            ])

        # 主解码器损失（边界感知）
        main_loss = self._compute_main_loss(sem_pred, targets)

        # 辅助解码器损失
        aux_loss = 0
        for i, aux_decoder in enumerate(self.auxiliary_decoders):
            # 所有辅助头输出尺寸与输入特征图相同
            target_size = features.shape[2:]

            # 调整目标尺寸
            target_resized = F.interpolate(
                targets.unsqueeze(1).float(),
                size=target_size,
                mode='nearest'
            ).squeeze(1).long()

            # 辅助解码器预测
            aux_pred = aux_decoder(features)

            # 检查尺寸并调整（确保安全）
            if aux_pred.shape[2:] != target_resized.shape[1:]:
                aux_pred = F.interpolate(
                    aux_pred,
                    size=target_resized.shape[1:],
                    mode='bilinear',
                    align_corners=False
                )

            # 辅助损失（权重随训练衰减）
            weight = max(0.3, 1.0 - epoch * 0.002)  # 更平缓的衰减
            aux_loss += weight * F.cross_entropy(aux_pred, target_resized)

        # 特征一致性损失（简化实现）
        feat_consistency_loss = 0.1 * F.mse_loss(features, features.detach())

        # 总语义损失
        total_sem_loss = (
                main_loss +
                0.3 * aux_loss +
                feat_consistency_loss
        )

        return total_sem_loss, {
            'main': main_loss.item(),
            'aux': aux_loss.item(),
            'consistency': feat_consistency_loss.item()
        }

    def _compute_main_loss(self, pred, target):
        """边界感知的主损失函数"""
        # 基础交叉熵损失
        ce_loss = F.cross_entropy(pred, target)

        # 边界权重图
        with torch.no_grad():
            # 计算目标语义图的边界
            target_edges = self._compute_edges(target.unsqueeze(1).float())
            edge_weights = 1.0 + 2.0 * target_edges  # 边界区域权重加倍

        # 边界加权的损失
        weighted_loss = F.cross_entropy(pred, target, reduction='none')
        weighted_loss = (weighted_loss * edge_weights).mean()

        return 0.7 * ce_loss + 0.3 * weighted_loss

    def _compute_edges(self, sem_map):
        """计算语义边界"""
        # 创建Sobel滤波器（如果尚未创建）
        if not hasattr(self, 'edge_weights'):
            sobel_x = torch.tensor([[[1, 0, -1], [2, 0, -2], [1, 0, -1]]],
                                   dtype=torch.float32, device=self.device).repeat(3, 1, 1, 1)
            sobel_y = torch.tensor([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]],
                                   dtype=torch.float32, device=self.device).repeat(3, 1, 1, 1)
            self.edge_weights = torch.cat([sobel_x, sobel_y], dim=0)

        # 计算边界
        grad_x = F.conv2d(sem_map, self.edge_weights[0:1], padding=1)
        grad_y = F.conv2d(sem_map, self.edge_weights[1:2], padding=1)
        edge_map = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        return edge_map

    def compute_generator_loss(self, real_A, real_B, sem_A, sem_B, outputs, epoch, total_epochs):
        """修正的生成器损失计算-新增雾效物理约束"""
        # 对抗损失
        D_fake_B = self.D_B(outputs['fake_B'])
        loss_G_A2B = self.criterion_gan(D_fake_B, torch.ones_like(D_fake_B))

        D_fake_A = self.D_A(outputs['fake_A'])
        loss_G_B2A = self.criterion_gan(D_fake_A, torch.ones_like(D_fake_A))

        loss_adv = loss_G_A2B + loss_G_B2A

        # 循环一致性损失
        loss_cycle_A = self.criterion_cycle(outputs['rec_A'], real_A)
        loss_cycle_B = self.criterion_cycle(outputs['rec_B'], real_B)
        loss_cycle = loss_cycle_A + loss_cycle_B

        # 创建天气区域掩码
        weather_mask_A = (sem_A == 3) | (sem_A == 4)  # 天空和动态物体
        weather_mask_B = (sem_B == 3) | (sem_B == 4)

        # 语义损失（应用天气掩码）
        sem_feature_fake_B, _ = self.semantic_encoder(outputs['fake_B'])
        sem_hat_fake_B = self.semantic_decoder(sem_feature_fake_B)
        loss_seg_A2B = self.semantic_decoder.compute_loss(
            sem_hat_fake_B, sem_A, weather_mask=weather_mask_A
        )

        sem_feature_fake_A, _ = self.semantic_encoder(outputs['fake_A'])
        sem_hat_fake_A = self.semantic_decoder(sem_feature_fake_A)
        loss_seg_B2A = self.semantic_decoder.compute_loss(
            sem_hat_fake_A, sem_B, weather_mask=weather_mask_B
        )

        loss_sem = loss_seg_A2B + loss_seg_B2A

        # 动态损失权重（针对雾天特殊调整）
        if self.weather_type == "fog":
            w_adv = 0.8  # 降低对抗损失影响
            w_cycle = 6  # 稍降低循环损失
            w_sem = max(0.5, 1.0 - epoch * 0.01)  # 逐步降低语义损失权重
            w_physics = min(3.0, 0.1 * epoch)  # 逐步增加物理约束
        else:
            w_adv = 1.0
            w_cycle = 8
            w_sem = 1.0
            w_physics = min(2.0, 0.05 * epoch)

        # 雾效可见性损失（确保雾效不过弱）
        fog_visibility = torch.mean(torch.abs(outputs['fake_B_dict']['weather_effect']))
        vis_loss = torch.relu(0.1 - fog_visibility) * 0.5

        # 使用新的键名访问物理损失
        physics_loss_B = outputs['fake_B_dict']['physics_loss']

        # 总损失
        loss_G = (w_adv * loss_adv +
                  w_cycle * loss_cycle +
                  w_sem * loss_sem +
                  w_physics * physics_loss_B +  # 物理约束损失
                  vis_loss)

        return loss_G, {
            'adv': w_adv * loss_adv.item(),
            'cycle': w_cycle * loss_cycle.item(),
            'sem': w_sem * loss_sem.item(),
            'physics': w_physics * physics_loss_B.item(),
            'visibility': vis_loss.item(),
            'w_adv': w_adv,
            'w_cycle': w_cycle,
            'w_sem': w_sem,
            'w_physics': w_physics
        }

    def compute_physics_loss(self, fake_dict, fake_img):
        """物理约束损失计算 - 按天气类型定制"""
        weather_type = fake_dict['weather_type']  # 从生成器获取天气类型

        # 基础损失项
        base_loss = torch.tensor(0.0, device=fake_img.device)

        # 通用约束：天气效果应与场景语义一致
        weather_effect = fake_dict['weather_effect']
        region_mask = fake_dict['region_mask']

        # 1. 雨/雪天气约束：雨雪主要出现在天空区域
        if weather_type in ["rain", "snow"]:
            in_sky = weather_effect * region_mask[:, 0:1]
            in_non_sky = weather_effect * (1 - region_mask[:, 0:1])
            sky_violation = torch.mean(in_non_sky) / (torch.mean(in_sky) + 1e-6)
            base_loss += sky_violation

        # 2. 雾天约束：雾效深度一致性
        # 雾天物理约束 - 完全重写
        if weather_type == "fog":
            try:
                fog_layer = self.G_A2B.weather_layer
                base_scene = fake_dict.get('base_scene', fake_img).detach()

                # 获取深度图
                depth_map = fog_layer.fog_generator.depth_predictor(base_scene)

                # 深度平滑约束
                depth_grad_x = torch.abs(depth_map[:, :, :, 1:] - depth_map[:, :, :, :-1])
                depth_grad_y = torch.abs(depth_map[:, :, 1:, :] - depth_map[:, :, :-1, :])
                smooth_loss = (depth_grad_x.mean() + depth_grad_y.mean()) * 0.2

                # 大气光一致性约束
                L_pred = fog_layer.fog_generator.light_predictor(base_scene)
                sky_mask = fake_dict['region_mask'][:, 0:1]
                sky_pixels = base_scene * sky_mask
                avg_sky_color = torch.sum(sky_pixels, dim=(2, 3)) / (torch.sum(sky_mask, dim=(2, 3)) + 1e-6)
                light_loss = F.l1_loss(L_pred, avg_sky_color.unsqueeze(2).unsqueeze(3))

                base_loss += smooth_loss + 0.5 * light_loss

            except Exception as e:
                print(f"物理损失计算错误: {e}")
                base_loss = torch.tensor(0.0, device=fake_img.device)

        # 3. 夜晚约束：人造光源应出现在合理位置
        # 夜晚约束
        if weather_type == "night":
            # 仅保留最核心的边界一致性损失
            base_scene = fake_dict.get('base_scene', fake_img)

            night_adjuster = self.G_A2B.weather_layer.night_adjuster
            edge_mask = night_adjuster._detect_edges(fake_img)
            edge_diff = torch.abs(fake_img - base_scene) * edge_mask
            edge_loss = torch.mean(edge_diff) * 5.0  # 适当放大损失权重

            # 2. 主要光源位置约束（简化版）
            light_mask = (fake_img > 0.7).float()
            region_mask = fake_dict['region_mask']
            sky_mask = region_mask[:, 0:1]
            invalid_light = light_mask * sky_mask  # 天空区域不应有强光源
            light_loss = torch.mean(invalid_light) * 3.0

            base_loss += edge_loss + light_loss
            gray = torch.mean(fake_img, dim=1, keepdim=True)  # [B, 1, H, W]

            # 使用安全切片确保维度一致
            # 水平梯度 (x方向)
            grad_x = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
            # 垂直梯度 (y方向)
            grad_y = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])

            # 填充梯度图以匹配原始尺寸
            grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')
            grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')

            # 计算清晰度损失
            sharpness_loss = -torch.mean(grad_x + grad_y) * 0.5  # 负号表示最大化梯度
            base_loss += sharpness_loss
        # 4. 雪天约束：雪片应具有运动模糊效果
        if weather_type == "snow":
            snow_effect = weather_effect
            # 转换为灰度图 (保留梯度计算图)
            gray_coeffs = torch.tensor([0.2989, 0.5870, 0.1140],
                                       device=snow_effect.device).view(1, 3, 1, 1)
            gray = torch.sum(snow_effect * gray_coeffs, dim=1, keepdim=True)  # [B, 1, H, W]

            # 计算灰度图的梯度
            sobel_x_kernel = torch.tensor([[[[-1.0, 0.0, 1.0],
                                             [-2.0, 0.0, 2.0],
                                             [-1.0, 0.0, 1.0]]]],
                                          device=snow_effect.device)
            sobel_y_kernel = torch.tensor([[[[1.0, 2.0, 1.0],
                                             [0.0, 0.0, 0.0],
                                             [-1.0, -2.0, -1.0]]]],
                                          device=snow_effect.device)

            grad_x = F.conv2d(gray, sobel_x_kernel, padding=1)
            grad_y = F.conv2d(gray, sobel_y_kernel, padding=1)
            gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

            # 雪片应该有明显的运动模糊（梯度较低）
            sharp_snow_loss = torch.mean(gradient_magnitude)
            base_loss += sharp_snow_loss

        return base_loss

    def compute_discriminator_loss(self, outputs):
        """计算判别器损失"""
        # A域判别器损失
        loss_D_real_A = self.criterion_gan(outputs['D_real_A'],
                                           torch.ones_like(outputs['D_real_A']))
        loss_D_fake_A = self.criterion_gan(outputs['D_fake_A'],
                                           torch.zeros_like(outputs['D_fake_A']))
        loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5

        # B域判别器损失
        loss_D_real_B = self.criterion_gan(outputs['D_real_B'],
                                           torch.ones_like(outputs['D_real_B']))
        loss_D_fake_B = self.criterion_gan(outputs['D_fake_B'],
                                           torch.zeros_like(outputs['D_fake_B']))
        loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5

        return loss_D_A + loss_D_B

    def get_semantic_features(self, image):
        """获取图像的语义特征"""
        with torch.no_grad():
            semantic, _ = self.semantic_encoder(image)  # 只返回主特征
        return semantic

    def generate_A_to_B(self, real_A, semantic=None):
        """从A域生成B域图像"""
        if semantic is None:
            semantic = self.get_semantic_features(real_A)



        return self.G_A2B(real_A, semantic)['output']

    def generate_B_to_A(self, real_B, semantic=None):
        """从B域生成A域图像"""
        if semantic is None:
            semantic = self.get_semantic_features(real_B)
        return self.G_B2A(real_B, semantic)['output']

    def compute_spatial_consistency(self, fake_img, sem_feat, sem_target):
        """空间一致性损失(关键创新)"""
        # 1. 获取生成图像的语义特征
        features = self.semantic_encoder(fake_img)[0]  # [B, 128, 32, 32]

        # 2. 使用语义解码器得到分割预测
        pred_sem = self.semantic_decoder(features)  # [B,6,H,W]

        # 获取预测的空间尺寸
        pred_size = pred_sem.shape[2:]

        # 3. 调整目标语义图分辨率以匹配预测
        target_down = F.interpolate(
            sem_target.unsqueeze(1).float(),
            size=pred_size,
            mode='nearest'
        ).squeeze(1).long()  # [B, H_pred, W_pred]

        # 4. 空间对齐损失(在特征空间)
        grid_loss = F.l1_loss(features, sem_feat)

        # 5. 语义边界损失
        edge_mask = self.compute_semantic_edges(sem_target)  # 原始分辨率

        # 下采样边界掩码以匹配分割预测的分辨率
        edge_mask_down = F.interpolate(
            edge_mask.unsqueeze(1),
            size=pred_size,
            mode='bilinear',
        )

        # 边界加权损失
        edge_loss = F.cross_entropy(pred_sem, target_down, reduction='none')
        edge_loss = (edge_loss * edge_mask_down).mean()

        return 0.7 * grid_loss + 0.3 * edge_loss

    def compute_semantic_edges(self, sem_map):
        """计算语义边界(辅助函数)"""
        # 创建 Sobel滤波器
        if not hasattr(self, 'sobel_x'):
            self.sobel_x = torch.tensor([
                [[[1, 0, -1],
                  [2, 0, -2],
                  [1, 0, -1]]]
            ], dtype=torch.float32, device=sem_map.device)

            self.sobel_y = torch.tensor([
                [[[1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]]]
            ], dtype=torch.float32, device=sem_map.device)

        # 确保输入是4维[B, C, H, W]
        if sem_map.dim() == 3:
            sem_map = sem_map.unsqueeze(1)  # 增加通道维度

        # 将语义图转换为浮点型
        sem_map = sem_map.float()

        # 计算边界
        grad_x = F.conv2d(sem_map, self.sobel_x, padding=1)
        grad_y = F.conv2d(sem_map, self.sobel_y, padding=1)

        # 计算梯度幅度
        edges = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        return edges.squeeze(1)  # 移除通道维度[B, H, W]

    def conditional_cycle_loss(self, rec, real, sem_map):
        """只在语义边界区域应用强约束"""
        # 生成语义边界掩码
        edges = self.sobel_edge_detection(sem_map)

        # 非边界区域使用宽松约束
        non_edge_loss = F.l1_loss(rec, real) * 0.1

        # 边界区域使用强约束
        edge_mask = (edges > 0.5).float()
        edge_loss = F.l1_loss(rec * edge_mask, real * edge_mask) * 2.0

        return non_edge_loss + edge_loss

    # 新增Sobel边缘检测
    def sobel_edge_detection(self, sem_map):
        """计算语义边界"""
        # 处理输入维度问题
        if sem_map.dim() == 5:
            # 如果是5维张量 [B, 1, 1, H, W]，压缩为4维 [B, 1, H, W]
            sem_map = sem_map.squeeze(1)  # 移除第2个维度
            if sem_map.dim() == 4:
                sem_map = sem_map.squeeze(1)  # 移除第1个维度
        elif sem_map.dim() != 4:
            raise ValueError(f"Expected 4D input (B, C, H, W), got {sem_map.dim()}D")

        # 确保语义图有正确的通道维度
        if sem_map.size(1) != 1:
            # 转换为灰度图
            gray_coeffs = torch.tensor([[0.2989, 0.5870, 0.1140]],
                                       device=sem_map.device).view(1, 3, 1, 1)
            gray = torch.sum(sem_map * gray_coeffs, dim=1, keepdim=True)  # [B, 1, H, W]
        else:
            gray = sem_map

        # 确保Sobel滤波器存在
        if not hasattr(self, 'sobel_x'):
            # 创建Sobel滤波器
            sobel_x = torch.tensor([[[[1.0, 0.0, -1.0],
                                      [2.0, 0.0, -2.0],
                                      [1.0, 0.0, -1.0]]]],
                                   dtype=torch.float32, device=sem_map.device)
            sobel_y = torch.tensor([[[[1.0, 2.0, 1.0],
                                      [0.0, 0.0, 0.0],
                                      [-1.0, -2.0, -1.0]]]],
                                   dtype=torch.float32, device=sem_map.device)
            # 注册为缓冲区
            self.register_buffer('sobel_x', sobel_x)
            self.register_buffer('sobel_y', sobel_y)

        # 应用滤波器
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)

        # 计算梯度幅度
        edge_map = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        return edge_map


# ====================== 判别器定义 ======================
class Discriminator(nn.Module):
    def __init__(self, input_nc=3):
        super().__init__()
        # 单一特征提取器
        self.feature_extractor = self._create_feature_extractor(input_nc)

        # 多尺度处理模块
        self.scales = nn.ModuleList([
            self._create_scale(256),
            self._create_scale(256),
        ])

        # 下采样层
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1)

        # 特征金字塔融合
        self.fpn = nn.ModuleList([
            nn.Conv2d(256, 64, 1) for _ in range(2)  # 通道数 128->64
        ])

        # 最终分类层
        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),  # 输入通道从384改为128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 4, padding=1)
        )

    def _create_feature_extractor(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=1, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def _create_scale(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        # 提取基础特征
        base_features = self.feature_extractor(x)

        # 多尺度处理(现在只有两个尺度)
        feat0 = base_features
        feat1 = self.downsample(feat0)  # 移除feat2

        # 处理两个尺度
        out0 = self.scales[0](feat0)
        out1 = self.scales[1](feat1)

        # 特征金字塔融合(也减少为两个)
        out0 = self.fpn[0](out0)
        out1 = self.fpn[1](out1)

        # 上采样到统一尺寸
        out1 = F.interpolate(out1, size=out0.shape[2:], mode='bilinear', align_corners=False)

        # 拼接特征(现在是两个特征图)
        combined = torch.cat([out0, out1], dim=1)  # 原来是三个

        return self.final_conv(combined)


# ====================== 辅助模块 ======================
class NoiseSuppression(nn.Module):
    """生成器末端噪声抑制模块"""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, x):
        # 分离信号和噪声
        signal = self.conv(x)
        noise = x - signal
        # 抑制高频噪声
        noise = noise * 0.3  # 降低噪声幅度
        return signal + noise


class ChannelAttention(nn.Module):
    """通道注意力机制"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return x * out.view(b, c, 1, 1)


class SpatialSemanticAlignment(nn.Module):
    def __init__(self, img_channels, sem_channels):
        super().__init__()
        # 特征对齐卷积
        self.align_conv = nn.Sequential(
            nn.Conv2d(sem_channels, img_channels, 3, padding=1),
            nn.GroupNorm(8, img_channels),
            nn.LeakyReLU(0.2)
        )

        # 门控融合模块
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(img_channels * 2, img_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(img_channels // 8, img_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, img_feat, sem_feat):
        # 动态调整语义特征分辨率
        sem_feat = F.interpolate(
            sem_feat,
            size=img_feat.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        # 特征转换
        sem_trans = self.align_conv(sem_feat)

        # 门控融合
        gate = self.fusion_gate(torch.cat([img_feat, sem_trans], dim=1))
        fused = gate * img_feat + (1 - gate) * sem_trans

        return img_feat + fused  # 残差连接


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.norm = nn.InstanceNorm2d(num_features, affine=False, eps=eps)

        # 使用全连接层替代卷积层,避免空间维度问题
        self.style_scale = nn.Linear(num_features, num_features)
        self.style_shift = nn.Linear(num_features, num_features)

        # 初始化参数
        self.style_scale.weight.data = torch.eye(num_features)
        self.style_scale.bias.data.zero_()
        self.style_shift.weight.data = torch.eye(num_features)
        self.style_shift.bias.data.zero_()

    def forward(self, x, style):
        # 标准化内容特征
        normalized = self.norm(x)

        # 确保风格特征形状正确
        if style.size(1) != self.num_features:
            raise ValueError(f"Style channels {style.size(1)} != {self.num_features}")

        # 计算风格统计量
        style_pooled = F.adaptive_avg_pool2d(style, (1, 1)).squeeze(-1).squeeze(-1)  # [B, C]

        # 应用风格变换
        scale = self.style_scale(style_pooled) + 1.0
        shift = self.style_shift(style_pooled)

        # 重塑为特征图维度
        scale = scale.view(-1, self.num_features, 1, 1)
        shift = shift.view(-1, self.num_features, 1, 1)

        return scale * normalized + shift