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

# ======================= 新增物理约束模块 =======================
class RaindropSimulator(nn.Module):
    """可微分雨滴物理模型"""

    def __init__(self):
        super().__init__()
        # 雨滴参数预测
        self.param_predictor = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 2, 1)  # [大小，密度]
        )
        # 初始化风场（可改为可学习参数）
        self.register_buffer('wind_field', torch.tensor([0.5, 0.5]))

    def forward(self, x, intensity):
        # 确保intensity有正确的维度 [B, 1, 1, 1]
        intensity = intensity.view(-1, 1, 1, 1)

        # 生成雨滴参数
        params = self.param_predictor(x)  # [B, 2, H, W]

        # 计算雨滴大小和密度
        drop_size = torch.sigmoid(params[:, 0:1]) * 3 + 1  # 1-4像素
        density = torch.sigmoid(params[:, 1:2]) * 0.5 * intensity  # 密度受强度控制

        # 创建雨滴噪声
        noise = torch.randn_like(x) * density

        # 运动模糊模拟（使用平均雨滴大小）
        avg_drop_size = drop_size.mean(dim=[2, 3], keepdim=True)  # [B, 1, 1, 1]
        motion_kernel = self._create_motion_kernel(avg_drop_size)

        # 应用运动模糊（分组卷积处理通道）
        rain_effect = F.conv2d(
            noise,
            motion_kernel.expand(noise.size(1), -1, -1, -1),
            padding='same',
            groups=noise.size(1)
        )

        return rain_effect * intensity

    def _create_motion_kernel(self, drop_size):
        # 简化计算提高精度
        kernel_size = min(15, max(3, int(drop_size.mean().item() * 1.2)))

        # 使用高斯分布替代线性计算
        kernel = torch.zeros(1, 1, kernel_size, kernel_size, device=drop_size.device)
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                dist = math.sqrt((i - center) ** 2 + (j - center) ** 2)
                kernel[0, 0, i, j] = math.exp(-dist ** 2 / (0.5 * center ** 2))

        return kernel / (kernel.sum() + 1e-6)


class FogGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_predictor = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # 移除下采样
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 1),  # 保持输出层
            nn.Sigmoid()
        )
        self.light_predictor = nn.Sequential(  # 动态预测大气光强
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3, 8, 1),
            nn.ReLU(),
            nn.Conv2d(8, 3, 1),
            nn.Tanh()  # 输出范围[-1,1]
        )

    def forward(self, x, intensity_map):  # 修改参数为强度图而非标量
        depth_map = self.depth_predictor(x)
        beta = 0.1 + intensity_map * 2.0
        transmission = torch.exp(-beta * depth_map)
        L = self.light_predictor(x) * 0.5 + 0.8
        fog_effect = x * transmission + L * (1 - transmission)
        return fog_effect  # 移除硬截断





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


class SpatialAttention(nn.Module):
    """空间注意力机制"""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        return x * attention


class UnifiedSkyProcessor(nn.Module):
    """统一天空处理器，确保整个天空区域一致转换"""

    def __init__(self):
        super().__init__()
        # 使用更精确的天空检测网络
        self.sky_detector = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, img, night_effect):
        # 检测天空区域
        sky_prob = self.sky_detector(img)

        # 使用形态学操作填充天空区域的小空洞
        kernel = torch.ones(1, 1, 5, 5, device=img.device)
        sky_prob_dilated = F.conv2d(sky_prob, kernel, padding=2) > 0.1
        sky_prob_eroded = F.conv2d(sky_prob, kernel, padding=2) > 0.9

        # 最终天空掩码：扩张区域减去腐蚀区域，确保连续性
        sky_mask = (sky_prob_dilated.float() - sky_prob_eroded.float()).clamp(0, 1)

        # 确保天空区域有统一的夜晚效果
        sky_region_night = night_effect * sky_mask

        # 计算天空区域的平均夜晚色调
        sky_pixels = sky_region_night * sky_mask
        num_sky_pixels = torch.sum(sky_mask, dim=[1, 2, 3], keepdim=True) + 1e-6
        avg_sky_color = torch.sum(sky_pixels, dim=[1, 2, 3], keepdim=True) / num_sky_pixels

        # 用平均色调统一整个天空区域
        unified_sky = avg_sky_color * sky_mask

        # 将统一后的天空与原始图像的非天空区域合并
        non_sky_mask = 1 - sky_mask
        result = unified_sky + night_effect * non_sky_mask

        return result


class EdgeAwareNightConverter(nn.Module):
    """边缘感知的夜晚转换器，解决边缘亮光问题"""

    def __init__(self):
        super().__init__()

    def forward(self, img, semantic_mask, night_intensity):
        # 检测图像边缘
        gray = torch.mean(img, dim=1, keepdim=True)

        # Sobel边缘检测
        sobel_x = torch.tensor([[[1, 0, -1], [2, 0, -2], [1, 0, -1]]],
                               device=img.device).float().unsqueeze(0)
        sobel_y = torch.tensor([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]],
                               device=img.device).float().unsqueeze(0)

        grad_x = F.conv2d(gray, sobel_x, padding=1, stride=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1, stride=1)
        edge_strength = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        # 创建边缘区域掩码
        edge_mask = (edge_strength > 0.1).float()

        # 关键修复：在边缘区域应用渐进式暗化，而不是保护
        # 简化的距离场计算
        batch_size, _, height, width = edge_mask.shape
        distance_field = torch.ones_like(edge_mask)

        # 使用卷积近似距离场
        for i in range(3):  # 多次迭代以获得更好的距离场
            kernel = torch.ones(1, 1, 3, 3, device=img.device) / 9.0
            distance_field = F.conv2d(distance_field, kernel, padding=1)
            # 在边缘处重置为0
            distance_field[edge_mask > 0.5] = 0

        # 边缘区域的暗化强度随距离递减
        edge_darkening = 1.0 - distance_field * 0.8
        edge_darkening = torch.clamp(edge_darkening, 0.3, 1.0)

        # 应用边缘感知的夜晚效果
        base_night_effect = img * (1 - night_intensity * 0.7)
        edge_aware_effect = base_night_effect * edge_darkening

        return edge_aware_effect


class ConsistentNightConverter(nn.Module):
    """一致的夜晚转换器，解决语义与物理模型的冲突"""

    def __init__(self):
        super().__init__()
        self.sky_processor = UnifiedSkyProcessor()
        self.edge_converter = EdgeAwareNightConverter()

    def forward(self, img, semantic_features, region_mask):
        # 基础夜晚效果
        base_intensity = 0.7
        base_night = img * (1 - base_intensity)

        # 统一处理天空区域
        sky_unified = self.sky_processor(img, base_night)

        # 边缘感知转换
        edge_aware = self.edge_converter(sky_unified, region_mask, base_intensity)

        # 最终一致性检查：确保没有明显的分界线
        final_result = self._ensure_consistency(edge_aware, region_mask)

        return final_result

    def _ensure_consistency(self, img, region_mask):
        """确保转换结果的一致性，消除明显分界线"""
        # 应用高斯模糊平滑过渡区域
        kernel_size = 11
        sigma = 3.0

        # 创建高斯核
        x = torch.arange(kernel_size, device=img.device) - kernel_size // 2
        y = torch.arange(kernel_size, device=img.device) - kernel_size // 2
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size)

        # 对每个通道应用高斯模糊
        blurred = torch.zeros_like(img)
        for c in range(img.shape[1]):
            channel_img = img[:, c:c + 1, :, :]
            blurred_channel = F.conv2d(channel_img, kernel, padding=kernel_size // 2)
            blurred[:, c:c + 1, :, :] = blurred_channel

        # 只在过渡区域使用模糊结果
        transition_mask = self._detect_transition_regions(region_mask)
        smooth_result = img * (1 - transition_mask) + blurred * transition_mask

        return smooth_result

    def _detect_transition_regions(self, region_mask):
        """检测不同区域之间的过渡区域"""
        # 计算每个区域的边界
        boundary_mask = torch.zeros_like(region_mask[:, 0:1, :, :])

        # 对每个区域类别检测边界
        for i in range(region_mask.shape[1]):
            region = region_mask[:, i:i + 1, :, :]

            # 使用Sobel检测区域边界
            sobel_x = torch.tensor([[[1, 0, -1], [2, 0, -2], [1, 0, -1]]],
                                   device=region_mask.device).float().unsqueeze(0)
            sobel_y = torch.tensor([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]],
                                   device=region_mask.device).float().unsqueeze(0)

            grad_x = F.conv2d(region, sobel_x, padding=1, stride=1)
            grad_y = F.conv2d(region, sobel_y, padding=1, stride=1)
            region_boundary = torch.sqrt(grad_x ** 2 + grad_y ** 2) > 0.1

            boundary_mask = torch.clamp(boundary_mask + region_boundary.float(), 0, 1)

        # 扩张边界区域以创建过渡区域
        kernel = torch.ones(1, 1, 5, 5, device=region_mask.device)
        transition_region = F.conv2d(boundary_mask, kernel, padding=2) > 0.1

        return transition_region.float()


class AggressiveNightAdjuster(nn.Module):
    """修复的夜晚调整器-确保颜色和亮度正确"""

    def __init__(self, semantic_dim):
        super().__init__()
        # 深度预测
        self.depth_predictor = nn.Sequential(
            nn.Conv2d(3 + semantic_dim, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

        # 全局光照预测-修复颜色问题
        self.global_light_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 1),
            nn.Tanh()
        )

        # 颜色变换-确保夜晚色调正确
        self.color_transformer = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )

        # 夜间特征增强
        self.night_enhancer = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x, semantic_features, intensity_map):
        # 修复: 确保语义特征分辨率匹配
        if semantic_features.shape[2:] != x.shape[2:]:
            semantic_features = F.interpolate(
                semantic_features, size=x.shape[2:],
                mode='bilinear', align_corners=False
            )

        # 深度预测
        depth_input = torch.cat([x, semantic_features], dim=1)
        depth_map = self.depth_predictor(depth_input)

        # 关键修复: 恢复原始的传输率计算
        beta = 0.1 + intensity_map * 1.5  # 恢复原始系数
        transmission = torch.exp(-beta * depth_map * 5.0)  # 恢复原始乘数

        # 关键修复: 恢复原始传输率范围
        transmission = torch.clamp(transmission, 0.1, 0.8)  # 恢复原始范围

        # 关键修复: 恢复原始全局光照
        global_light = self.global_light_predictor(x) * 0.5 + 0.8  # 恢复原始设置

        # 基础夜间效果-使用原始公式
        base_night = x * transmission + global_light * (1 - transmission)

        # 颜色变换-恢复原始强度
        color_shift = self.color_transformer(base_night) * 0.3  # 恢复原始强度
        night_effect = base_night + color_shift

        # 夜间特征增强
        night_effect = self.night_enhancer(night_effect)

        # 关键修复: 恢复原始混合策略
        final_effect = x * (1 - intensity_map) + night_effect * intensity_map

        # 最终数值检查
        if torch.isnan(final_effect).any():
            print("警告: 最终效果包含NaN，使用备用方案")
            final_effect = torch.clamp(x * 0.7, 0, 1)  # 温和的暗化方案

        return final_effect


class PhysicsInformedNightLayer(nn.Module):
    """完全重写的夜晚物理层-确保全局效果"""

    def __init__(self, semantic_dim=128):
        super().__init__()
        # 强度预测器-直接预测全局强度
        self.intensity_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(semantic_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        # 使用原始的夜晚调整器
        self.night_adjuster = AggressiveNightAdjuster(semantic_dim)

    def forward(self, base_scene, semantic_features):
        # 修复: 确保强度预测正确形状
        night_intensity = self.intensity_predictor(semantic_features)
        # 修复: 彻底检查形状处理
        if night_intensity.dim() == 2:
            night_intensity = night_intensity.squeeze(1)
        # 修复: 正确的维度扩展
        night_intensity = night_intensity.view(-1, 1, 1, 1)  # [B,1,1,1]

        # 关键修复: 恢复原始强度范围，避免过度转换
        night_intensity = torch.clamp(night_intensity, 0.1, 0.9)  # 恢复原始范围

        # 应用夜晚效果
        night_effect = self.night_adjuster(base_scene, semantic_features, night_intensity)
        return night_effect, night_intensity, torch.zeros_like(night_intensity)


class RegionAwareFusion(nn.Module):
    """修复区域感知融合模块-恢复原始逻辑"""

    def forward(self, base_scene, weather_effect, region_mask, weather_type="night"):
        # 关键修复: 对于夜晚，恢复原始的区域感知融合逻辑
        if weather_type == "night":
            # 恢复原始的区域融合逻辑
            if region_mask.size(1) >= 1:
                sky_mask = region_mask[:, 0:1]
            else:
                sky_mask = torch.ones_like(base_scene[:, :1, :, :])

            # 确保天空掩码与图像尺寸匹配
            if sky_mask.shape[2:] != base_scene.shape[2:]:
                sky_mask = F.interpolate(
                    sky_mask, size=base_scene.shape[2:],
                    mode='bilinear', align_corners=False
                )

            # 恢复原始的天空区域处理
            non_sky_mask = 1 - sky_mask
            weather_non_sky = weather_effect * non_sky_mask
            sky_effect = base_scene * sky_mask * 0.7 + weather_effect * sky_mask * 0.3

            return sky_effect + weather_non_sky

        # 其他天气类型的原有逻辑保持不变
        if region_mask.size(1) != 1 and region_mask.size(1) == 3:
            sky_mask = region_mask[:, 0:1]
        else:
            sky_mask = region_mask

        if sky_mask.shape[2:] != base_scene.shape[2:]:
            sky_mask = F.interpolate(
                sky_mask, size=base_scene.shape[2:],
                mode='bilinear', align_corners=False
            )
        non_sky_mask = 1 - sky_mask
        weather_non_sky = weather_effect * non_sky_mask
        sky_effect = base_scene * sky_mask * 0.7 + weather_effect * sky_mask * 0.3
        return sky_effect + weather_non_sky


class EnhancedSceneParser(nn.Module):
    """增强型场景解析器，更准确地区分不同区域"""

    def __init__(self, semantic_dim):
        super().__init__()
        self.region_classifier = nn.Sequential(
            nn.Conv2d(semantic_dim, 256, 3, padding=1),
            nn.ReLU(),
            DetailEnhanceBlock(256),  # 添加细节增强
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            EdgeAwareBlock(128),  # 添加边缘感知
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 5, 1),  # 增加类别数量：天空、建筑、地面、植被、其他
            nn.Softmax(dim=1)
        )

    def forward(self, semantic_features):
        return self.region_classifier(semantic_features)






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

    def __init__(self, input_nc=3, output_nc=3, semantic_dim=128, weather_type="night"):
        super().__init__()
        self.weather_type = weather_type

        # 基础场景生成器
        self.base_generator = BaseSceneGenerator(input_nc, output_nc, semantic_dim)

        # 物理天气生成层
        if weather_type == "night":
            self.weather_layer = PhysicsInformedNightLayer(semantic_dim)
        elif weather_type == "rain":
            self.weather_layer = PhysicsInformedRainLayer(semantic_dim)
        elif weather_type == "fog":
            self.weather_layer = PhysicsInformedFogLayer(semantic_dim)
        elif weather_type == "snow":
            self.weather_layer = PhysicsInformedSnowLayer(semantic_dim)
        else:
            raise ValueError(f"Unsupported weather type: {weather_type}")

        # 区域感知融合模块
        self.fusion_module = RegionAwareFusion()

        # 场景解析器
        self.scene_parser = EnhancedSceneParser(semantic_dim)

        # 上采样器
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)

    def forward(self, x, semantic_features):
        # 基础场景生成
        base_scene = self.base_generator(x, semantic_features)

        # 场景解析
        region_mask = self.scene_parser(semantic_features)

        # 物理天气生成
        weather_effect, rain_intensity, fog_intensity = self.weather_layer(base_scene, semantic_features)

        # 上采样区域掩码
        region_mask_up = self.upsample(region_mask)

        # 区域感知融合
        output = self.fusion_module(base_scene, weather_effect, region_mask_up, self.weather_type)

        # 修复：确保返回 semantic_features
        return {
            "output": output,
            "base_scene": base_scene,
            "weather_effect": weather_effect,
            "region_mask": region_mask_up,
            "rain_intensity": rain_intensity,
            "fog_intensity": fog_intensity,
            "weather_type": self.weather_type,
            "semantic_features": semantic_features  # 添加这行
        }


# 原有雨天层保持不变
class PhysicsInformedRainLayer(nn.Module):
    """物理约束雨天生成层"""

    def __init__(self, input_dim=128):
        super().__init__()
        # 天气参数预测 - 只预测雨强度
        self.weather_param_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 只输出雨强度
        )
        self.raindrop_simulator = RaindropSimulator()

    def forward(self, base_scene, semantic_features):
        rain_intensity = torch.sigmoid(self.weather_param_predictor(semantic_features))
        rain_effect = self.raindrop_simulator(base_scene, rain_intensity)
        return rain_effect, rain_intensity, torch.zeros_like(rain_intensity)  # 保持原有输出格式


# 新增雾天层
class PhysicsInformedFogLayer(nn.Module):
    def __init__(self, input_dim=128):
        super().__init__()
        # 空间感知的雾强度预测器
        self.fog_intensity_predictor = nn.Sequential(
            nn.Conv2d(input_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1),  # 1x1卷积减少计算
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        self.fog_generator = FogGenerator()

    def forward(self, base_scene, semantic_features):
        # 上采样语义特征匹配场景分辨率
        h, w = base_scene.shape[2], base_scene.shape[3]
        sem_up = F.interpolate(semantic_features, size=(h, w), mode='bilinear')

        # 预测空间变化的强度图
        fog_intensity_map = self.fog_intensity_predictor(sem_up)
        fog_effect = self.fog_generator(base_scene, fog_intensity_map)

        # 返回雾效和强度图（用于监控）
        return fog_effect, torch.zeros(1), fog_intensity_map.mean()


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
        sem_feature_A, _ = self.semantic_encoder(real_A)  # 解包元组

        # 使用解耦生成器生成结果
        fake_B_dict = self.G_A2B(real_A, sem_feature_A)
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
            'fake_B_dict': fake_B_dict,  # 添加解耦生成器的输出
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
        """修正的生成器损失计算-新增物理约束损失"""
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
        # 假设语义类别:3=天空,4=动态物体
        weather_mask_A = (sem_A == 3) | (sem_A == 4)
        weather_mask_B = (sem_B == 3) | (sem_B == 4)

        # 语义损失(应用天气掩码)
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

        # 新增物理约束损失z
        physics_loss = self.compute_physics_loss(outputs['fake_B_dict'], outputs['fake_B'])
        physics_loss = torch.tanh(physics_loss)  # 将损失映射到[-1,1]范围

        # 动态损失权重
        w_physics = 1  # 限制最大权重
        # 动态损失权重(随训练进度调整)
        w_adv = 1.0
        w_cycle = 8
        w_sem = 1.5  # 语义损失随训练衰减

        # 在训练后期加强物理约束
        w_physics = min(2.0, 0.5 + epoch / total_epochs * 1.5)
        # 降低循环一致性权重，避免过度保留白天特征
        w_cycle = max(5.0, 10.0 - epoch / total_epochs * 5.0)

        # 总损失
        loss_G = (w_adv * loss_adv +
                  w_cycle * loss_cycle +
                  w_sem * loss_sem +
                  w_physics * physics_loss)

        return loss_G, {
            'adv': w_adv * loss_adv.item(),
            'cycle': w_cycle * loss_cycle.item(),
            'sem':  w_sem * loss_sem.item(),
            'physics': w_physics * physics_loss.item(),
            'w_adv': w_adv,
            'w_cycle': w_cycle,
            'w_sem': w_sem,
            'w_physics': w_physics
        }

    def compute_physics_loss(self, fake_dict, fake_img):
        """修复物理约束损失计算-恢复原始逻辑"""
        weather_type = fake_dict.get('weather_type', 'night')
        base_loss = torch.tensor(0.0, device=fake_img.device)

        if weather_type == "night":
            region_mask = fake_dict.get('region_mask', None)
            semantic_features = fake_dict.get('semantic_features', None)

            if region_mask is None:
                return base_loss

            # 1. 天空一致性约束-恢复原始逻辑
            sky_mask = region_mask[:, 0:1]  # 假设第一个通道是天空

            # 确保天空掩码与图像尺寸匹配
            if sky_mask.shape[2:] != fake_img.shape[2:]:
                sky_mask = F.interpolate(
                    sky_mask, size=fake_img.shape[2:],
                    mode='bilinear', align_corners=False
                )

            sky_pixels = fake_img * sky_mask
            sky_brightness = torch.mean(sky_pixels)

            # 恢复原始的天空亮度约束
            darkness_loss = F.relu(sky_brightness - 0.25) * 5.0 + F.relu(0.05 - sky_brightness) * 2.0

            # 2. 车辆亮度合理性约束-恢复原始逻辑
            vehicle_loss = torch.tensor(0.0, device=fake_img.device)
            if semantic_features is not None:
                # 获取语义类别
                semantic_class = torch.argmax(semantic_features, dim=1, keepdim=True)

                # 上采样语义类别到图像尺寸
                if semantic_class.shape[2:] != fake_img.shape[2:]:
                    semantic_class = F.interpolate(
                        semantic_class.float(),
                        size=fake_img.shape[2:],
                        mode='nearest'
                    ).long()

                vehicle_mask = (semantic_class == 5).float()
                vehicle_pixels = fake_img * vehicle_mask
                vehicle_brightness = torch.mean(vehicle_pixels)

                # 恢复原始的车辆亮度约束
                vehicle_loss = F.relu(vehicle_brightness - 0.6) * 3.0 + F.relu(0.2 - vehicle_brightness) * 1.0

            # 3. 光源空间分布约束-恢复原始逻辑
            distribution_loss = torch.tensor(0.0, device=fake_img.device)
            if semantic_features is not None:
                light_sources = self._detect_light_sources(fake_img, semantic_features)
                light_distribution = torch.mean(light_sources, dim=[2, 3])

                # 恢复原始的光源分布约束
                distribution_loss = F.relu(light_distribution - 0.3) * 2.0 + F.relu(0.01 - light_distribution) * 1.0

            base_loss = darkness_loss + vehicle_loss + distribution_loss

        # 确保返回标量
        if base_loss.dim() > 0:
            base_loss = base_loss.mean()

        return base_loss

    def _detect_vehicles(self, img):
        """简单车辆检测（实际应使用语义分割）"""
        # 基于颜色和亮度的启发式方法
        r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
        is_red = (r > 0.6) & (g < 0.4) & (b < 0.4)
        is_white = (r > 0.7) & (g > 0.7) & (b > 0.7)
        return (is_red | is_white).float()

    def _detect_road_markings(self, img):
        """检测道路标记（斑马线等）- 修复维度问题"""
        # 转换为灰度 [B, 1, H, W]
        gray = torch.mean(img, dim=1, keepdim=True)

        # 修复：正确创建 Sobel 滤波器维度
        # 滤波器形状应该是 [out_channels, in_channels, H, W]
        sobel_x = torch.tensor([[[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]]], dtype=torch.float32, device=img.device)
        sobel_x = sobel_x.unsqueeze(0)  # [1, 1, 3, 3]

        sobel_y = torch.tensor([[[1, 2, 1],
                                 [0, 0, 0],
                                 [-1, -2, -1]]], dtype=torch.float32, device=img.device)
        sobel_y = sobel_y.unsqueeze(0)  # [1, 1, 3, 3]

        # 修复：显式指定 stride 参数
        grad_x = F.conv2d(gray, sobel_x, padding=1, stride=(1, 1))
        grad_y = F.conv2d(gray, sobel_y, padding=1, stride=(1, 1))

        # 检测水平线特征（斑马线）
        horizontal_lines = (torch.abs(grad_x) < 0.1) & (torch.abs(grad_y) > 0.3)
        return horizontal_lines.float()

    # 然后修改现有的_detect_light_sources方法，在返回前抑制道路标记区域的光源
    def _detect_light_sources(self, fake_img, semantic_features):
        """改进的光源检测，结合语义信息 - 修复滤波器维度"""
        if semantic_features is None:
            return torch.zeros_like(fake_img[:, :1, :, :])

        # 确保语义特征有正确的空间分辨率
        if semantic_features.shape[2:] != fake_img.shape[2:]:
            semantic_features = F.interpolate(
                semantic_features,
                size=fake_img.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        semantic_class = torch.argmax(semantic_features, dim=1, keepdim=True)

        # 只允许在特定类别上检测光源
        vehicle_mask = (semantic_class == 5).float()  # 动态物体
        building_mask = (semantic_class == 0).float()  # 建筑
        allowed_regions = vehicle_mask + building_mask

        # 使用更复杂的光源检测
        gray = torch.mean(fake_img, dim=1, keepdim=True)

        # 高亮区域检测
        bright_regions = (gray > 0.7).float()

        # 颜色特征: 寻找偏白或偏黄的光源
        r, g, b = fake_img[:, 0:1], fake_img[:, 1:2], fake_img[:, 2:3]
        white_light = ((r > 0.8) & (g > 0.8) & (b > 0.6)).float()
        yellow_light = ((r > 0.8) & (g > 0.7) & (b < 0.4)).float()
        color_condition = (white_light + yellow_light).clamp(0, 1)  # 白色或黄色光源

        light_sources = bright_regions * color_condition * allowed_regions

        # 形态学操作去除噪点
        kernel = torch.ones(1, 1, 3, 3, device=fake_img.device)
        light_sources = (F.conv2d(light_sources, kernel, padding=1, stride=(1, 1)) > 0.5).float()

        # 排除道路和标记区域
        road_mask = (semantic_class == 1).float()  # 道路
        sign_mask = (semantic_class == 2).float()  # 标志
        excluded_regions = road_mask + sign_mask

        # 检测道路标记（斑马线）并排除
        road_markings = self._detect_road_markings(fake_img)
        excluded_regions = torch.clamp(excluded_regions + road_markings, 0, 1)

        light_sources = light_sources * (1 - excluded_regions)

        return light_sources

    def _get_building_edges(self, img):
        """专用建筑边缘检测"""
        # 转换为灰度
        gray = torch.mean(img, dim=1, keepdim=True)
        # 水平梯度
        grad_x = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')
        # 垂直梯度
        grad_y = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')
        # 合并边缘
        edges = (grad_x + grad_y) > 0.3
        return edges.float()

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
        """计算语义边界 - 修复滤波器维度"""
        # 处理输入维度问题
        if sem_map.dim() == 5:
            # 如果是5维张量[B,1,1,H,W]，压缩为4维[B,1,H,W]
            sem_map = sem_map.squeeze(1)
        elif sem_map.dim() == 4:
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
            # 修复：正确创建 Sobel 滤波器
            sobel_x = torch.tensor([[[1.0, 0.0, -1.0],
                                     [2.0, 0.0, -2.0],
                                     [1.0, 0.0, -1.0]]], dtype=torch.float32, device=sem_map.device)
            sobel_x = sobel_x.unsqueeze(0)  # [1, 1, 3, 3]

            sobel_y = torch.tensor([[[1.0, 2.0, 1.0],
                                     [0.0, 0.0, 0.0],
                                     [-1.0, -2.0, -1.0]]], dtype=torch.float32, device=sem_map.device)
            sobel_y = sobel_y.unsqueeze(0)  # [1, 1, 3, 3]

            # 注册为缓冲区
            self.register_buffer('sobel_x', sobel_x)
            self.register_buffer('sobel_y', sobel_y)

        # 应用滤波器
        grad_x = F.conv2d(gray, self.sobel_x, padding=1, stride=(1, 1))
        grad_y = F.conv2d(gray, self.sobel_y, padding=1, stride=(1, 1))

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

