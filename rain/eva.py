import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
import argparse


# 定义图像数据集类，保留原始文件名
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, crop_size=1080, target_size=256, transform=None):
        """
        初始化图像数据集

        参数:
            root_dir (str): 图像文件所在的根目录
            crop_size (int): 中心裁剪尺寸
            target_size (int): 目标图像尺寸
            transform (callable, optional): 应用于图像的转换函数
        """
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.target_size = target_size
        self.transform = transform
        self.image_files = sorted([
            f for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __len__(self):
        """返回数据集中的样本数量"""
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        获取单个样本

        参数:
            idx (int): 样本索引

        返回:
            tuple: (转换后的图像张量, 原始文件名)
        """
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # 应用预处理
        # 1. 中心裁剪到指定尺寸
        width, height = image.size
        left = (width - self.crop_size) // 2
        top = (height - self.crop_size) // 2
        right = left + self.crop_size
        bottom = top + self.crop_size
        image = image.crop((left, top, right, bottom))

        # 2. 调整到目标尺寸
        image = image.resize((self.target_size, self.target_size), Image.BILINEAR)

        if self.transform:
            image = self.transform(image)

        return image, img_name  # 返回图像和原始文件名


# 反归一化函数：将[-1,1]范围的张量转换为[0,1]范围的图像
def denormalize(tensor):
    """
    将归一化的图像张量反归一化到[0,1]范围

    参数:
        tensor (tor.Tensor): 归一化的图像张量

    返回:
        torch.Tensor: 反归一化后的图像张量
    """
    return (tensor.clamp(-1, 1) + 1) / 2


# 模型加载函数
def load_model(model_path, device, weather_type="fog"):
    """
    加载训练好的模型

    参数:
        model_path (str): 模型文件路径
        device (str): 使用的设备(cuda/cpu)
        weather_type (str): 天气类型

    返回:
        SemanticCycleGAN: 加载的模型
    """
    # 导入模型类
    from model import SemanticCycleGAN

    # 创建模型实例
    model = SemanticCycleGAN(
        device=device,
        semantic_dim=128,
        weather_type=weather_type
    )
    model.to(device)

    # 加载模型参数
    checkpoint = torch.load(model_path, map_location=device)

    # 加载生成器和判别器参数
    model.G_A2B.load_state_dict(
        {k: v for k, v in checkpoint['G_A2B_state'].items()
         if not k.startswith('weather_layer.night_adjuster.sobel')},
        strict=False
    )

    # 同样处理其他生成器
    model.G_B2A.load_state_dict(
        {k: v for k, v in checkpoint['G_B2A_state'].items()
         if not k.startswith('weather_layer.night_adjuster.sobel')},
        strict=False
    )
    model.D_A.load_state_dict(checkpoint['D_A_state'])
    model.D_B.load_state_dict(checkpoint['D_B_state'])

    # 加载语义编码器和解码器参数
    model.semantic_encoder.load_state_dict(checkpoint['semantic_encoder_state'])
    model.semantic_decoder.load_state_dict(checkpoint['semantic_decoder_state'])

    # 设置为评估模式
    model.eval()

    return model


# 图像生成函数（仅处理测试集）
def generate_test_images(model, dataloader, output_dir, device):
    """
    生成并保存测试集图像

    参数:
        model (nn.Module): 训练好的模型
        dataloader (DataLoader): 测试数据加载器
        output_dir (str): 输出目录
        device (str): 使用的设备
    """
    # 创建输出目录
    generated_dir = os.path.join(output_dir, "test", "generated")
    os.makedirs(generated_dir, exist_ok=True)

    print(f"正在生成测试集图像...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # 解包批次数据（图像和文件名）
            images, filenames = batch
            real_A = images.to(device)

            # 生成恶劣天气图像
            fake_B = model.generate_A_to_B(real_A)

            # 保存生成的图像（使用原始文件名）
            for i in range(real_A.size(0)):
                # 获取原始文件名（不含路径）
                original_name = os.path.basename(filenames[i])
                # 保存生成的图像
                save_image(
                    denormalize(fake_B[i]),
                    os.path.join(generated_dir, original_name)
                )

            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                print(f"已处理 {batch_idx + 1} 批次，生成 {len(filenames)} 张图像")


# 主函数
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="生成恶劣天气图像")
    parser.add_argument("--model_path", type=str, default="./output/checkpoints/final_epoch_200.pth",
                        help="训练好的模型路径(.pth文件)")
    parser.add_argument("--data_dir", type=str, default="../eva_data/rain",
                        help="数据集根目录")
    parser.add_argument("--output_dir", type=str, default="../output_data",
                        help="输出目录")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="批处理大小")
    parser.add_argument("--image_size", type=int, default=256,
                        help="图像尺寸")
    parser.add_argument("--weather_type", type=str, default="rain",
                        choices=["rain", "snow", "fog", "night"],
                        help="要生成的天气类型")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="使用的设备")
    parser.add_argument("--crop_size", type=int, default=1080,
                        help="中心裁剪尺寸")
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 定义图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print(f"使用设备: {args.device}")
    print(f"加载模型: {args.model_path}")
    print(f"天气类型: {args.weather_type}")
    print(f"预处理设置: 中心裁剪 {args.crop_size}x{args.crop_size} -> 调整尺寸 {args.image_size}x{args.image_size}")

    # 加载模型
    model = load_model(args.model_path, args.device, args.weather_type)

    # 创建测试集数据加载器
    print("创建测试集数据加载器...")
    test_A_dir = args.data_dir
    test_dataset = ImageDataset(
        root_dir=test_A_dir,
        crop_size=args.crop_size,
        target_size=args.image_size,
        transform=transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: (
            torch.stack([item[0] for item in batch]),
            [item[1] for item in batch]  # 保留文件名列表
        )
    )

    # 生成测试集图像
    generate_test_images(model, test_loader, args.output_dir, args.device)

    print(f"所有测试图像已生成并保存到: {os.path.join(args.output_dir, 'test', 'generated')}")


if __name__ == "__main__":
    main()