import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
import argparse

# 导入新模型
from model import SPCModel


class ImageDataset(torch.utils.data.Dataset):
    """与原始 eva.py 相同，保留文件名"""
    def __init__(self, root_dir, crop_size=1080, target_size=256, transform=None):
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.target_size = target_size
        self.transform = transform
        self.image_files = sorted([
            f for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # 中心裁剪
        width, height = image.size
        left = (width - self.crop_size) // 2
        top = (height - self.crop_size) // 2
        right = left + self.crop_size
        bottom = top + self.crop_size
        image = image.crop((left, top, right, bottom))

        # 调整尺寸
        image = image.resize((self.target_size, self.target_size), Image.BILINEAR)

        if self.transform:
            image = self.transform(image)

        return image, img_name


def denormalize(tensor):
    """[-1,1] -> [0,1]"""
    return (tensor.clamp(-1, 1) + 1) / 2


def load_model(model_path, device, weather_type="rain"):
    """
    加载训练好的 SPCModel 生成器
    """
    # 创建模型实例（仅生成器部分会被加载）
    model = SPCModel(weather_type=weather_type)
    model.to(device)

    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)

    # 只加载生成器权重（推理不需要判别器）
    model.generator.load_state_dict(checkpoint['generator_state'])

    # 设置为评估模式
    model.eval()

    return model


def generate_test_images(model, dataloader, output_dir, device, weather_type):
    """
    生成并保存测试集图像（适配新模型）
    """
    generated_dir = os.path.join(output_dir, "test", "generated")
    os.makedirs(generated_dir, exist_ok=True)

    print(f"正在生成 {weather_type} 天气图像...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images, filenames = batch
            real_A = images.to(device)

            # 新模型前向：返回 (I_out, info_dict)
            fake_B, _ = model.generator(
                real_A,
                weather_type=weather_type,
                weather_intensity=1.0,
                enable_aux=False
            )

            # 保存每张图像
            for i in range(real_A.size(0)):
                original_name = os.path.basename(filenames[i])
                save_image(
                    denormalize(fake_B[i]),
                    os.path.join(generated_dir, original_name)
                )

            if (batch_idx + 1) % 10 == 0:
                print(f"已处理 {batch_idx + 1} 批次，生成 {len(filenames)} 张图像")


def main():
    parser = argparse.ArgumentParser(description="生成恶劣天气图像（新模型适配版）")
    parser.add_argument("--model_path", type=str,
                        default="./output/checkpoints/best_epoch_10.pth",
                        help="训练好的模型路径(.pth文件)")
    parser.add_argument("--data_dir", type=str, default="../eva_data/ALL",
                        help="输入图像目录")
    parser.add_argument("--output_dir", type=str, default="../output_data/ALL/rain",
                        help="输出目录")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--weather_type", type=str, default="rain",
                        choices=["rain", "snow", "fog", "night"],
                        help="要生成的天气类型")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--crop_size", type=int, default=1080)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 预处理（与训练一致）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print(f"使用设备: {args.device}")
    print(f"加载模型: {args.model_path}")
    print(f"天气类型: {args.weather_type}")

    # 加载模型
    model = load_model(args.model_path, args.device, args.weather_type)

    # 创建数据集和数据加载器
    test_dataset = ImageDataset(
        root_dir=args.data_dir,
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
            [item[1] for item in batch]
        )
    )

    # 生成图像
    generate_test_images(model, test_loader, args.output_dir, args.device, args.weather_type)

    print(f"所有测试图像已保存至: {os.path.join(args.output_dir, 'test', 'generated')}")


if __name__ == "__main__":
    main()