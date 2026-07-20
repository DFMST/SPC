import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
import argparse
import time
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


def load_model(model_path, device, weather_type="night"):
    model = SPCModel(weather_type=weather_type)
    model.to(device)
    checkpoint = torch.load(model_path, map_location=device)

    # 使用 strict=False 忽略不匹配的缓冲区（如 x_grid, y_grid）
    missing_keys, unexpected_keys = model.generator.load_state_dict(
        checkpoint['generator_state'], strict=False
    )

    if missing_keys:
        print(f"缺失的键（将被忽略）: {missing_keys}")
    if unexpected_keys:
        print(f"额外的键（将被忽略）: {unexpected_keys}")

    model.eval()
    return model


def generate_test_images(model, dataloader, output_dir, device, weather_type):
    """
    生成并保存测试集图像，同时统计平均推理时间。
    """
    generated_dir = os.path.join(output_dir, "test", "generated")
    os.makedirs(generated_dir, exist_ok=True)

    print(f"正在生成 {weather_type} 天气图像...")

    total_inference_time = 0.0   # 毫秒
    total_samples = 0

    # 判断是否使用 CUDA，选择不同的计时方法
    use_cuda = device.type == 'cuda'

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images, filenames = batch
            real_A = images.to(device)
            batch_size = real_A.size(0)

            # ---------- 开始计时 ----------
            if use_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_time = time.perf_counter()

            # 模型前向传播（唯一需要计时的部分）
            fake_B, _ = model.generator(
                real_A,
                weather_type=weather_type,
                weather_intensity=1.0,
                enable_aux=False
            )

            # ---------- 结束计时 ----------
            if use_cuda:
                end_event.record()
                torch.cuda.synchronize()
                batch_time_ms = start_event.elapsed_time(end_event)   # 毫秒
            else:
                end_time = time.perf_counter()
                batch_time_ms = (end_time - start_time) * 1000       # 转换为毫秒

            total_inference_time += batch_time_ms
            total_samples += batch_size

            # 保存每张图像（不计入推理时间）
            for i in range(batch_size):
                original_name = os.path.basename(filenames[i])
                save_image(
                    denormalize(fake_B[i]),
                    os.path.join(generated_dir, original_name)
                )

            if (batch_idx + 1) % 10 == 0:
                print(f"已处理 {batch_idx + 1} 批次，共 {total_samples} 张图像")

    # 计算平均推理时间
    avg_time_per_image = total_inference_time / total_samples if total_samples > 0 else 0
    print(f"\n===== 推理时间统计 =====")
    print(f"总推理时间: {total_inference_time:.2f} ms")
    print(f"总图像数: {total_samples}")
    print(f"平均每张图像推理时间: {avg_time_per_image:.2f} ms")
    print(f"=========================\n")
    print(f"所有测试图像已保存至: {generated_dir}")


def main():
    parser = argparse.ArgumentParser(description="生成恶劣天气图像（新模型适配版）")
    parser.add_argument("--model_path", type=str,
                        default="./output/checkpoints/final_epoch_200.pth",
                        help="训练好的模型路径(.pth文件)")
    parser.add_argument("--data_dir", type=str, default="../eva_data/ALL",
                        help="输入图像目录")
    parser.add_argument("--output_dir", type=str, default="../output_data/ALL/night",
                        help="输出目录")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--weather_type", type=str, default="night",
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