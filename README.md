# SPC: Semantic-Physical Collaborative Framework

A semantic-physical collaborative driven framework for adverse weather image generation in autonomous driving scenarios.

## Paper Information

**Title**: SPC: A Semantic-Physical Collaborative Framework for Adverse Weather Image Generation in Autonomous Driving

**Authors**: Wenjun Zhang, Chuanqi Tao

**Affiliation**: College of Computer Science and Technology, Nanjing University of Aeronautics and Astronautics

## Environment Configuration

```
# Install dependencies
pip install -r requirements.txt
```

Dependency version requirements:

- torch==2.4.1+cu124

- torchvision==0.19.1

- opencv_python==4.11.0.86

- Other dependencies as specified in requirements.txt

## Project Structure

```
SPC/                           # Main project directory
├── fog/                       # Fog weather model
│   ├── data/
│   │   ├── train/
│   │   │   ├── A/images/     # Clear weather images
│   │   │   ├── A/semantics/ # Semantic segmentation maps
│   │   │   ├── B/images/     # Adverse weather images
│   │   │   └── B/semantics/  # Semantic segmentation maps
│   │   └── val/
│   │       ├── A/
│   │       └── B/
│   ├── output/
│   │   ├── checkpoints/      # Model checkpoints
│   │   ├── samples/          # Training sample visualizations
│   │   └── logs/             # TensorBoard logs
│   ├── model.py              # Model definition
│   ├── train.py              # Training script
│   ├── eval.py               # Evaluation and image translation
│   └── utils.py              # Utility functions
├── rain/                      # Rain weather model (same structure as fog)
├── snow/                      # Snow weather model (same structure as fog)
├── night/                     # Night model (same structure as fog)
└── requirements.txt           # Dependency file
```

## Dataset

This project is built upon the following public datasets:

- **ACDC**​ (Adverse Conditions Dataset with Correspondences)

- **BDD100K**​ (Berkeley Deep Drive)

### Data Preparation Requirements

1. Images and semantic segmentation maps must have **identical filenames**

2. Semantic maps should use single-channel label format (not color images)

3. Recommended to use Cityscapes 19-class semantic labeling system

## Quick Start

### 1. Data Preparation

```
# Navigate to specific weather directory
cd SPC/fog

# Ensure correct directory structure
# data/train/A/images/ contains clear weather images
# data/train/A/semantics/ contains corresponding semantic segmentation maps
# data/train/B/images/ contains adverse weather images
# data/train/B/semantics/ contains corresponding semantic segmentation maps
```

### 2. Model Training

```
# Train fog weather model in fog directory
python train.py --data_dir ./data --weather_type fog --epochs 150

# Training parameters
# --data_dir: data directory path
# --weather_type: weather type (fog/rain/snow/night)
# --epochs: number of training epochs
# --batch_size: batch size (default: 4)
# --image_size: image size (default: 256)
```

### 3. Generate Adverse Weather Images

```
# Generate images using trained model
python eval.py \
    --model_path ./output/checkpoints/final_epoch_150.pth \
    --data_dir ./test_images \
    --output_dir ./generated_results \
    --weather_type fog
```

## Supported Weather Types

This project supports four types of adverse weather generation:

- **Fog**​ - located in `SPC/fog/`directory

- **Rain**​ - located in `SPC/rain/`directory

- **Snow**​ - located in `SPC/snow/`directory

- **Night**​ - located in `SPC/night/`directory

## Training Monitoring

```
# Start TensorBoard to monitor training process
tensorboard --logdir ./output/logs --port 6006
```

Visit `http://localhost:6006`to view training loss curves and generated samples.

## Important Notes

1. Each weather type requires separate training and evaluation

2. Ensure correct directory structure for data

3. Check GPU memory availability before training

4. Select the corresponding weather type directory when generating images

## License

MIT License
