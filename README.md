# MAE_SIMP

**Masked Autoencoder Simplified** - A collection of Vision Transformer models for image reconstruction and colorization tasks using Masked Autoencoder (MAE) architecture.

## 📋 Overview

This project implements various Vision Transformer (ViT) based models for:
- **Grayscale to Color Image Reconstruction**: Converting grayscale images to colored images using an encoder-decoder transformer architecture
- **Masked Autoencoder (MAE)**: Self-supervised learning with image reconstruction from masked patches
- **Decoder-Only Vision Transformer**: GPT-like architecture for image generation and reconstruction
- **ImGPT**: GPT-style model adapted for image processing

## 🏗️ Project Structure

```
MAE_SIMP/
├── classWvit/              # Decoder-only ViT and visual embeddings
│   ├── models_gpt.py       # Decoder-only ViT implementation
│   ├── train_gpt.py        # Training script for decoder-only model
│   ├── test_gpt.py         # Testing/inference script
│   ├── util/               # Utility functions (positional embeddings, etc.)
│   └── visual_embed/       # MAE and ViT models for visual embeddings
├── grey2color/             # Grayscale to color conversion
│   ├── models.py           # Transformer models for colorization
│   ├── train.py            # Full training script
│   ├── downsized_train.py  # Training with smaller model
│   └── util/               # Shared utilities
├── ImGPT/                  # GPT-style model for images
│   └── models/
│       └── model.py        # GPT implementation for images
├── download.py             # COCO dataset downloader
├── MAE_env.yml            # Conda environment configuration
└── submit_job*.sh         # SLURM job submission scripts
```

## 🚀 Getting Started

### Prerequisites

1. **Create Conda Environment**:
```bash
conda env create -f MAE_env.yml
conda activate MAE_env
```

2. **Download Dataset**:
```bash
python download.py
```
This downloads the COCO 2017 unlabeled dataset (~20GB) which is used for training.

### Environment Setup

The project requires:
- Python 3.8+
- PyTorch 2.3+ with CUDA support
- torchvision
- timm (PyTorch Image Models)
- PIL/Pillow for image processing
- matplotlib for visualization
- tqdm for progress bars

## 🎯 Models and Training

### 1. Grayscale to Color Conversion

**Location**: `grey2color/`

- **Full Model**: Run `python train.py` for complete grayscale-to-color conversion
- **Downsized Model**: Run `python downsized_train.py` for a lighter version with fewer parameters

**Architecture**: 
- Encoder-decoder transformer with patch-based processing
- Input: 1-channel grayscale images (224×224)
- Output: 3-channel RGB images
- Uses MAE-style masking during training

### 2. Decoder-Only Vision Transformer

**Location**: `classWvit/`

- **Training**: `python train_gpt.py`
- **Testing**: `python test_gpt.py`

**Features**:
- GPT-like decoder-only architecture
- Supports variable masking ratios (0.10 by default)
- Grayscale input processing (1-channel)
- Self-supervised reconstruction learning

### 3. Masked Autoencoder (MAE)

**Location**: `classWvit/visual_embed/`

Classic MAE implementation with:
- Vision Transformer encoder
- Lightweight decoder for reconstruction
- Random patch masking strategy
- Self-supervised pre-training capability

### 4. ImGPT

**Location**: `ImGPT/`

GPT-style architecture adapted for images:
- Patch-based image tokenization
- Autoregressive generation capabilities
- Large-scale transformer architecture (1024 dim, 24 layers)

## 💻 High-Performance Computing

### SLURM Job Submission

For training on HPC clusters:

```bash
# Full decoder training
sbatch submit_job.sh

# Downsized model training  
sbatch submit_job_downsized.sh
```

**Resource Requirements**:
- GPU: CUDA-capable (tested with V100/A100)
- Memory: 64-100GB RAM depending on model
- Storage: ~25GB for dataset + model checkpoints
- Time: 4-5 days for full training

## 🔧 Key Features

### Model Architectures
- **Patch Embedding**: 16×16 patches with configurable embedding dimensions
- **Positional Encoding**: Sinusoidal position embeddings
- **Attention Mechanisms**: Multi-head self-attention with configurable heads
- **Flexible Depths**: Adjustable encoder/decoder depths for different computational budgets

### Training Features
- **Mixed Precision Training**: Automatic mixed precision for faster training
- **Learning Rate Scheduling**: ReduceLROnPlateau scheduler
- **Model Checkpointing**: Automatic saving of training checkpoints
- **Visualization**: Real-time reconstruction visualization during training
- **Progress Tracking**: tqdm progress bars with loss monitoring

### Data Processing
- **Automatic Resizing**: Images resized to 224×224
- **Grayscale Conversion**: Automatic RGB to grayscale conversion
- **Normalization**: Standard ImageNet normalization
- **Augmentation**: Built-in support for data augmentation

## 📊 Model Performance

The models are designed for:
- **Image Reconstruction**: High-fidelity reconstruction from masked/grayscale inputs
- **Self-Supervised Learning**: No labeled data required
- **Scalability**: Models range from lightweight (384 dim) to large-scale (1024 dim)
- **Flexibility**: Easy adaptation to different input/output channels

## 🛠️ Customization

### Hyperparameter Tuning
Key parameters to adjust:
- `embed_dim`: Model width (384, 512, 768, 1024)
- `depth`: Number of transformer layers (6, 8, 12, 24)
- `num_heads`: Multi-head attention heads (4, 6, 8, 12, 16)
- `mask_ratio`: Masking ratio for MAE training (0.1-0.75)
- `learning_rate`: Training learning rate (1e-5 to 1e-4)

### Adding New Models
1. Create model class inheriting from `nn.Module`
2. Implement `patchify`/`unpatchify` methods for patch processing
3. Add forward pass with attention mechanisms
4. Create corresponding training script

## 📝 Citation

If you use this code in your research, please cite the original MAE paper:
```bibtex
@article{he2022masked,
  title={Masked autoencoders are scalable vision learners},
  author={He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📄 License

This project builds upon Meta's MAE implementation and follows similar licensing terms. Please refer to individual file headers for specific license information.
