"""
Training Script for Decoder-Only Vision Transformer

This script trains a GPT-style decoder-only Vision Transformer for image reconstruction
from grayscale inputs. The model learns to reconstruct images through self-supervised
learning with random masking.

Usage:
    python train_gpt.py [--batch_size 32] [--epochs 100] [--lr 1e-5]
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torchvision import transforms
    import matplotlib.pyplot as plt
    from PIL import Image
    from functools import partial
    from tqdm import tqdm
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install required packages: torch torchvision timm pillow matplotlib tqdm")
    sys.exit(1)

try:
    from models_gpt import DecoderOnlyViT
except ImportError:
    print("Error: Cannot import DecoderOnlyViT model. Make sure models_gpt.py is in the same directory.")
    sys.exit(1)

def setup_device() -> torch.device:
    """
    Setup and validate compute device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == 'cuda':
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.info("Using CPU - training will be slow")
    
    return device


def create_model(args, device: torch.device) -> nn.Module:
    """
    Create and initialize the Decoder-Only Vision Transformer model.
    
    Args:
        args: Command line arguments
        device: Compute device
        
    Returns:
        Initialized model
    """
    model = DecoderOnlyViT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=1,  # Grayscale input
        decoder_embed_dim=args.embed_dim,
        decoder_depth=args.depth,
        decoder_num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_rate=args.dropout
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model created with {total_params:,} total parameters")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model


def create_data_transforms(args) -> transforms.Compose:
    """
    Create data transformation pipeline.
    
    Args:
        args: Command line arguments
        
    Returns:
        Composed transforms
    """
    transform_list = [
        transforms.Resize((args.img_size, args.img_size)),
    ]
    
    # Add data augmentation if specified
    if args.augment:
        transform_list.extend([
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomRotation(5)
        ])
    
    transform_list.extend([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])
    
    return transforms.Compose(transform_list)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CustomImageDataset(Dataset):
    """
    Custom dataset for loading images from a directory.
    
    Supports various image formats and includes error handling for corrupted files.
    """
    
    def __init__(self, image_dir: str, transform=None, max_samples: Optional[int] = None):
        """
        Initialize dataset.
        
        Args:
            image_dir: Directory containing images
            transform: Optional torchvision transforms
            max_samples: Optional limit on number of samples (for debugging)
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        
        # Support multiple image formats
        supported_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        self.image_paths = []
        
        for ext in supported_extensions:
            self.image_paths.extend(list(self.image_dir.glob(ext)))
            self.image_paths.extend(list(self.image_dir.glob(ext.upper())))
        
        if not self.image_paths:
            raise ValueError(f"No supported images found in {image_dir}")
        
        # Limit samples for debugging
        if max_samples:
            self.image_paths = self.image_paths[:max_samples]
        
        logger.info(f"Found {len(self.image_paths)} images in {image_dir}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_paths[idx]
        
        try:
            # Load and convert image to RGB
            image = Image.open(img_path).convert('RGB')
            
            # Verify image is not corrupted
            image.verify()
            
            # Reload image after verify (verify closes the file)
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image
            
        except (IOError, OSError) as e:
            logger.warning(f"Corrupted image {img_path}: {e}. Using random tensor.")
            # Return random tensor with same dimensions as expected output
            if self.transform:
                # Create a dummy image and apply transforms
                dummy_img = Image.new('RGB', (224, 224), color=(128, 128, 128))
                return self.transform(dummy_img)
            else:
                return torch.randn(3, 224, 224)
        except Exception as e:
            logger.error(f"Unexpected error loading {img_path}: {e}")
            raise


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, scheduler, 
                   epoch: int, loss: float, checkpoint_dir: Path, args) -> None:
    """
    Save model checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'args': vars(args)
    }
    
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, checkpoint_path)
    
    # Keep only last 3 checkpoints to save space
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    if len(checkpoints) > 3:
        for old_checkpoint in checkpoints[:-3]:
            old_checkpoint.unlink()


def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, 
                   scheduler, checkpoint_path: str) -> int:
    """
    Load model checkpoint and return starting epoch.
    """
    if not os.path.exists(checkpoint_path):
        return 0
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logger.info(f"Resumed from checkpoint at epoch {checkpoint['epoch']}")
    return checkpoint['epoch']


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer,
               criterion: nn.Module, device: torch.device, epoch: int, args) -> float:
    """
    Train for one epoch.
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    # Vary masking ratio during training for robustness
    base_mask_ratio = args.mask_ratio
    mask_ratio = base_mask_ratio + 0.1 * torch.sin(torch.tensor(epoch * 0.1))
    mask_ratio = torch.clamp(mask_ratio, 0.1, 0.8).item()
    
    with tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch") as pbar:
        for batch_idx, images in enumerate(pbar):
            try:
                images = images.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(images, mask_ratio=mask_ratio)
                loss = criterion(outputs, images)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                running_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                if batch_idx % 10 == 0:
                    pbar.set_postfix({
                        'loss': f"{loss.item():.6f}",
                        'mask_ratio': f"{mask_ratio:.2f}",
                        'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                    })
                    
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
    
    return running_loss / num_batches if num_batches > 0 else float('inf')


def main():
    """
    Main training function.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Decoder-Only Vision Transformer')
    parser.add_argument('--data_dir', type=str, default='../unlabeled2017/unlabeled2017',
                       help='Directory containing training images')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_decoder',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    
    # Model parameters
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size')
    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--depth', type=int, default=8, help='Number of decoder blocks')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help='MLP ratio')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='Masking ratio')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples for debugging')
    
    args = parser.parse_args()
    
    # Setup
    device = setup_device()
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Create data transforms and dataset
    transform = create_data_transforms(args)
    
    try:
        train_dataset = CustomImageDataset(
            image_dir=args.data_dir,
            transform=transform,
            max_samples=args.max_samples
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Dataset error: {e}")
        return
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True
    )
    
    # Create model, loss, and optimizer
    model = create_model(args, device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.6, verbose=True)
    
    # Resume from checkpoint if specified
    start_epoch = load_checkpoint(model, optimizer, scheduler, args.resume) if args.resume else 0
    
    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs")
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Train one epoch
        avg_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch + 1, args)
        
        # Update scheduler
        scheduler.step(avg_loss)
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, scheduler, epoch + 1, avg_loss, checkpoint_dir, args)
        
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch + 1}/{args.epochs} completed in {epoch_time:.1f}s - Loss: {avg_loss:.6f}")
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
