import os
import glob
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from functools import partial
from models_gpt import DecoderOnlyViT 

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Used Device:", device)

# Create decoder-only model
def create_model():
    return DecoderOnlyViT(
        img_size=224, patch_size=16, in_chans=3,
        decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6)
    ).to(device)

# Define custom dataset
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path  # Return image and path for saving

# Define transformations for testing data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load custom dataset
image_dir = '../unlabeled2017/unlabeled2017'
test_dataset = CustomImageDataset(image_dir=image_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the latest model checkpoint
def load_latest_checkpoint(model, checkpoint_dir='checkpoints_decoder'):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found in the directory.")
    latest_checkpoint = max(checkpoints, key=os.path.getctime)  # Get the latest checkpoint
    print(f"Loading checkpoint: {latest_checkpoint}")
    model.load_state_dict(torch.load(latest_checkpoint))
    model.eval()

# Function to save ground truth and prediction pairs
def save_comparison(images, predictions, image_paths, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    for img, pred, img_path in zip(images, predictions, image_paths):
        img = img.permute(1, 2, 0).cpu().numpy()  # Convert tensor to numpy array (H, W, C)
        pred = pred.permute(1, 2, 0).cpu().numpy()  # Convert prediction to numpy array (H, W, C)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(img)
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')

        axes[1].imshow(pred)
        axes[1].set_title('Prediction')
        axes[1].axis('off')

        filename = os.path.basename(img_path).replace('.jpg', '_comparison.png')
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()

# Model setup
model = create_model()
load_latest_checkpoint(model)

# Criterion for loss (if you plan to evaluate loss too)
criterion = nn.MSELoss()

# Evaluate and save the predictions
model.eval()
with torch.no_grad():
    for batch_idx, (images, image_paths) in enumerate(test_loader):
        images = images.to(device)

        # Forward pass through the model to get the predictions
        predictions, mask = model(images, mask_ratio=0.5)  # You can adjust the masking ratio

        # Convert predicted masked patches into the reconstructed image format
        reconstructed_images = model.patch_embed(images)
        
        # Save comparisons of ground truth and prediction pairs
        save_comparison(images, reconstructed_images, image_paths)

print("Results saved in the 'results' directory.")
