import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Import the MAE ViT model (assuming the module is saved as 'mae_vit.py')
from models_mae import mae_vit_base_patch16, mae_vit_large_patch16

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the MAE model (example: base version)
model = mae_vit_base_patch16()
model.to(device)
model.eval()

# Create a random sample image tensor (batch size: 1, channels: 3, height: 224, width: 224)
img_size = 224
sample_img = torch.randn(1, 3, img_size, img_size).to(device)

# Forward pass through the encoder only (no masking)
with torch.no_grad():
    latent, mask, ids_restore = model.forward_encoder(sample_img, mask_ratio=0.0)  # No masking

# Display the shape of the encoder output
print(f"Latent shape: {latent.shape}")  # Encoder output shape

# Visualize the random input image
def visualize_image(img):
    img = img.permute(1, 2, 0).cpu().numpy()
    plt.imshow(img)
    plt.show()

# Visualizing the random sample image
visualize_image(sample_img[0])