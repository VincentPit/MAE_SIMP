import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from models import GrayscaleToColorTransformer
from functools import partial
from tqdm import tqdm

def create_model():
    return GrayscaleToColorTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )

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
        return image

# Define transformations for training data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load custom dataset
image_dir = '../unlabeled2017/unlabeled2017'
train_dataset = CustomImageDataset(image_dir=image_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define grayscale transformation for visualization purposes
to_grayscale = transforms.Grayscale(num_output_channels=1)

# Define model, loss function, and optimizer
model = create_model()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Function to visualize reconstructed images
def visualize_reconstruction(model, images, epoch):
    model.eval()
    with torch.no_grad():
        grayscale_images = to_grayscale(images)
        outputs = model(grayscale_images)
    
    fig, axes = plt.subplots(3, 6, figsize=(12, 6))
    for i in range(6):
        # Original Image
        ax = axes[0, i]
        ax.imshow(images[i].permute(1, 2, 0))
        ax.axis('off')
        ax.set_title("Original")

        # Grayscale Image
        ax = axes[1, i]
        ax.imshow(grayscale_images[i].permute(1, 2, 0).squeeze(), cmap='gray')
        ax.axis('off')
        ax.set_title("Grayscale")

        # Reconstructed Image
        ax = axes[2, i]
        ax.imshow(outputs[i].permute(1, 2, 0).clamp(0, 1))
        ax.axis('off')
        ax.set_title("Reconstructed")
    
    plt.suptitle(f'Epoch {epoch}')
    plt.show()

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
        for batch_idx, images in enumerate(train_loader):
            # Convert images to grayscale
            grayscale_images = to_grayscale(images)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(grayscale_images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()

            # Update statistics
            running_loss += loss.item()
            if (batch_idx + 1) % 100 == 0:
                pbar.set_postfix({'loss': running_loss / 100})
                running_loss = 0.0
            
            # Update progress bar
            pbar.update(1)

    # Visualize reconstruction after each epoch
    visualize_reconstruction(model, images, epoch + 1)

print('Finished Training')
