import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm
from models_gpt import DecoderOnlyViT 
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Used Device:", device)

# Create decoder-only model with grayscale input (1 channel)
def create_model():
    return DecoderOnlyViT(
        img_size=224, patch_size=16, in_chans=1,  # Set in_chans to 1 for grayscale
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8,
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
        return image

# Define transformations for training data with single-channel grayscale conversion
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert images to grayscale with 1 channel
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load custom dataset
image_dir = '../unlabeled2017/unlabeled2017'
train_dataset = CustomImageDataset(image_dir=image_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model, loss, optimizer setup
model = create_model()
criterion = nn.MSELoss()  # Using MSE Loss as it's a reconstruction task
optimizer = optim.Adam(model.parameters(), lr=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.6, verbose=True)

# Directory to save model checkpoints
checkpoint_dir = 'checkpoints_decoder'
os.makedirs(checkpoint_dir, exist_ok=True)

# Training Loop
num_epochs = 100
masking_rates = [0.10, 0.10, 0.10]  # Different masking rates to be used
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    mask_rate = masking_rates[epoch % len(masking_rates)]  # Vary masking rates
    
    with tqdm(total=len(train_loader), desc='Epoch {}/{}'.format(epoch + 1, num_epochs), unit='batch') as pbar:
        for batch_idx, images in enumerate(train_loader):
            
            images = images.to(device)
            optimizer.zero_grad()

            # Forward pass through the decoder-only model with masking
            outputs = model(images, mask_ratio=mask_rate)
            loss = criterion(outputs, images)
            
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                pbar.set_postfix({'loss': running_loss / 100})
                running_loss = 0.0
            
            pbar.update(1)

    # Save the model checkpoint
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'model_epoch_{}_mask_{}.pth'.format(epoch + 1, int(mask_rate * 100))))
    
    # Calculate average loss for the epoch (use the validation loss if available)
    avg_loss = running_loss / len(train_loader)
    
    # Update the learning rate based on the average loss
    scheduler.step(avg_loss)

print('Finished Training')
