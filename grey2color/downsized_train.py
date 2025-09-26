import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from models import Downsized
from functools import partial
from tqdm import tqdm

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Used Device:", device)


def create_downsized_model():
    return Downsized(
        patch_size=16,
        embed_dim=384,
        depth=6,
        num_heads=6,
        decoder_embed_dim=256,
        decoder_depth=4,
        decoder_num_heads=4,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ).to(device)


# Define custom dataset
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


# Define transformations for training data
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# Load custom dataset
image_dir = "../unlabeled2017/unlabeled2017"
train_dataset = CustomImageDataset(image_dir=image_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

to_grayscale = transforms.Grayscale(num_output_channels=1)

model = create_downsized_model()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Directory to save model checkpoints
checkpoint_dir = "checkpoints_downsized"
os.makedirs(checkpoint_dir, exist_ok=True)


def visualize_reconstruction(model, images, epoch):
    model.eval()
    with torch.no_grad():
        grayscale_images = to_grayscale(images).to(device)
        outputs = model(grayscale_images).cpu()

    fig, axes = plt.subplots(3, 6, figsize=(12, 6))
    for i in range(6):
        # Original Image
        ax = axes[0, i]
        ax.imshow(images[i].permute(1, 2, 0))
        ax.axis("off")
        ax.set_title("Original")

        # Grayscale Image
        ax = axes[1, i]
        ax.imshow(grayscale_images[i].permute(1, 2, 0).squeeze().cpu(), cmap="gray")
        ax.axis("off")
        ax.set_title("Grayscale")

        # Reconstructed Image
        ax = axes[2, i]
        ax.imshow(outputs[i].permute(1, 2, 0).clamp(0, 1))
        ax.axis("off")
        ax.set_title("Reconstructed")

    plt.suptitle(f"Epoch {epoch}")
    plt.show()


# Training Loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    with tqdm(
        total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"
    ) as pbar:
        for batch_idx, images in enumerate(train_loader):
            try:
                images = images.to(device)
                grayscale_images = to_grayscale(images)
                optimizer.zero_grad()
                outputs = model(grayscale_images)
                loss = criterion(outputs, images)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if (batch_idx + 1) % 100 == 0:
                    pbar.set_postfix({"loss": running_loss / 100})
                    running_loss = 0.0

                pbar.update(1)
            except Exception as e:
                print(f"Error during training at batch {batch_idx}: {e}")
                break

    # Save the model checkpoint
    torch.save(
        model.state_dict(), os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
    )

    # Visualize the reconstruction
    visualize_reconstruction(model, images.cpu(), epoch + 1)

print("Finished Training")
