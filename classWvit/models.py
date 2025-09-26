import torch
import torch.nn as nn
from visual_embed.models import MAEEncoder, prepare_model


class ViTClassificationModel(nn.Module):
    def __init__(self, vit_model, num_classes):
        super(ViTClassificationModel, self).__init__()
        self.vit_model = vit_model
        self.classifier = nn.Linear(
            1024, num_classes
        )  # Assuming 1024 is the hidden size of the ViT

    def forward(self, image_tensor):
        # Get ViT embeddings
        image_embeddings = self.vit_model.forward(
            image_tensor
        )  # Shape: [batch_size, num_patches, 1024]

        # Use the [CLS] token or perform pooling
        cls_token_embedding = image_embeddings[
            :, 0, :
        ]  # Assuming the first token is the [CLS] token

        # Pass through the classification head
        logits = self.classifier(
            cls_token_embedding
        )  # Shape: [batch_size, num_classes]

        return logits


if __name__ == "__main__":
    # Load the pretrained ViT model
    vit_model = prepare_model(
        chkpt_dir="visual_embed/mae_visualize_vit_large.pth",
        arch="mae_vit_large_patch16",
        only_encoder=True,
    )

    # Define the number of classes (e.g., 10 for CIFAR-10)
    num_classes = 10

    # Initialize the model
    model = ViTClassificationModel(vit_model, num_classes)

    # Example image tensor
    image_tensor = torch.randn(
        1, 3, 224, 224
    )  # Batch size of 1, 3 color channels, 224x224 image

    model.eval()

    with torch.no_grad():
        output = model.forward(image_tensor)
        print("Model output shape:", output.shape)  # Should be [1, num_classes]
        predicted_class = torch.argmax(output, dim=1)
        print("Predicted class:", predicted_class.item())
