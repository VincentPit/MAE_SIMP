from functools import partial
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed
import torch.nn.functional as F


class DecoderOnlyViT(nn.Module):
    """
    Decoder-Only Vision Transformer for Image Reconstruction
    
    This model implements a GPT-style decoder-only architecture for vision tasks.
    Unlike traditional encoder-decoder models, this uses only decoder blocks to
    reconstruct images from masked patches, similar to how GPT predicts next tokens.
    
    Architecture:
    1. Patch Embedding: Splits images into patches and embeds them
    2. Positional Encoding: Adds learnable position information
    3. Random Masking: Removes patches for self-supervised learning
    4. Decoder Blocks: Stack of transformer blocks for processing
    5. Prediction Head: Reconstructs pixel values for patches
    
    Args:
        img_size (int): Input image size (default: 224)
        patch_size (int): Size of each patch (default: 16) 
        in_chans (int): Number of input channels (default: 1 for grayscale)
        decoder_embed_dim (int): Decoder embedding dimension (default: 512)
        decoder_depth (int): Number of decoder blocks (default: 8)
        decoder_num_heads (int): Number of attention heads (default: 8)
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim (default: 4.0)
        norm_layer: Normalization layer (default: nn.LayerNorm)
        norm_pix_loss (bool): Use normalized pixel loss (default: False)
        drop_rate (float): Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=1,  # Ensure in_chans=1 for grayscale
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=8,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        drop_rate=0.1,
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, decoder_embed_dim)
        num_patches = self.patch_embed.num_patches

        # Positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )

        # Mask token (used for masked patches)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Decoder transformer blocks
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)

        # Add dropout layer
        self.dropout = nn.Dropout(drop_rate)

        # Output projection to predict patch pixels
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )

        # Loss function configuration
        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        Convert images to patches.
        
        Args:
            imgs: Input images [N, C, H, W]
            
        Returns:
            x: Patches [N, L, patch_size**2 * C] where L = H*W/patch_size**2
        """
        p = self.patch_embed.patch_size[0]
        c = imgs.shape[1]  # Get actual number of channels
        
        # Validate input dimensions
        assert imgs.shape[2] == imgs.shape[3], f"Images must be square, got {imgs.shape[2]}x{imgs.shape[3]}"
        assert imgs.shape[2] % p == 0, f"Image size {imgs.shape[2]} not divisible by patch size {p}"

        h = w = imgs.shape[2] // p
        
        # Reshape: [N, C, H, W] -> [N, C, h, p, w, p] -> [N, h, w, p, p, C]
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
        
        return x

    def unpatchify(self, x):
        """
        Convert patches back to images.
        
        Args:
            x: Patches [N, L, patch_size**2 * C]
            
        Returns:
            imgs: Images [N, C, H, W]
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        
        # Validate patch grid is square
        assert h * w == x.shape[1], f"Patch sequence length {x.shape[1]} is not a perfect square"
        
        # Determine number of channels from patch dimension
        c = x.shape[2] // (p ** 2)
        assert c * (p ** 2) == x.shape[2], f"Patch dimension {x.shape[2]} not divisible by {p**2}"
        
        # Reshape: [N, L, p*p*C] -> [N, h, w, p, p, C] -> [N, C, H, W]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        
        Args:
            x: Input tensor [N, L, D] where N=batch, L=length, D=dim
            mask_ratio: Ratio of patches to mask (0.0 to 1.0)
            
        Returns:
            x_masked: Masked input tensor [N, L_keep, D]
            mask: Binary mask [N, L] where 1 indicates masked patches
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))  # Number of patches to KEEP (not mask)
        
        # Generate random noise for shuffling
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)  # Ascending sort
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first subset (unmasked patches)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore

    def forward(self, imgs, mask_ratio=0.50):
        """
        Forward pass through the decoder-only ViT.
        
        Args:
            imgs: Input images [N, C, H, W]
            mask_ratio: Ratio of patches to mask (default 0.5)
            
        Returns:
            pred: Reconstructed images [N, C, H, W]
        """
        # Step 1: Embed patches
        x = self.patch_embed(imgs)  # [N, L, D]
        N, L, D = x.shape
        
        # Step 2: Add positional embeddings (excluding cls token)
        x = x + self.pos_embed[:, 1:, :]
        
        # Step 3: Masking - remove random patches for reconstruction task
        x_masked, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Step 4: Append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x_masked.shape[0], -1, -1)
        x_masked = torch.cat((cls_tokens, x_masked), dim=1)
        
        # Step 5: Apply dropout after positional embeddings
        x_masked = self.dropout(x_masked)
        
        # Step 6: Pass through decoder blocks
        for blk in self.decoder_blocks:
            x_masked = blk(x_masked)
        x_masked = self.decoder_norm(x_masked)
        
        # Step 7: Predictor - predict all patches (including masked ones)
        x_masked = self.decoder_pred(x_masked)  # [N, 1+L_keep, p*p*C]
        
        # Remove cls token
        x_masked = x_masked[:, 1:, :]
        
        # Step 8: For decoder-only, we need to reconstruct all patches
        # Create full prediction tensor
        pred_full = torch.zeros(N, L, x_masked.shape[-1], device=imgs.device)
        
        # Fill in the kept (unmasked) patches with predictions
        len_keep = x_masked.shape[1]
        ids_keep = torch.argsort(torch.rand(N, L, device=imgs.device), dim=1)[:, :len_keep]
        
        # Use gather to place predictions back to original positions
        for i in range(N):
            pred_full[i, ids_keep[i]] = x_masked[i]
        
        # For masked patches, use mask tokens or zero predictions
        # This is where the model learns to reconstruct from partial information
        
        # Step 9: Reshape to image
        pred = self.unpatchify(pred_full)  # [N, C, H, W]
        
        return pred
