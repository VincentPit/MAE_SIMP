from functools import partial
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed


class DecoderOnlyViT(nn.Module):
    """ Decoder-only architecture with Vision Transformer backbone """
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 decoder_embed_dim=1024, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, decoder_embed_dim)
        num_patches = self.patch_embed.num_patches

        # Positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)

        # Mask token (used for masked patches)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Decoder transformer blocks
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)

        # Output projection to predict patch pixels
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

        # Loss function configuration
        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize positional embeddings using sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** 0.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Initialize patch embedding weights (like nn.Linear)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize tokens
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # Initialize linear layers and layer norm
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
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (mask_ratio))
        
        # Create a mask with the same shape as x
        mask = torch.ones([N, L], device=x.device)
        
        # Generate random noise and sort to determine which patches to keep
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        
        # Compute indices to keep
        ids_keep = ids_shuffle[:, :len_keep]
        
        # Compute indices for restoration
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Create the masked version of x
        x_masked = x.clone()
        mask = mask.scatter(1, ids_keep, 0)  # Set mask values to 0 for kept patches
        
        # Apply the mask to x
        x_masked = x_masked * mask.unsqueeze(-1)  # Element-wise multiplication with the mask

        # Ensure the mask is in the same order as the input tensor
        mask = mask.gather(1, ids_restore)
        
        return x_masked, mask

    def forward(self, imgs, mask_ratio=0.50):
        # Step 1: Patchify the input image and embed patches
        x = self.patch_embed(imgs)

        # Step 2: Apply positional embedding
        x = x + self.pos_embed[:, 1:, :]

        # Step 3: Mask a portion of patches
        x, mask = self.random_masking(x, mask_ratio)

        # Step 4: Append the class token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Step 5: Pass through decoder transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        # Step 6: Predict the pixels for the masked patches
        x = self.decoder_pred(x)
        
        x = x[:, 1:, :]  # Remove the class token for the output
        x = self.unpatchify(x)
        return x
