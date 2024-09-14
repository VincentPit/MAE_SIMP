import torch
import torch.nn as nn
from models_gpt import GPTLikeDecoder

# Define a sample Block class for demonstration purposes
class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )
        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)
    
    def forward(self, x):
        x = x + self.attn(x, x, x)[0]
        x = x + self.mlp(self.norm1(x))
        return self.norm2(x)

# Define a function to test the GPT-like Decoder
def test_gpt_like_decoder():
    img_size = 224
    patch_size = 16
    in_chans = 3
    decoder_embed_dim = 512
    decoder_depth = 8
    decoder_num_heads = 16
    mlp_ratio = 4.0
    norm_layer = nn.LayerNorm
    
    # Create an instance of the GPT-like Decoder
    model = GPTLikeDecoder(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        mlp_ratio=mlp_ratio,
        norm_layer=norm_layer
    )
    
    patch_size_squared = patch_size ** 2
    
    # Generate synthetic data
    batch_size = 1
    num_patches = (img_size // patch_size) ** 2
    patch_dim = patch_size ** 2 * in_chans
    
    # Create a synthetic image tensor with shape [batch_size, in_chans, img_size, img_size]
    x_image = torch.randn(batch_size, in_chans, img_size, img_size)  # [batch_size, in_chans, img_size, img_size]

    print("input size:", x_image.shape)
    
    # Test different mask ratios
    mask_ratios = [0.0, 0.25, 0.5, 0.75]
    for mask_ratio in mask_ratios:
        print(f"Testing with mask ratio: {mask_ratio}")
        
        # Forward pass
        x_pred, mask = model(x_image, mask_ratio=mask_ratio)
        
        # Check output shapes
        assert x_pred.shape == (batch_size, int(num_patches*(1-mask_ratio)), patch_size_squared * in_chans), f"Unexpected shape for x_pred: {x_pred.shape}"
        assert mask.shape == (batch_size, num_patches), f"Unexpected shape for mask: {mask.shape}"
        
        # Check the number of masked patches
        num_masked_patches = mask.sum().item()
        expected_num_masked = int(num_patches * mask_ratio) * batch_size
        
        print("num_masked_patches:", num_masked_patches)
        print("expected_num_masked:", expected_num_masked)
        
        # If mask ratio is 0.0, expect no masked patches
        if mask_ratio == 0.0:
            assert num_masked_patches == 0, "Mask ratio 0.0 should have 0 masked patches"
        
        assert abs(num_masked_patches - expected_num_masked) < 1e-5, f"Unexpected number of masked patches: {num_masked_patches}"

        print(f"Mask ratio {mask_ratio} test passed.")

if __name__ == "__main__":
    test_gpt_like_decoder()
