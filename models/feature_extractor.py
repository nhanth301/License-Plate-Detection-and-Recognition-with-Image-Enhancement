import torch.nn as nn
import torch.nn.functional as F
from models.cross_attn import CrossAttention 

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, feature_dim=64):
        super(FeatureExtractor, self).__init__()
        
        # Encoder for clear image
        self.enc1_clear = self.conv_block(in_channels, feature_dim)
        self.enc2_clear = self.conv_block(feature_dim, feature_dim * 2)
        self.enc3_clear = self.conv_block(feature_dim * 2, feature_dim * 4)
        
        # Encoder for blurry image
        self.enc1_blur = self.conv_block(in_channels, feature_dim)
        self.enc2_blur = self.conv_block(feature_dim, feature_dim * 2)
        self.enc3_blur = self.conv_block(feature_dim * 2, feature_dim * 4)
        
        # Cross-Attention
        self.cross_attn = CrossAttention(feature_dim * 4)
        
        # Final projection to output blur features
        self.feature_projection = nn.Linear(feature_dim * 4, feature_dim * 4)
        
        self.pool = nn.MaxPool2d(2, 2)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, clear_img, blur_img):
        # Encode clear image
        enc1_clear = self.enc1_clear(clear_img)
        enc2_clear = self.enc2_clear(self.pool(enc1_clear))
        enc3_clear = self.enc3_clear(self.pool(enc2_clear))
        
        # Encode blurry image
        enc1_blur = self.enc1_blur(blur_img)
        enc2_blur = self.enc2_blur(self.pool(enc1_blur))
        enc3_blur = self.enc3_blur(self.pool(enc2_blur))
        
        # Reshape for cross-attention
        B, C, H, W = enc3_clear.shape
        clear_flat = enc3_clear.view(B, C, H * W).transpose(1, 2)  # [B, H*W, feature_dim*4]
        blur_flat = enc3_blur.view(B, C, H * W).transpose(1, 2)    # [B, H*W, feature_dim*4]
        
        # Apply cross-attention to extract blur features
        blur_features = self.cross_attn(clear_flat, blur_flat, blur_flat)
        blur_features = self.feature_projection(blur_features)  # [B, H*W, feature_dim*4]
        
        return blur_features