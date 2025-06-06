import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cross_attn import CrossAttention

class BlurGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, feature_dim=64):
        super(BlurGenerator, self).__init__()
        
        # Encoder for clear image
        self.enc1 = self.conv_block(in_channels, feature_dim)
        self.enc2 = self.conv_block(feature_dim, feature_dim * 2)
        self.enc3 = self.conv_block(feature_dim * 2, feature_dim * 4)
        self.enc4 = self.conv_block(feature_dim * 4, feature_dim * 8)
        
        # Cross-Attention for blur features
        self.cross_attn = CrossAttention(feature_dim * 4)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(feature_dim * 8, feature_dim * 4, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(feature_dim * 8, feature_dim * 4)
        self.upconv3 = nn.ConvTranspose2d(feature_dim * 4, feature_dim * 2, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(feature_dim * 4, feature_dim * 2)
        self.upconv2 = nn.ConvTranspose2d(feature_dim * 2, feature_dim, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(feature_dim * 2, feature_dim)
        self.final_conv = nn.Conv2d(feature_dim, out_channels, kernel_size=1)
        
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

    def forward(self, clear_img, blur_features):
        # Encode clear image
        enc1 = self.enc1(clear_img)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Reshape for cross-attention
        B, C, H, W = enc3.shape
        enc3_flat = enc3.view(B, C, H * W).transpose(1, 2)  # [B, H*W, feature_dim*4]
        
        # Apply cross-attention with blur features
        attn_output = self.cross_attn(enc3_flat, blur_features, blur_features)
        attn_output = attn_output.transpose(1, 2).view(B, C, H, W)
        
        # Decoder with skip connections
        dec4 = self.upconv4(enc4)
        dec4 = torch.cat([dec4, attn_output], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc2], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc1], dim=1)
        dec2 = self.dec2(dec2)
        
        output = self.final_conv(dec2)
        return output