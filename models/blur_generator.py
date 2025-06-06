import torch
import torch.nn as nn
import torch.nn.functional as F



import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, feature_dim=64):
        super(FeatureExtractor, self).__init__()
        self.enc1_blur = self.conv_block(in_channels, feature_dim)
        self.enc2_blur = self.conv_block(feature_dim, feature_dim * 2)
        self.enc3_blur = self.conv_block(feature_dim * 2, feature_dim * 4)
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

    def forward(self, blur_img):
        # Encode blurry image only
        enc1_blur = self.enc1_blur(blur_img)
        enc2_blur = self.enc2_blur(self.pool(enc1_blur))
        enc3_blur = self.enc3_blur(self.pool(enc2_blur))
        return enc1_blur, enc2_blur, enc3_blur

class BlurGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, feature_dim=64):
        super(BlurGenerator, self).__init__()
        self.fe = FeatureExtractor(in_channels, feature_dim)

        # Encoder for clear image
        self.enc1 = self.conv_block(in_channels, feature_dim)
        self.enc2 = self.conv_block(feature_dim, feature_dim * 2)
        self.enc3 = self.conv_block(feature_dim * 2, feature_dim * 4)
        self.enc4 = self.conv_block(feature_dim * 4, feature_dim * 8)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(feature_dim * 8, feature_dim * 4, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(feature_dim * 8, feature_dim * 4)
        self.upconv3 = nn.ConvTranspose2d(feature_dim * 4, feature_dim * 2, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(feature_dim * 4, feature_dim * 2)
        self.upconv2 = nn.ConvTranspose2d(feature_dim * 2, feature_dim, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(feature_dim * 2, feature_dim)
        self.final_conv = nn.Conv2d(feature_dim, out_channels, kernel_size=1)

        # Modulation layers for spatially adaptive blur style
        self.mod_scale3 = nn.Conv2d(feature_dim * 4, feature_dim * 4, kernel_size=1)
        self.mod_shift3 = nn.Conv2d(feature_dim * 4, feature_dim * 4, kernel_size=1)
        self.mod_scale2 = nn.Conv2d(feature_dim * 2, feature_dim * 2, kernel_size=1)
        self.mod_shift2 = nn.Conv2d(feature_dim * 2, feature_dim * 2, kernel_size=1)
        self.mod_scale1 = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.mod_shift1 = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.sigmoid = nn.Sigmoid()

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
        # Extract blur features at multiple scales
        enc1_blur, enc2_blur, enc3_blur = self.fe(blur_img)

        # Encode clear image
        enc1 = self.enc1(clear_img)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Decoder with modulation
        dec4_up = self.upconv4(enc4)
        scale3 = self.mod_scale3(enc3_blur)
        shift3 = self.mod_shift3(enc3_blur)
        dec4_mod = dec4_up * (1 + scale3) + shift3  # Spatially adaptive modulation
        dec4 = torch.cat([dec4_mod, enc3], dim=1)  # Combine with encoder features
        dec4 = self.dec4(dec4)

        dec3_up = self.upconv3(dec4)
        scale2 = self.mod_scale2(enc2_blur)
        shift2 = self.mod_shift2(enc2_blur)
        dec3_mod = dec3_up * (1 + scale2) + shift2
        dec3 = torch.cat([dec3_mod, enc2], dim=1)
        dec3 = self.dec3(dec3)

        dec2_up = self.upconv2(dec3)
        scale1 = self.mod_scale1(enc1_blur)
        shift1 = self.mod_shift1(enc1_blur)
        dec2_mod = dec2_up * (1 + scale1) + shift1
        dec2 = torch.cat([dec2_mod, enc1], dim=1)
        dec2 = self.dec2(dec2)

        output = self.final_conv(dec2)
        return self.sigmoid(output)