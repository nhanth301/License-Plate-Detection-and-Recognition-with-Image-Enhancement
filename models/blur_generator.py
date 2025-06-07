import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaIN(nn.Module):
    def __init__(self):
        super(AdaIN, self).__init__()

    def forward(self, content, style_mean, style_std):
       
        content_mean = torch.mean(content, dim=[2, 3], keepdim=True)
        content_std = torch.std(content, dim=[2, 3], keepdim=True)
        normalized_content = (content - content_mean) / (content_std + 1e-5)
        return style_std * normalized_content + style_mean

class StyleEncoder(nn.Module):
    def __init__(self, in_channels=3, feature_dim=64):
        super(StyleEncoder, self).__init__()
        self.enc1 = self.conv_block(in_channels, feature_dim)          # 64 kênh
        self.enc2 = self.conv_block(feature_dim, feature_dim * 2)     # 128 kênh
        self.enc3 = self.conv_block(feature_dim * 2, feature_dim * 4) # 256 kênh
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
        enc1 = self.enc1(blur_img)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
     
        style_mean1 = torch.mean(enc1, dim=[2, 3], keepdim=True)  # [B, 64, 1, 1]
        style_std1 = torch.std(enc1, dim=[2, 3], keepdim=True)
        style_mean2 = torch.mean(enc2, dim=[2, 3], keepdim=True)  # [B, 128, 1, 1]
        style_std2 = torch.std(enc2, dim=[2, 3], keepdim=True)
        style_mean3 = torch.mean(enc3, dim=[2, 3], keepdim=True)  # [B, 256, 1, 1]
        style_std3 = torch.std(enc3, dim=[2, 3], keepdim=True)
        return (style_mean1, style_std1), (style_mean2, style_std2), (style_mean3, style_std3)

class ContentEncoder(nn.Module):
    def __init__(self, in_channels=3, feature_dim=64):
        super(ContentEncoder, self).__init__()
        self.enc1 = self.conv_block(in_channels, feature_dim)
        self.enc2 = self.conv_block(feature_dim, feature_dim * 2)
        self.enc3 = self.conv_block(feature_dim * 2, feature_dim * 4)
        self.enc4 = self.conv_block(feature_dim * 4, feature_dim * 8)
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

    def forward(self, clear_img):
        enc1 = self.enc1(clear_img)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        return enc1, enc2, enc3, enc4

class BlurGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, feature_dim=64):
        super(BlurGenerator, self).__init__()
        self.style_encoder = StyleEncoder(in_channels, feature_dim)
        self.content_encoder = ContentEncoder(in_channels, feature_dim)  

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(feature_dim * 8, feature_dim * 4, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(feature_dim * 8, feature_dim * 4)  # 256
        self.upconv3 = nn.ConvTranspose2d(feature_dim * 4, feature_dim * 2, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(feature_dim * 4, feature_dim * 2)  # 128 
        self.upconv2 = nn.ConvTranspose2d(feature_dim * 2, feature_dim, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(feature_dim * 2, feature_dim)      # 64 
        self.final_conv = nn.Conv2d(feature_dim, out_channels, kernel_size=1)

        self.adain = AdaIN()
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
        (style_mean1, style_std1), (style_mean2, style_std2), (style_mean3, style_std3) = self.style_encoder(blur_img)
        enc1, enc2, enc3, enc4 = self.content_encoder(clear_img)

        dec4_up = self.upconv4(enc4)
        dec4_mod = self.adain(dec4_up, style_mean3, style_std3)  # 256 
        dec4 = torch.cat([dec4_mod, enc3], dim=1)
        dec4 = self.dec4(dec4)

        dec3_up = self.upconv3(dec4)
        dec3_mod = self.adain(dec3_up, style_mean2, style_std2)  # 128 
        dec3 = torch.cat([dec3_mod, enc2], dim=1)
        dec3 = self.dec3(dec3)

        dec2_up = self.upconv2(dec3)
        dec2_mod = self.adain(dec2_up, style_mean1, style_std1)  # 64 
        dec2 = torch.cat([dec2_mod, enc1], dim=1)
        dec2 = self.dec2(dec2)

        output = self.final_conv(dec2)
        return self.sigmoid(output)