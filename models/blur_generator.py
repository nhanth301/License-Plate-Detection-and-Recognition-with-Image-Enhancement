import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGenerator(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32):
        super(AttentionGenerator, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

class KernelEncoder(nn.Module):
    def __init__(self, in_channels=3, feature_dim=64, kernel_size=11):
        super(KernelEncoder, self).__init__()
        self.kernel_size = kernel_size
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim * 2, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim * 2, feature_dim * 4, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 4, kernel_size * kernel_size)
        )
        
    def forward(self, attended_blur_img):
        features = self.encoder(attended_blur_img)
        pooled_features = self.global_avg_pool(features)
        flattened_features = pooled_features.view(pooled_features.size(0), -1)
        kernel_vector = self.fc(flattened_features)
        kernel_softmax = F.softmax(kernel_vector, dim=1)
        kernel = kernel_softmax.view(-1, 1, self.kernel_size, self.kernel_size)
        return kernel

class BlurGenerator(nn.Module):
    def __init__(self, in_channels=3, feature_dim=64, kernel_size=11):
        super(BlurGenerator, self).__init__()
        self.kernel_size = kernel_size
        self.attention_gen = AttentionGenerator(in_channels=in_channels)
        self.kernel_encoder = KernelEncoder(in_channels=in_channels, feature_dim=feature_dim, kernel_size=kernel_size)
        self.padding = (self.kernel_size - 1) // 2

    def forward(self, clear_img, blur_img):
        attention_map = self.attention_gen(blur_img)
        attention_map_3_channels = attention_map.repeat(1, blur_img.size(1), 1, 1)
        attended_blur_img = blur_img * attention_map_3_channels
        
        blur_kernel = self.kernel_encoder(attended_blur_img)
        
        batch_size, num_channels, H, W = clear_img.shape
        clear_img_reshaped = clear_img.view(1, batch_size * num_channels, H, W)
        repeated_kernel = blur_kernel.repeat(1, num_channels, 1, 1)
        repeated_kernel_reshaped = repeated_kernel.view(batch_size * num_channels, 1, self.kernel_size, self.kernel_size)
        
        fake_blur_reshaped = F.conv2d(
            input=clear_img_reshaped, 
            weight=repeated_kernel_reshaped, 
            padding=self.padding, 
            groups=batch_size * num_channels
        )
        fake_blur = fake_blur_reshaped.view(batch_size, num_channels, H, W)
        
        return torch.sigmoid(fake_blur), blur_kernel