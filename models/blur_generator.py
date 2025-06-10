import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class HybridParameterEncoder(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256]):
        super(HybridParameterEncoder, self).__init__()
        
        self.downs = nn.ModuleList()
        for feature in features:
            self.downs.append(ConvBlock(in_channels, feature))
            in_channels = feature
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)
        
        self.global_param_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(features[-1] * 2, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1 + 1 + 2)
        )

        self.ups = nn.ModuleList()
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(ConvBlock(feature * 2, feature))
            
        self.final_flow_conv = nn.Conv2d(features[0], 2, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        x = self.bottleneck(x)
        
        global_params = self.global_param_head(x)
        
        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i//2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i+1](concat_skip)
            
        flow_field = self.final_flow_conv(x)
        
        return flow_field, global_params

class HybridBlurGenerator(nn.Module):
    def __init__(self, in_channels=3, kernel_size=11, num_flow_steps=7):
        super(HybridBlurGenerator, self).__init__()
        self.param_encoder = HybridParameterEncoder(in_channels=in_channels)
        self.kernel_size = kernel_size
        self.num_flow_steps = num_flow_steps

    def apply_flow_blur(self, img, flow_field):
        B, C, H, W = img.shape
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H, device=img.device), 
                                        torch.linspace(-1, 1, W, device=img.device), indexing='ij')
        grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0).repeat(B, 1, 1, 1)

        warped_images = []
        for t in range(self.num_flow_steps):
            step = (t + 1) / self.num_flow_steps
            scaled_flow = flow_field.permute(0, 2, 3, 1) * step
            new_grid = grid + scaled_flow
            warped_img = F.grid_sample(img, new_grid, mode='bilinear', padding_mode='border', align_corners=True)
            warped_images.append(warped_img)
        
        return torch.mean(torch.stack(warped_images), dim=0)

    def apply_focus_blur(self, img, kernel_sigma):
        batch_size, num_channels, H, W = img.shape
        
        kernel = self.create_simple_gaussian_kernel(kernel_sigma, self.kernel_size, img.device)
        
        padding = (self.kernel_size - 1) // 2
        
        img_reshaped = img.view(1, batch_size * num_channels, H, W)
        kernel_reshaped = kernel.repeat(1, num_channels, 1, 1).view(batch_size * num_channels, 1, self.kernel_size, self.kernel_size)
        
        blurred_reshaped = F.conv2d(
            input=img_reshaped,
            weight=kernel_reshaped,
            padding=padding,
            groups=batch_size * num_channels
        )
        
        return blurred_reshaped.view(batch_size, num_channels, H, W)

    def create_simple_gaussian_kernel(self, sigma, kernel_size, device):
        # Input sigma shape: [B, 1, 1, 1]
        
        ax = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32, device=device)
        xx, yy = torch.meshgrid(ax, ax, indexing='xy')
        
        # grid shape: [k, k]
        grid = xx**2 + yy**2
        
        # sigma_reshaped: [B, 1, 1]
        sigma_reshaped = sigma.view(-1, 1, 1)

        # Broadcasting [k, k] với [B, 1, 1] -> kết quả là [B, k, k]
        exponent = -grid / (2 * sigma_reshaped**2 + 1e-6)
        exponent = torch.clamp(exponent, max=0.0)
        
        kernel = torch.exp(exponent)
        
        # Chuẩn hóa từng kernel trong batch
        kernel_sum = torch.sum(kernel, dim=[1, 2], keepdim=True)
        kernel = kernel / (kernel_sum + 1e-6)
        
        # Trả về tensor 4D [B, 1, k, k] để tương thích với F.conv2d
        return kernel.unsqueeze(1)

    def forward(self, clear_img, blur_img):
        flow_field, global_params = self.param_encoder(blur_img)
        
        noise_sigma = torch.sigmoid(global_params[:, 0]).view(-1, 1, 1, 1) * 0.1
        kernel_sigma = (torch.sigmoid(global_params[:, 1]) * 2.0 + 0.1).view(-1, 1, 1, 1)
        color_alpha = (global_params[:, 2] * 0.2 + 1.0).view(-1, 1, 1, 1)
        color_beta = (global_params[:, 3] * 0.1).view(-1, 1, 1, 1)

        x = clear_img
        x = self.apply_flow_blur(x, flow_field)
        x = self.apply_focus_blur(x, kernel_sigma)
        noise = torch.randn_like(x) * noise_sigma
        x = x + noise
        x = x * color_alpha + color_beta
        
        fake_blur = torch.sigmoid(x)

        degradation_params = {"flow": flow_field, "noise": noise_sigma, "kernel_sigma": kernel_sigma}
        
        return fake_blur, degradation_params