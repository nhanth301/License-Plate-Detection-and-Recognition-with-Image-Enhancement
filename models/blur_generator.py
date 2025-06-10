import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

def create_gaussian_kernel(params, kernel_size):
    batch_size = params.size(0)
    device = params.device
    
    mu_x = params[:, 0].unsqueeze(-1).unsqueeze(-1) * (kernel_size // 4)
    mu_y = params[:, 1].unsqueeze(-1).unsqueeze(-1) * (kernel_size // 4)
    sigma_x = (torch.sigmoid(params[:, 2]) * (kernel_size / 2) + 0.5).unsqueeze(-1).unsqueeze(-1)
    sigma_y = (torch.sigmoid(params[:, 3]) * (kernel_size / 2) + 0.5).unsqueeze(-1).unsqueeze(-1)
    theta = (torch.tanh(params[:, 4]) * math.pi).unsqueeze(-1).unsqueeze(-1)

    ax = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32, device=device)
    xx, yy = torch.meshgrid(ax, ax, indexing='xy')
    xx = xx.expand(batch_size, -1, -1)
    yy = yy.expand(batch_size, -1, -1)
    
    x = xx - mu_x
    y = yy - mu_y

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    x_rot = x * cos_t + y * sin_t
    y_rot = -x * sin_t + y * cos_t

    a = 1.0 / (2.0 * sigma_x**2 + 1e-6)
    b = 1.0 / (2.0 * sigma_y**2 + 1e-6)
    exponent = - (a * x_rot**2 + b * y_rot**2)
    

    exponent = torch.clamp(exponent, max=0.0)
    
    kernel = torch.exp(exponent)

    kernel = kernel / torch.sum(kernel, dim=[1, 2], keepdim=True)
    
    return kernel.unsqueeze(1)

class KernelParameterEncoder(nn.Module):
    def __init__(self, in_channels=3, feature_dim=64, num_kernels=5, params_per_kernel=5):
        super(KernelParameterEncoder, self).__init__()
        self.num_kernels = num_kernels
        self.params_per_kernel = params_per_kernel
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim * 2, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim * 2, feature_dim * 4, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_params = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 4, num_kernels * params_per_kernel)
        )
        self.fc_weights = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, num_kernels)
        )
        
    def forward(self, attended_blur_img):
        features = self.encoder(attended_blur_img)
        pooled_features = self.global_avg_pool(features)
        flattened_features = pooled_features.view(pooled_features.size(0), -1)
        raw_params = self.fc_params(flattened_features)
        raw_weights = self.fc_weights(flattened_features)
        normalized_weights = F.softmax(raw_weights, dim=1)
        return raw_params, normalized_weights

class BlurGeneratorMixture(nn.Module):
    def __init__(self, in_channels=3, feature_dim=64, kernel_size=11, num_kernels=5):
        super(BlurGeneratorMixture, self).__init__()
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.params_per_kernel = 5
        self.attention_gen = AttentionGenerator(in_channels=in_channels)
        self.param_encoder = KernelParameterEncoder(
            in_channels=in_channels, 
            feature_dim=feature_dim, 
            num_kernels=num_kernels,
            params_per_kernel=self.params_per_kernel
        )
        self.padding = (self.kernel_size - 1) // 2

    def forward(self, clear_img, blur_img):
        batch_size = blur_img.size(0)
        attention_map = self.attention_gen(blur_img)
        attention_map_3_channels = attention_map.repeat(1, blur_img.size(1), 1, 1)
        attended_blur_img = blur_img * attention_map_3_channels
        raw_params, weights = self.param_encoder(attended_blur_img)
        params = raw_params.view(batch_size, self.num_kernels, self.params_per_kernel)
        final_kernel = torch.zeros(batch_size, 1, self.kernel_size, self.kernel_size).to(blur_img.device)
        
        for i in range(self.num_kernels):
            kernel_params_i = params[:, i, :]
            weight_i = weights[:, i].view(-1, 1, 1, 1)
            basis_kernel_i = create_gaussian_kernel(kernel_params_i, self.kernel_size)
            final_kernel += weight_i * basis_kernel_i
            
        _, num_channels, H, W = clear_img.shape
        clear_img_reshaped = clear_img.view(1, batch_size * num_channels, H, W)
        repeated_kernel = final_kernel.repeat(1, num_channels, 1, 1)
        repeated_kernel_reshaped = repeated_kernel.view(batch_size * num_channels, 1, self.kernel_size, self.kernel_size)
        fake_blur_reshaped = F.conv2d(
            input=clear_img_reshaped, 
            weight=repeated_kernel_reshaped, 
            padding=self.padding, 
            groups=batch_size * num_channels
        )
        fake_blur = fake_blur_reshaped.view(batch_size, num_channels, H, W)
        return torch.sigmoid(fake_blur), final_kernel