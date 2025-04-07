import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import sys 
import os
sys.path.append(os.path.abspath('../models'))
from base_sp_lpr import LPSR



# ----- Load model -----
model = LPSR(num_channels=3, num_features=124, growth_rate=64, num_blocks=8, num_layers=4, scale_factor=2)
model.load_state_dict(torch.load("../weights/best_model.pth", map_location=torch.device("cpu")))
model.eval()

# ----- Hàm tiền xử lý ảnh -----
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((96, 32))  # Đảm bảo ảnh có kích thước 128x128
    # Chuyển đổi ảnh thành tensor
    transform = transforms.Compose([
        transforms.ToTensor(),  # scale về [0, 1]
    ])
    img_tensor = transform(img).unsqueeze(0)  # thêm batch dimension
    return img_tensor, img

# ----- Hàm hậu xử lý ảnh -----
def postprocess_image(tensor):
    tensor = tensor.squeeze(0).clamp(0, 1)  # bỏ batch dim và giới hạn giá trị
    img = transforms.ToPILImage()(tensor)
    return img

# ----- Inference -----
def infer(image_path):
    input_tensor, original_image = preprocess_image(image_path)
    with torch.no_grad():
        output_tensor = model(input_tensor)

    output_image = postprocess_image(output_tensor)
    print(output_image.size)
    output_image.save('new_image.png')  # Lưu ảnh đầu ra    
    # Hiển thị ảnh trước và sau
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Input")
    plt.imshow(original_image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Super-resolved Output")
    plt.imshow(output_image)
    plt.axis("off")

    plt.show()

# ----- Gọi hàm inference -----
infer("../data/test/plate_image copy 3.png_8.jpg")  # Thay bằng đường dẫn ảnh thật
