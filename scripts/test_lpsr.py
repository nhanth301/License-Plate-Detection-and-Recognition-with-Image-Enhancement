import argparse
import os
import sys
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from loguru import logger

sys.path.append(os.path.abspath('../models'))
from detection import Detection
from base_sp_lpr import LPSR


def load_image(image_path):
    """
    Load and preprocess an image for super-resolution inference.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        torch.Tensor: Tensor of shape (1, 3, 32, 96), normalized to [0, 1].

    Example:
        >>> tensor = load_image("example.jpg")
    """
    image = Image.open(image_path).convert("RGB").resize((96, 32), Image.BICUBIC)
    transform = T.Compose([T.ToTensor()])
    return transform(image).unsqueeze(0)


def parse_opt():
    """
    Parse command-line arguments for the character recognition pipeline.

    Returns:
        argparse.Namespace: Parsed arguments including weights path, input/output directories, 
        confidence threshold, image size, device, etc.
    """
    parser = argparse.ArgumentParser(description="Character recognition from license plate images")
    parser.add_argument('--weights', nargs='+', type=str, default='char.pt',
                        help='Model path or Triton URL')
    parser.add_argument('--source', type=str, default='ch_imgs',
                        help='Input image directory')
    parser.add_argument('--des', type=str, default='ch_out',
                        help='Output image directory')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[128],
                        help='Inference image size (h, w)')
    parser.add_argument('--conf-thres', type=float, default=0.1,
                        help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3,
                        help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000,
                        help='Maximum detections per image')
    parser.add_argument('--device', default='cpu',
                        help='CUDA device, e.g. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--keep-size', action='store_true',
                        help='Keep original image size in output')
    parser.add_argument('--sr', action='store_true',
                        help='Enable Super Resolution mode')

    opt = parser.parse_args()
    if len(opt.imgsz) == 1:
        opt.imgsz = opt.imgsz * 2
    return opt


def is_image_file(filename):
    """
    Check if a file is a valid image file based on its extension.

    Args:
        filename (str): The name of the file.

    Returns:
        bool: True if file is an image, False otherwise.
    """
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))


def main():
    """
    Main execution function for character recognition.

    Loads a character detection model and optionally a super-resolution model, processes
    images from the specified folder, detects characters, and calculates accuracy metrics.
    """
    folder = "../new_legible"
    opt = parse_opt()
    os.makedirs(opt.des, exist_ok=True)

    char_model = Detection(
        size=opt.imgsz,
        weights_path=opt.weights,
        device=opt.device,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres
    )

    total_chars = 0
    correct_chars = 0
    correct_plates = 0
    total_plates = 0

    if opt.sr:
        logger.info("🧠 Running in Super Resolution (SR) mode")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LPSR(
            num_channels=3,
            num_features=124,
            growth_rate=64,
            num_blocks=8,
            num_layers=4,
            scale_factor=2
        ).to(device)

        checkpoint = torch.load('../weights/best_model.pth', map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()

        with torch.no_grad():
            for filename in os.listdir(folder):
                if not is_image_file(filename) or total_plates >= 1000:
                    continue

                image_path = os.path.join(folder, filename)
                image = load_image(image_path).to(device)
                sr_tensor = model(image).squeeze(0).cpu().clamp(0, 1)

                sr_np = sr_tensor.permute(1, 2, 0).numpy()
                sr_np = (sr_np * 255).astype('uint8')
                hr_plate_img = cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)

                plate_img_resized = char_model.ResizeImg(hr_plate_img, size=(128, 128))
                results, _ = char_model.detect(plate_img_resized.copy())

                pred_chars = ''
                if results:
                    results = sorted(results, key=lambda x: x[2][0])
                    pred_chars = ''.join([res[0] for res in results]).upper()

                gt = os.path.splitext(filename)[0].upper()
                match_count = sum(1 for p, g in zip(pred_chars, gt) if p == g)
                correct_chars += match_count
                total_chars += max(len(gt), len(pred_chars))

                if pred_chars == gt:
                    correct_plates += 1
                else:
                    print(f"[WRONG] {filename} | GT: {gt} | Pred: {pred_chars}")

                total_plates += 1

    else:
        logger.info("⚡ Running in Normal (No SR) mode")

        for filename in os.listdir(folder):
            if not is_image_file(filename) or total_plates >= 1000:
                continue

            image_path = os.path.join(folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"[WARNING] Cannot read image: {filename}")
                continue

            image = cv2.resize(image, (192, 64))
            plate_img_resized = char_model.ResizeImg(image, size=(128, 128))
            results, _ = char_model.detect(plate_img_resized.copy())

            pred_chars = ''
            if results:
                results = sorted(results, key=lambda x: x[2][0])
                pred_chars = ''.join([res[0] for res in results]).upper()

            gt = os.path.splitext(filename)[0].upper()
            match_count = sum(1 for p, g in zip(pred_chars, gt) if p == g)
            correct_chars += match_count
            total_chars += max(len(gt), len(pred_chars))

            if pred_chars == gt:
                correct_plates += 1
            else:
                print(f"[WRONG] {filename} | GT: {gt} | Pred: {pred_chars}")

            total_plates += 1

    char_acc = correct_chars / total_chars * 100 if total_chars else 0
    plate_acc = correct_plates / total_plates * 100 if total_plates else 0

    mode_str = "(no SR)" if not opt.sr else ""
    print(f"\n✅ Character Accuracy {mode_str}: {correct_chars}/{total_chars} ({char_acc:.2f}%)")
    print(f"✅ Plate Accuracy {mode_str}: {correct_plates}/{total_plates} ({plate_acc:.2f}%)")


if __name__ == '__main__':
    main()
