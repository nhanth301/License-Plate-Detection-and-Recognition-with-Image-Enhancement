# file: utils/export_onnx.py
import sys
import os
import argparse
import torch
from my_models.detection import Detection
from my_models.lpsr import LPSR


def main():
    parser = argparse.ArgumentParser(
        description="Export PyTorch models to ONNX format."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["sr", "detection", "ocr"],
        help="The type of model to export: 'sr', 'detection', or 'ocr'.",
    )   
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the input pretrained weights file (.pt or .pth).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path for the output .onnx file.",
    )
    args = parser.parse_args()
    device = torch.device("cpu")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.model_type == "sr":
        print("Exporting Super-Resolution Model...")
        model = LPSR(
            num_channels=3,
            num_features=32,
            growth_rate=16,
            num_blocks=4,
            num_layers=4,
            scale_factor=None,
        ).to(device)
        model.load_state_dict(torch.load(args.weights, map_location=device))
        model.eval()
        dummy_input = torch.randn(1, 3, 32, 192, device=device)

        torch.onnx.export(
            model,
            dummy_input,
            args.output,
            input_names=["input_image"],
            output_names=["output_image"],
            opset_version=16,
            do_constant_folding=True,
            dynamic_axes={
                "input_image": {0: "batch_size", 2: "height", 3: "width"},
                "output_image": {0: "batch_size", 2: "height", 3: "width"},
            },
        )

    elif args.model_type in ["detection", "ocr"]:
        print(f"Exporting {args.model_type.upper()} Model...")
        if args.model_type == "detection":
            size = [1280, 1280]
            conf_thres = 0.7
            iou_thres = 0.3
        else:  # ocr
            size = [128, 128]
            conf_thres = 0.25
            iou_thres = 0.3

        model = Detection(
            size=size,
            weights_path=args.weights,
            device='cpu',
            iou_thres=iou_thres,
            conf_thres=conf_thres,
        ).char_model
        model.eval()
        dummy_input = torch.randn(1, 3, size[0], size[1], device=device)

        torch.onnx.export(
            model,
            dummy_input,
            args.output,
            input_names=["input_image"],
            output_names=["predictions"],
            opset_version=16,
            do_constant_folding=True,
            dynamic_axes={
                "input_image": {0: "batch_size"},
                "predictions": {0: "batch_size"},
            },
        )
    else:
        print(f"Error: Unknown model type '{args.model_type}'")
        return

    print(f"Model successfully saved to {args.output}")


if __name__ == "__main__":
    main()