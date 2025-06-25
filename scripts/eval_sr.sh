python -m evaluation.eval \
    --sr-weights weights/best_model.pth \
    --ocr-weights weights/char.pt \
    --eval-folder "/home/nhan/Desktop/Plate/iqa_training_v2.1/origin_clear" \
    --imgsz-ocr 128 128 \
    --ocr-conf 0.45 \
    --iou 0.3 \
    --device cuda
