python -m train.cyclegans \
    --dataroot /home/nhan/Desktop/Plate/iqa_training_v2.1/cycle_data \
    --epochs 100 \
    --batch_size 1 \
    --lr 0.0001 \
    --width 192 \
    --height 32 \
    --checkpoint_dir output/checkpoints \
    --output_dir output/visualizations
#     --hr_train_dir /home/nhan/Downloads/archive/