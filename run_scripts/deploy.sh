DEVICE=$1

CKPT_PATH="runs/pb_pf-L1/openvla-7b+ur454_dataset+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--pb_pf-L1--20000_chkpt"

CUDA_VISIBLE_DEVICES=${DEVICE} python vla-scripts/deploy.py \
  --pretrained_checkpoint $CKPT_PATH \
  --use_l1_regression True \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio True \
  --center_crop True \
  --num_open_loop_steps 10 \
  --unnorm_key ur454_dataset