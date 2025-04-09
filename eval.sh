python experiments/robot/ur454/run_ur454_eval.py \
  --use_vla_server False \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b \
  --use_l1_regression True \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --center_crop True \
  --num_open_loop_steps 10 \
  --num_rollouts_planned 50 \
  --max_steps 1000 \
  --unnorm_key bridge_orig

# python experiments/robot/ur454/run_ur454_eval.py \
#   --model_family openvla \
#   --pretrained_checkpoint runs/openvla-7b+ur454_dataset+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug \
#   --unnorm_key ur454_dataset \
#   # --sanity_check True
