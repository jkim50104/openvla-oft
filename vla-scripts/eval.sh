# python experiments/robot/ur454/run_ur454_eval.py \
#   --use_vla_server False \
#   --model_family openvla \
#   --pretrained_checkpoint openvla/openvla-7b \
#   --use_l1_regression True \
#   --use_film False \
#   --num_images_in_input 2 \
#   --use_proprio True \
#   --center_crop True \
#   --num_open_loop_steps 10 \
#   --num_rollouts_planned 50 \
#   --max_steps 1000 \
#   --unnorm_key bridge_orig

CKPT="runs/openvla-7b+ur454_dataset+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--pb_pf--10000_chkpt"
# CKPT="/home/jokim/projects/openvla-oft/runs/openvla-7b+ur454_dataset+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--10_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state--1000_chkpt"

python experiments/robot/ur454/run_ur454_eval.py \
  --use_vla_server False \
  --model_family openvla \
  --pretrained_checkpoint $CKPT \
  --use_l1_regression True \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --center_crop True \
  --num_open_loop_steps 10 \
  --num_rollouts_planned 50 \
  --max_steps 1000 \
  --unnorm_key ur454_dataset \
#   # --sanity_check True
