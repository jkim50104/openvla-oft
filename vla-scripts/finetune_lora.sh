set -e  # ⬅️ Exit on error

# RUN_ID_NOTE="parallel_dec--10_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state"
RUN_ID_NOTE="pb_pf"

# export NCCL_P2P_LEVEL=NVL

DEVICE="1" #0,1
CUDA_VISIBLE_DEVICES=${DEVICE} torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir data \
  --dataset_name ur454_dataset \
  --run_root_dir runs \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 1 \
  --grad_accumulation_steps 4 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 5000 \
  --max_steps 10005 \
  --use_val_set True \
  --val_freq 1000 \
  --save_freq 1000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity robi-ai \
  --wandb_project openvla-oft_ur454 \
  --run_id_note $RUN_ID_NOTE

  # Use film if there are multiple language types

  scp -r "runs/openvla-7b+ur454_dataset+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--$RUN_ID_NOTE--9000_chkpt" levine:/home/jokim/projects/openvla-oft/runs/
  
