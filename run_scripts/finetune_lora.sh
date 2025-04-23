#!/bin/bash
set -e  # ⬅️ Exit on error
# export NCCL_P2P_LEVEL=NVL

DEVICE=$1  # e.g., "0" or "0,1" or "1,2,3"

# Dynamically calculate NUM_GPU from DEVICE input
IFS=',' read -ra GPU_ARRAY <<< "$DEVICE"
NUM_GPU=${#GPU_ARRAY[@]}
echo $NUM_GPU

L1=false
FILM=true
BATCH_SIZE=8

# Determine DIFF based on L1
if [ "$L1" = false ]; then
  DIFF=true
else
  DIFF=false
fi

# Adjust BATCH_SIZE if FILM or DIFF is true
if [ "$FILM" = true ] || [ "$DIFF" = true ]; then
  BATCH_SIZE=6
fi

# Construct RUN_ID_NOTE dynamically
RUN_ID_NOTE="pb_pf"

if [ "$L1" = true ]; then
  RUN_ID_NOTE="${RUN_ID_NOTE}-L1"
elif [ "$DIFF" = true ]; then
  RUN_ID_NOTE="${RUN_ID_NOTE}-DF"
fi

if [ "$FILM" = true ]; then
  RUN_ID_NOTE="${RUN_ID_NOTE}-FM"
fi

if [ "$NUM_GPU" -gt 1 ]; then
  RUN_ID_NOTE="M-${RUN_ID_NOTE}"
fi

CUDA_VISIBLE_DEVICES=${DEVICE} torchrun --standalone --nnodes 1 --nproc-per-node $NUM_GPU vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir data \
  --dataset_name ur454_dataset \
  --run_root_dir runs \
  --use_l1_regression $L1 \
  --use_diffusion $DIFF \
  --use_film $FILM \
  --num_images_in_input 1 \
  --use_proprio True \
  --batch_size $BATCH_SIZE \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 10000 \
  --max_steps 20005 \
  --use_val_set True \
  --val_freq 2000 \
  --save_freq 2000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity robi-ai \
  --wandb_project openvla-oft_ur454 \
  --run_id_note $RUN_ID_NOTE

# scp -r "runs/openvla-7b+ur454_dataset+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--$RUN_ID_NOTE--9000_chkpt" levine:/home/jokim/projects/openvla-oft/runs/
