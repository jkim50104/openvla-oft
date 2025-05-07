#!/bin/bash
set -e  # ⬅️ Exit on error
# export NCCL_P2P_LEVEL=NVL

DEVICE=$1  # e.g., "0" or "0,1" or "1,2,3"
DEBUG=false

# Dynamically calculate NUM_GPU from DEVICE input
IFS=',' read -ra GPU_ARRAY <<< "$DEVICE"
NUM_GPU=${#GPU_ARRAY[@]}

############################################# set parameters
L1=true
# Determine DIFF based on L1
if [ "$L1" = false ]; then
  DIFF=true
else
  DIFF=false
fi

FILM=false

S2A_fuse=false
S2A_token=false

S2A_film=true

if [ "$S2A_fuse" = true ] || [ "$S2A_token" = true ] || [ "$S2A_film" = true ]; then
  S2A=true
else
  S2A=false
fi

S2A_use_robot_mask=false
S2A_use_lang=false
S2A_merge_masks=true
if [ "$S2A_merge_masks" = true ]; then
  S2A_use_lang=false
fi

BATCH_SIZE=8
NUM_IMAGES=1

# Construct RUN_ID_NOTE dynamically
RUN_ID_NOTE="put_X_Y"
NUM_STEPS_BEFORE_DECAY=20000 #10000
MAX_STEPS=40005 #20005
SAVE_FREQ=4000

if [ "$DEBUG" = true ]; then
  NUM_STEPS_BEFORE_DECAY=5000000
  MAX_STEPS=10000005 #20005
  SAVE_FREQ=5000000
fi

############################################# Automatically set based on the parameter settings
# Adjust batch size based on conditions
if [ "$FILM" = true ] || [ "$S2A" = true ]; then
  BATCH_SIZE=$((BATCH_SIZE - 1))
fi
if [ "$S2A_film" = true ]; then
  BATCH_SIZE=$((BATCH_SIZE - 3))
fi
if [ "$S2A_token" = true ]; then
  BATCH_SIZE=$((BATCH_SIZE - 3))
fi
if [ "$S2A" = true ] && { [ "$S2A_use_lang" = false ] || [ "$S2A_merge_masks" = true ]; }; then
  BATCH_SIZE=$((BATCH_SIZE + 1))
fi
if [ "$DIFF" = true ]; then
  BATCH_SIZE=$((BATCH_SIZE - 1))
fi
if [ "$NUM_IMAGES" -gt 1 ]; then
  BATCH_SIZE=$((BATCH_SIZE - 3))
fi
echo "[INFO] Using batch size $BATCH_SIZE!"

# Edit run id note based on the parameters
if [ "$L1" = true ]; then
  RUN_ID_NOTE="${RUN_ID_NOTE}-L1"
elif [ "$DIFF" = true ]; then
  RUN_ID_NOTE="${RUN_ID_NOTE}-DF"
fi

if [ "$FILM" = true ]; then
  RUN_ID_NOTE="${RUN_ID_NOTE}-FM"
fi

if [ "$S2A" = true ]; then
  RUN_ID_NOTE="${RUN_ID_NOTE}-S2A"
  if [ "$S2A_fuse" = true ]; then
    RUN_ID_NOTE="${RUN_ID_NOTE}_fs"
  fi
  if [ "$S2A_token" = true ]; then
    RUN_ID_NOTE="${RUN_ID_NOTE}_tk"
  fi
  if [ "$S2A_film" = true ]; then
    RUN_ID_NOTE="${RUN_ID_NOTE}_film"
  fi

  if [ "$S2A_use_robot_mask" = true ]; then
    RUN_ID_NOTE="${RUN_ID_NOTE}_Yrm"
  else
    RUN_ID_NOTE="${RUN_ID_NOTE}_Nrm"
  fi
  if [ "$S2A_use_lang" = true ]; then
    RUN_ID_NOTE="${RUN_ID_NOTE}_Yla"
  else
    RUN_ID_NOTE="${RUN_ID_NOTE}_Nla"
  fi
  if [ "$S2A_merge_masks" = true ]; then
    RUN_ID_NOTE="${RUN_ID_NOTE}_Ymg"
  else
    RUN_ID_NOTE="${RUN_ID_NOTE}_Nmg"
  fi
fi

# If use wrist image
if [ "$NUM_IMAGES" -gt 1 ]; then
  RUN_ID_NOTE="W-${RUN_ID_NOTE}"
fi

# If multi-gpu
if [ "$NUM_GPU" -gt 1 ]; then
  RUN_ID_NOTE="M-${RUN_ID_NOTE}"
fi

# DEBUG overwrite the RUN_ID_NOTE
if [ "$DEBUG" = true ]; then
  RUN_ID_NOTE="DEBUG"
fi

############################################# Train VLA!

CUDA_VISIBLE_DEVICES=${DEVICE} torchrun --standalone --nnodes 1 --nproc-per-node $NUM_GPU vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir data \
  --dataset_name ur454_dataset \
  --run_root_dir runs \
  --use_l1_regression $L1 \
  --use_diffusion $DIFF \
  --use_film $FILM \
  --use_s2a $S2A \
  --use_s2a_fuse $S2A_fuse \
  --use_s2a_token $S2A_token \
  --use_s2a_film $S2A_film\
  --use_robot_mask $S2A_use_robot_mask \
  --use_s2a_lang $S2A_use_lang\
  --s2a_merge_masks $S2A_merge_masks \
  --num_images_in_input $NUM_IMAGES \
  --use_proprio True \
  --batch_size $BATCH_SIZE \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --num_steps_before_decay $NUM_STEPS_BEFORE_DECAY \
  --max_steps $MAX_STEPS \
  --use_val_set True \
  --val_freq 2000 \
  --save_freq $SAVE_FREQ \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity robi-ai \
  --wandb_project openvla-oft_ur454 \
  --run_id_note $RUN_ID_NOTE \
  --debug $DEBUG

# scp -r "runs/openvla-7b+ur454_dataset+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--$RUN_ID_NOTE--9000_chkpt" levine:/home/jokim/projects/openvla-oft/runs/
