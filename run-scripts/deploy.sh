DEVICE=$1

############################################# set parameters
L1=true
# Determine DIFF based on L1
if [ "$L1" = false ]; then
  DIFF=true
else
  DIFF=false
fi

FILM=true

S2A_fuse=false
S2A_token=false
S2A_film=false
if [ "$S2A_fuse" = true ] || [ "$S2A_token" = true ] || [ "$S2A_film" = true ]; then
  S2A=true
else
  S2A=false
fi

S2A_use_robot_mask=true
S2A_use_lang=false
S2A_merge_masks=true
if [ "$S2A_merge_masks" = true ]; then
  S2A_use_lang=false
fi

BATCH_SIZE=8
NUM_IMAGES=1

# Construct RUN_ID_NOTE dynamically
RUN_ID_NOTE="put_X_Y"
EPOCH="45000_chkpt"

############################################# Automatically set based on the parameter settings
# Adjust batch size based on conditions
if [ "$FILM" = true ] || [ "$S2A" = true ]; then
  BATCH_SIZE=$((BATCH_SIZE - 1))
fi
if [ "$S2A" = true ]; then
  if [ "$S2A_film" = true ]; then
    BATCH_SIZE=$((BATCH_SIZE - 3))
  fi
  if [ "$S2A_token" = true ]; then
    BATCH_SIZE=$((BATCH_SIZE - 3))
  fi
  if [ "$S2A" = true ] || [ "$S2A_use_lang" = false ] || [ "$S2A_merge_masks" = true ]; then
    BATCH_SIZE=$((BATCH_SIZE + 1))
  fi
  if [ "$DIFF" = true ]; then
    BATCH_SIZE=$((BATCH_SIZE - 1))
  fi
  if [ "$NUM_IMAGES" -gt 1 ]; then
    BATCH_SIZE=$((BATCH_SIZE - 3))
  fi
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

echo "[INFO] Model: $RUN_ID_NOTE"

CKPT_PATH="runs/$RUN_ID_NOTE/openvla-7b+ur454_dataset+b$BATCH_SIZE+lr-0.0005+lora-r32+dropout-0.0--image_aug--$RUN_ID_NOTE--$EPOCH"
echo "[INFO] Loading: $CKPT_PATH"
# CKPT_PATH="runs/pb_bf-DF/openvla-7b+ur454_dataset+b6+lr-0.0005+lora-r32+dropout-0.0--image_aug--pb_pf-DF--20000_chkpt"

CUDA_VISIBLE_DEVICES=${DEVICE} python vla-scripts/deploy.py \
  --pretrained_checkpoint $CKPT_PATH \
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
  --center_crop True \
  --num_open_loop_steps 10 \
  --unnorm_key ur454_dataset