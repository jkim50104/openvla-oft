SCP_PATH="/home/robi/PycharmProjects/openvla-oft/runs"
CKPT_NAME="openvla-7b+ur454_dataset+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--pb_pf--10000_chkpt"

LORA_FINETUNED="openvla-7b+ur454_dataset+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--10_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state--9000_chkpt"

scp -r dia:"$SCP_PATH/$CKPT_NAME" ./runs

python vla-scripts/merge_lora_weights_and_save.py \
    --base_checkpoint openvla/openvla-7b \
    --lora_finetuned_checkpoint_dir "./runs/$CKPT_NAME"