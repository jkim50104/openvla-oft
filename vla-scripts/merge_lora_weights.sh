LORA_FINETUNED="runs/openvla-7b+ur454_dataset+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--10_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state--9000_chkpt"

python vla-scripts/merge_lora_weights_and_save.py \
    --base_checkpoint openvla/openvla-7b \
    --lora_finetuned_checkpoint_dir $LORA_FINETUNED