
  torchrun \
  --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_gripper.py \
  --vla_path "/sdc1/piper_subtask/openvla/openvla-7b+piper5_hz_subtask+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug" \
  --data_root_dir "/home/sylee/tensorflow_datasets" \
  --dataset_name "piper5_hz_subtask" \
  --run_root_dir "/sdc1/piper_subtask/openvla/gripper" \
  --adapter_tmp_dir "/sdc1/piper_subtask/openvla/gripper" \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project "openvla" \
  --wandb_entity "suyunlee-seoul-national-university" \
  --save_steps 1000 \
  --max_steps 10000 \
