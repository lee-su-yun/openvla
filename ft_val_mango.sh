
  torchrun \
  --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_with_val.py \
  --vla_path "/ckpt/openvla-7b" \
  --data_root_dir "/home/sylee/tensorflow_datasets" \
  --dataset_name "piper5_hz_subtask" \
  --run_root_dir "/ckpt/piper_subtask/openvla/Fully" \
  --adapter_tmp_dir "/ckpt/piper_subtask/openvla/Fully" \
  --lora_rank 32 \
  --batch_size 8 \
  --grad_accumulation_steps 2 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project "openvla" \
  --wandb_entity "suyunlee-seoul-national-university" \
  --save_steps 1000 \
  --max_steps 10000 \

