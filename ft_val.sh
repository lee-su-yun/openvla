
  torchrun \
  --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_with_val.py \
  --vla_path "/sda1/openvla-7b" \
  --data_root_dir "/home/sylee/tensorflow_datasets" \
  --dataset_name "piper5_hz_subtask" \
  --run_root_dir "/sdc1/piper_subtask/openvla/Norm" \
  --adapter_tmp_dir "/sdc1/piper_subtask/openvla/Norm" \
  --lora_rank 32 \
  --batch_size 2 \
  --grad_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project "openvla" \
  --wandb_entity "suyunlee-seoul-national-university" \
  --save_steps 1000 \
  --max_steps 10000 \

d