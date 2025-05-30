CUDA_VISIBLE_DEVICES=1,2,3,4,5 torchrun \
  --standalone --nnodes 1 --nproc-per-node 5 vla-scripts/finetune_fully.py \
  --vla_path "/ckpt/openvla-7b" \
  --data_root_dir "/home/sylee/tensorflow_datasets" \
  --dataset_name "piper5_hz_subtask" \
  --run_root_dir "/ckpt/piper_subtask/openvla/FullFT_table" \
  --batch_size 1 \
  --grad_accumulation_steps 16 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project "openvla" \
  --wandb_entity "suyunlee-seoul-national-university" \
  --save_steps 1000 \
  --max_steps 10000 \
  --use_lora False \
  --use_quantization True