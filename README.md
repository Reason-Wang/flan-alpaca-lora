

```bash
torchrun --nproc_per_node=1 --master_port=29500 train.py \
    --model_name_or_path google/flan-t5-base \
    --data_path ./alpaca_data_cleaned.json \
    --bf16 True \
    --output_dir ./ckpts/cleaned_data/ \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 5e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 50 \
    --tf32 True
```

