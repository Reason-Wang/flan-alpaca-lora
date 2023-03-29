## 🍮🦙🤏Flan-Alpaca-LoRA: Instruction Tuning from Humans and Machines with Low-Rank Adaptation

This repo trains *google/flan-t5* on alpaca dataset with low-rank adaptation training method. It reduces the GPU memory needed and speeds the training.

#### Dependcies

```
torch == 1.13.1
transformers == 4.27.1
peft == 0.2.0
```



#### Training

The following command finetune Flan-T5-base with only 30mins on a single 3090 GPU

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

Example usage:

```python
import transformers
from peft import PeftModel

base_model = transformers.AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
peft_model = PeftModel.from_pretrained(base_model, "./ckpts/cleaned_data/")

inputs = tokenizer("Any instruction that you like.", return_tensors="pt")
outputs = peft_model.generate(**inputs, max_length=128, do_sample=True)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
```



We are still improving the repo...