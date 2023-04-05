## 🍮🦙🤏Flan-Alpaca-LoRA: Instruction Tuning from Humans and Machines with Low-Rank Adaptation

This repo trains *google/flan-t5* on alpaca dataset with low-rank adaptation training method. It reduces the GPU memory needed and speeds the training.

| model                                                        | adapter_params | GPU  | time   |
| ------------------------------------------------------------ | -------------- | ---- | ------ |
| [flan-alpaca-lora-base](https://huggingface.co/reasonwang/flan-alpaca-lora-base) | 0.9M           | 3090 | 20mins |
| [flan-alpaca-lora-large](https://huggingface.co/reasonwang/flan-alpaca-lora-large) | 2.4M           | 3090 | 50mins |
| [flan-alpaca-lora-xl](https://huggingface.co/reasonwang/flan-alpaca-lora-xl) | 4.7M           | 3090 | 2.5hrs |
| [flan-alpaca-lora-xxl]([reasonwang/flan-alpaca-lora-xxl · Hugging Face](https://huggingface.co/reasonwang/flan-alpaca-lora-xxl)) | 9.4M           | 3090 | 10hrs  |

#### Dependcies

```
torch == 1.13.1
transformers == 4.27.3
peft == 0.2.0
```

#### Training

The following command finetune Flan-T5-base with only 20 mins on a single 3090 GPU

```bash
torchrun --nproc_per_node=1 --master_port=29500 train.py \
    --model_name_or_path google/flan-t5-base \
    --data_path ./alpaca_data_cleaned.json \
    --bf16 True \
    --output_dir ./ckpts/ \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
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

# Where peft_model_id should be the saving directory or huggingface model id
model_name = "google/flan-t5-large"; peft_model_id = "reasonwang/flan-alpaca-lora-large"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
base_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)
peft_model = PeftModel.from_pretrained(base_model, peft_model_id)

# Input an instruction or any other questions.
inputs = tokenizer("List a few tips to get good scores in math.", return_tensors="pt")
outputs = peft_model.generate(**inputs, max_length=128, do_sample=True)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
```

We are still improving the repo...