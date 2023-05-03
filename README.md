## 🍮🦙🤏Flan-Alpaca-LoRA: Instruction Tuning from Humans and Machines with Low-Rank Adaptation

This repo trains *google/flan-t5* on alpaca dataset with low-rank adaptation training method. It reduces the GPU memory needed and speeds the training.

May 3, 2023: train flan-t5-xl using alpaca-gpt4 dataset.

Apr 13, 2023: train flan-t5-xl using GPTeacher dataset (Instruct and Roleplay), which seems to perform well.

Apr 5, 2023: train flan-t5-xxl using 8bit quantization. The model can be fitted into a single 3090 GPU. All of the models can be found in huggingface.

| model                                                        | adapter_params | data                                                         | GPU  | time   |
| ------------------------------------------------------------ | -------------- | ------------------------------------------------------------ | ---- | ------ |
| [flan-alpaca-lora-base](https://huggingface.co/reasonwang/flan-alpaca-lora-base) | 0.9M           | [alpaca cleaned](https://github.com/gururise/AlpacaDataCleaned) | 3090 | 20mins |
| [flan-alpaca-lora-large](https://huggingface.co/reasonwang/flan-alpaca-lora-large) | 2.4M           | [alpaca cleaned](https://github.com/gururise/AlpacaDataCleaned) | 3090 | 50mins |
| [flan-alpaca-lora-xl](https://huggingface.co/reasonwang/flan-alpaca-lora-xl) | 4.7M           | [alpaca cleaned](https://github.com/gururise/AlpacaDataCleaned) | 3090 | 2.5hrs |
| [flan-alpaca-lora-xxl](https://huggingface.co/reasonwang/flan-alpaca-lora-xxl) | 9.4M           | [alpaca cleaned](https://github.com/gururise/AlpacaDataCleaned) | 3090 | 10hrs  |
| [flan-gpteacher-lora-xl](https://huggingface.co/reasonwang/flan-gpteacher-lora-xl) | 4.7M           | [GPTeacher](https://github.com/teknium1/GPTeacher)           | 3090 | 80mins |
| [flan-alpaca-gpt4-lora-xl](https://huggingface.co/reasonwang/flan-alpaca-gpt4-lora-xl) | 4.7M           | [alpaca-gpt4](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM) | 3090 | 80mins |

#### Dependcies

```
torch == 1.13.1
transformers == 4.27.3
peft == 0.2.0
```

#### Training

The following command finetune Flan-T5-base with only 20 mins on a single 3090 GPU

```bash
python train.py \
    --model_name_or_path google/flan-t5-base \
    --data_path ./alpaca_data_cleaned.json \
    --bf16 True \
    --output_dir ./ckpts/ \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
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

