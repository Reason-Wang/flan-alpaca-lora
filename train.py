import os
from dataclasses import field, dataclass
from typing import Optional, Any

import torch
import transformers
from transformers import Trainer

from dataset import Seq2SeqDataset, Seq2SeqCollator

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from typing import List
import logging
logging.basicConfig(level=logging.INFO)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default="google/flan-t5-base")
    data_paths: List[str] = field(default_factory=lambda: ["./alpaca_data.json"], metadata={"help": "Path to the training data."})
    instruction_length: int = 40
    output_length: int = 160
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    load_in_8bit: bool = field(default=True)
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    # lora arguments
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q", "v",])


def train():
    parser = transformers.HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False
    )

    device_map = "auto"

    if args.model_name_or_path == "google/flan-t5-xxl" and args.load_in_8bit == False:
        logging.info("You are training flan-t5-xxl with float32 data type. "
                     "To save the memory, you may set load_in_8bit to True.")


    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        load_in_8bit=args.load_in_8bit,
        use_cache=False,
        torch_dtype=torch.float16,
        cache_dir=args.cache_dir,
        device_map=device_map,
    )

    if args.load_in_8bit:
        model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    dataset = Seq2SeqDataset(args.data_paths)
    collator = Seq2SeqCollator(tokenizer, args.instruction_length, args.output_length)

    trainer = Trainer(
        model,
        args=args,
        data_collator=collator,
        train_dataset=dataset,
    )

    trainer.train()

    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    train()
