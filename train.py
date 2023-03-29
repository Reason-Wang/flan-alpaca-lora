from dataclasses import field, dataclass
from typing import Optional, Any

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

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="google/flan-t5-base")


@dataclass
class DataArguments:
    data_path: str = field(default="./alpaca_data.json", metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
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
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False
    )

    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir)

    config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=training_args.lora_target_modules,
        lora_dropout=training_args.lora_dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    dataset = Seq2SeqDataset(data_path=data_args.data_path)
    collator = Seq2SeqCollator(tokenizer=tokenizer)

    trainer = Trainer(
        model,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset,
    )

    trainer.train()

    model.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    train()
