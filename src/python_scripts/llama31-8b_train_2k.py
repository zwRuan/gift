import os
import copy
import json
import fire
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from peft import LoraConfig, get_peft_model, TaskType
os.environ["WANDB_MODE"] = "offline"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

IGNORE_INDEX = -100

def load_jsonl(file_path: str):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

PROMPT_DICT = {
    "prompt_no_input": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\n{instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
}

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="/volume/pt-train/users/wzhang/ghchen/zh/models/Llama-2-7b")

@dataclass
class DataArguments:
    data_path: str = field(default="alpaca_data.json")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512)
    output_dir: str = field(default="./output")
    per_device_train_batch_size: int = field(default=4)
    num_train_epochs: float = field(default=3.0)
    learning_rate: float = field(default=2e-5)

class EnhancedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Training mode: sft")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            valid_mask = shift_labels != IGNORE_INDEX
            
            if valid_mask.sum() == 0:
                loss = torch.tensor(0.0, device=shift_logits.device, requires_grad=True)
            else:
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                token_losses = loss_fct(shift_logits, shift_labels)
                
                weighted_losses = token_losses
                loss = weighted_losses[valid_mask].mean()
        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    tokenized_list = [
        tokenizer(text, return_tensors="pt", padding="longest", max_length=tokenizer.model_max_length, truncation=True, add_special_tokens=False)
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    return dict(input_ids=input_ids, labels=labels, input_ids_lens=input_ids_lens, labels_lens=labels_lens)

def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

class SupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        list_data_dict = load_jsonl(data_path)
        prompt_no_input = PROMPT_DICT["prompt_no_input"]
        sources = [prompt_no_input.format_map(example) for example in list_data_dict]
        targets = [f"{example['response']}{tokenizer.eos_token}" for example in list_data_dict]
        data_dict = preprocess(sources, targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.sources = sources
        self.targets = targets

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def show_first_example(data_path: str, tokenizer: transformers.PreTrainedTokenizer):
    """Show first training example with tokenization details"""
    print("\n" + "="*50)
    print("FIRST TRAINING EXAMPLE")
    print("="*50)
    
    # Load first example
    data = load_jsonl(data_path)
    if not data:
        print("No data found")
        return
    
    example = data[0]
    instruction = example.get('instruction', '')
    response = example.get('response', '')
    
    print(f"Instruction: {instruction}")
    print(f"Response: {response}")
    
    # Format with prompt
    prompt = PROMPT_DICT["prompt_no_input"].format_map(example)
    full_text = prompt + response + tokenizer.eos_token
    
    print(f"\nFull prompt:\n{prompt}")
    print(f"Target text: {response}{tokenizer.eos_token}")
    
    # Tokenize
    tokenized = tokenizer(full_text, return_tensors="pt", max_length=tokenizer.model_max_length, truncation=True, add_special_tokens=False)
    input_ids = tokenized.input_ids[0]
    
    # Calculate instruction length for masking
    instruction_tokenized = tokenizer(prompt, return_tensors="pt", max_length=tokenizer.model_max_length, truncation=True, add_special_tokens=False)
    instruction_len = instruction_tokenized.input_ids[0].shape[0]
    
    print(f"\nTokenization:")
    print(f"Total tokens: {len(input_ids)}")
    print(f"Instruction tokens: {instruction_len}")
    print(f"Response tokens: {len(input_ids) - instruction_len}")
    
    # Show loss computation part
    labels = input_ids.clone()
    labels[:instruction_len] = IGNORE_INDEX
    
    print(f"\nLoss computation tokens (response part):")
    loss_tokens = input_ids[instruction_len:]
    decoded_loss_part = tokenizer.decode(loss_tokens, skip_special_tokens=False)
    print(f"Tokens for loss: {decoded_loss_part}")
    print(f"Token IDs: {loss_tokens.tolist()}")
    print("="*50 + "\n")


def train(
    model_name_or_path: str = "models/Llama-2-7b",
    data_path: str = "data/train_medmcqa_alpaca_10k.jsonl",
    cache_dir: str = None,
    model_max_length: int = 512,
    per_device_train_batch_size: int = 4,
    num_train_epochs: float = 3.0,
    learning_rate: float = 2e-5,
    global_batch_size: int = 64,
    output_dir: str = None,
    **kwargs
):
    """Standard SFT training entry."""
    
    model_args = ModelArguments(model_name_or_path=model_name_or_path)
    data_args = DataArguments(data_path=data_path)
    
    if output_dir is None:
        output_dir: str = f"./output/sft/{os.path.basename(model_name_or_path)}"
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    gradient_accumulation_steps = global_batch_size // (per_device_train_batch_size * world_size)
    print("########### global_batch_size ###########",global_batch_size)
    print("########### per_device_train_batch_size ###########",per_device_train_batch_size)
    print("########### world_size ###########",world_size)

    training_args = TrainingArguments(
        output_dir=output_dir,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        logging_steps=1,
        save_steps=2,
        save_strategy="steps",
        save_only_model=True,
        save_total_limit=50,
        dataloader_num_workers=0,
        warmup_ratio=0.1,
        dataloader_pin_memory=False,
        # logging_steps=1,
        **kwargs
    )

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.eos_token="<|eot_id|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Applying LoRA for Llama-like architecture (Llama 3.1 / Qwen 2.5 / OLMo 2)...")
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=64, 
        lora_alpha=128, 
        lora_dropout=0.05,
        bias="none",
        # 这三个模型的标准全量微调模块
        target_modules=[
            "q_proj", 
            "k_proj", 
            "v_proj", 
            "o_proj", 
            "gate_proj", 
            "up_proj", 
            "down_proj"
        ]
    )
    
    model = get_peft_model(model, peft_config)

    # Show first example
    show_first_example(data_args.data_path, tokenizer)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args) 
    trainer = EnhancedTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )
    
    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    fire.Fire(train)
    