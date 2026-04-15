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
        "User: {instruction}\nPlease reason step by step, "
        "and put your final answer within \\boxed{{}}.\n\nAssistant:"
    ),
}

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="")

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
    def __init__(self, mode="sft", focal_gamma=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if mode not in {"sft", "gift"}:
            raise ValueError(f"Unsupported mode: {mode}. Supported modes are: sft, gift.")
        self.mode = mode
        self.focal_gamma = focal_gamma
        print(f"Training mode: {mode}")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        loss_weights = inputs.get("loss_weights")
        model_inputs = {k: v for k, v in inputs.items() if k != "loss_weights"}
        outputs = model(**model_inputs)
        logits = outputs.get("logits")
        
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            valid_mask = shift_labels != IGNORE_INDEX

            if self.mode == "gift" and loss_weights is None:
                raise ValueError("loss_weights are required for gift mode. Ensure dataset contains valid scores.")
            shift_weights = None
            if loss_weights is not None:
                # ✅ 修复：与 logits[:-1] 对齐，使用 [:-1] 而不是 [1:]
                shift_weights = loss_weights[..., :-1].contiguous().view(-1)
            
            if valid_mask.sum() == 0:
                loss = torch.tensor(0.0, device=shift_logits.device, requires_grad=True)
            else:
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                token_losses = loss_fct(shift_logits, shift_labels)
                
                if self.mode == "sft":
                    weighted_losses = token_losses
                    
                elif self.mode == "gift":
                    probs = torch.softmax(shift_logits, dim=-1)
                    valid_labels = torch.clamp(shift_labels, min=0, max=probs.size(-1)-1)
                    p_cur = probs.gather(1, valid_labels.unsqueeze(-1)).squeeze(-1).detach()
                    base_probs = shift_weights
                    if base_probs is None:
                        raise ValueError("Base probabilities missing for gift mode.")
                    if base_probs.shape[0] != shift_labels.shape[0]:
                        raise ValueError(f"Shape mismatch for base probabilities: got {base_probs.shape[0]} vs labels {shift_labels.shape[0]}.")
                    
                    # 计算权重：p_current * (1 - p_base)^gamma
                    novelty_factor = torch.pow(base_probs.clamp(min=1e-8, max=1.0), self.focal_gamma)
                    weight = novelty_factor
                    # 只在有效位置应用权重
                    weight = torch.where(valid_mask, weight, torch.zeros_like(weight))
                    weighted_losses = token_losses * weight

                else:
                    raise ValueError(f"Unsupported mode: {self.mode}")
                
                # 🔧 核心修复：使用 masked_mean 方式（与 OpenRLHF 一致）
                # 对于 gift，除以 weight.sum() 而不是 valid_mask.sum()
                if self.mode == "gift" and valid_mask.sum() > 0:
                    # 使用加权平均：sum(loss * weight) / sum(weight)
                    # 这样权重会被归一化，避免 loss scale 问题
                    weight_sum = weight[valid_mask].sum()
                    if weight_sum > 1e-8:
                        loss = weighted_losses[valid_mask].sum() / weight_sum
                    else:
                        loss = weighted_losses[valid_mask].mean()
                else:
                    # 其他模式使用简单平均
                    loss = weighted_losses[valid_mask].mean() if valid_mask.sum() > 0 else torch.tensor(0.0, device=shift_logits.device, requires_grad=True)
        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=False,
        )
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
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, mode: str = "sft"):
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
        self.mode = mode
        self.loss_weights = []

        for idx, example in enumerate(list_data_dict):
            labels = self.labels[idx]
            source_len = tokenizer(
                self.sources[idx],
                return_tensors="pt",
                truncation=True,
                max_length=tokenizer.model_max_length,
                add_special_tokens=False,
            ).input_ids.shape[1]

            weights_tensor = torch.zeros_like(labels, dtype=torch.float)
            weights_tensor[labels != IGNORE_INDEX] = 1.0

            if self.mode == "gift":
                if "scores" not in example:
                    raise ValueError(f"Example index {idx} missing 'scores' required for gift mode.")
                raw_scores = example["scores"]
                base_probs = []
                for s in raw_scores:
                    if isinstance(s, (list, tuple)):
                        base_probs.append(float(s[0]))
                    else:
                        base_probs.append(float(s))

                seq_len = self.input_ids[idx].shape[0]
                resp_len = seq_len - source_len

                # ✅ 修复：scores 的数量应该比 response tokens 少 1（因为最后一个 token 不需要预测下一个）
                # 对于长度为 N 的 response，有 N-1 个 prediction step
                if len(base_probs) not in (resp_len - 1, resp_len):
                    raise ValueError(
                        f"Score/token length mismatch at example {idx}: scores={len(base_probs)}, response_tokens={resp_len}, expected={resp_len-1}."
                    )
                
                # ✅ 修复：与 OpenRLHF 对齐，从 source_len - 1 位置开始填充
                # logits[source_len - 1] 预测 labels[source_len]（response 第 1 个 token）
                # 所以 loss_weights[source_len - 1] 应该存储 response 第 1 个 token 的 base_prob
                for j, prob in enumerate(base_probs):
                    pos = source_len - 1 + j  # ✅ 从 source_len - 1 开始，而不是 source_len
                    if pos >= weights_tensor.shape[0]:
                        break
                    weights_tensor[pos] = max(prob, 1e-4)

            self.loss_weights.append(weights_tensor)
        
        # 🔍 Debug: 打印第一个样本的对齐信息
        if len(self.loss_weights) == 1 and self.mode == "gift":
            print("\n" + "="*80)
            print("🔍 [GIFT Debug] 第一个样本的权重对齐检查")
            print("="*80)
            example = list_data_dict[0]
            print(f"Instruction: {example.get('instruction', '')[:100]}...")
            print(f"Response length: {len(example.get('response', ''))}")
            
            # 找到第一个非 IGNORE_INDEX 的位置
            non_ignore_indices = (self.labels[0] != IGNORE_INDEX).nonzero(as_tuple=True)[0]
            if len(non_ignore_indices) > 0:
                first_response_idx = non_ignore_indices[0].item()
                print(f"\\nFirst response token position (in labels): {first_response_idx}")
                print(f"Source length (calculated): {source_len}")
                print(f"Loss weight填充起始位置: {source_len - 1}")
                
                # 显示前5个 response token 的权重
                print(f"\\n前5个 response prediction 的 loss_weights:")
                for i in range(min(5, len(base_probs))):
                    weight_pos = source_len - 1 + i
                    label_pos = source_len + i
                    print(f"  - loss_weights[{weight_pos}] = {weights_tensor[weight_pos]:.6f} (预测 labels[{label_pos}])")
                
                print(f"\\nBase probs 数量: {len(base_probs)}")
                print(f"Response tokens 数量: {resp_len}")
                print(f"应该有 {resp_len - 1} 个预测步骤")
            print("="*80 + "\\n")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], loss_weights=self.loss_weights[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        loss_weights = [instance.get("loss_weights", torch.tensor([])) for instance in instances]
        loss_weights = torch.nn.utils.rnn.pad_sequence(loss_weights, batch_first=True, padding_value=0.0)
        return dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id), loss_weights=loss_weights)

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, mode: str) -> Dict:
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, mode=mode)
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
    tokenized = tokenizer(
        full_text,
        return_tensors="pt",
        max_length=tokenizer.model_max_length,
        truncation=True,
        add_special_tokens=False,
    )
    input_ids = tokenized.input_ids[0]
    
    # Calculate instruction length for masking
    instruction_tokenized = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=tokenizer.model_max_length,
        truncation=True,
        add_special_tokens=False,
    )
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
    mode: str = "sft",
    focal_gamma: float = 1.0,
    output_dir: str = None,
    **kwargs
):
    """Training entry for SFT and GIFT modes."""
    
    model_args = ModelArguments(model_name_or_path=model_name_or_path)
    data_args = DataArguments(data_path=data_path)
    
    if output_dir is None:
        output_dir: str = f"./output/{mode}/{os.path.basename(model_name_or_path)}"
    if mode not in {"sft", "gift"}:
        raise ValueError(f"Unsupported mode: {mode}. Supported modes are: sft, gift.")
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    gradient_accumulation_steps = global_batch_size // (per_device_train_batch_size * world_size)
    print("########### global_batch_size ###########",global_batch_size)
    print("########### per_device_train_batch_size ###########",per_device_train_batch_size)
    print("########### world_size ###########",world_size)
    print("########### gradient_accumulation_steps ##########",gradient_accumulation_steps)

    training_args = TrainingArguments(
        output_dir=output_dir,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        logging_steps=1,
        save_steps=7,
        save_strategy="steps",
        save_only_model=True,
        save_total_limit=50,
        dataloader_num_workers=0,
        warmup_ratio=0.1,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
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

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, mode=mode) 
    trainer = EnhancedTrainer(
        mode=mode,
        focal_gamma=focal_gamma,
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )
    
    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    fire.Fire(train)
    