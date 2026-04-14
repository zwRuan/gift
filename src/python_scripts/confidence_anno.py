import torch
import json
import os
import argparse
import math
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.multiprocessing as mp
import time

def llama_base_prompt(query: str, tokenizer=None) -> str:
    template = (
        "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|end_of_text|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\n{query}<|end_of_text|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return template.format(query=query)

def llama_ins_prompt(query: str, tokenizer=None) -> str:
    template = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\n{query}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return template.format(query=query)




def deepseek_base_prompt(query: str, tokenizer=None) -> str:
    template = (
        "User: {query}\nPlease reason step by step, "
        "and put your final answer within \\boxed{{}}.\n\nAssistant:"
    )
    return template.format(query=query)


def qwen25_prompt(query: str, tokenizer=None) -> str:
    template = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n{query}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return template.format(query=query)


# === 新增代码开始 ===
PROMPT_MAP = {
    "llama31-8b": llama_ins_prompt,
    "qwen25": qwen25_prompt,
    "deepseek-math":deepseek_base_prompt,
    "llama32-3b":llama_ins_prompt,

}

def compute_confidence_scores(model, tokenizer, prompt, response, device, max_len=None):
    full_text = prompt + response
    
    prompt_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    

    if max_len is not None:

        effective_max_len = max_len - 1
        if inputs["input_ids"].shape[1] > effective_max_len:
            inputs["input_ids"] = inputs["input_ids"][:, :effective_max_len]
            inputs["attention_mask"] = inputs["attention_mask"][:, :effective_max_len]

    input_ids = inputs["input_ids"].to(device)
    prompt_len = prompt_inputs["input_ids"].shape[1]
    
    if input_ids.shape[1] <= prompt_len:
        return []

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits # Shape: [1, seq_len, vocab_size]

    shift_logits = logits[0, :-1, :]  # [seq_len-1, vocab_size]
    shift_labels = input_ids[0, 1:]   # [seq_len-1]

    probs = torch.softmax(shift_logits, dim=-1) # [seq_len-1, vocab_size]

    target_probs = probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1) # [seq_len-1]

    start_idx = max(0, prompt_len - 1)
    
    response_probs = target_probs[start_idx:]
    
    return response_probs.tolist()

def worker_process(rank, gpu_id, model_path, prompt_type, samples, output_queue, max_len=None):
    """
    Worker process to handle a chunk of data on a specific GPU.
    """
    print(f"[Rank {rank}] Waiting {rank * 5}s to initialize...")
    time.sleep(rank * 5)
    try:
        # Set the device for this process
        device = torch.device(f"cuda:{gpu_id}")
        
        # Load model and tokenizer
        print(f"[Rank {rank}] Loading model on {device}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(device)
        model.eval()
        
        prompt_fn = PROMPT_MAP[prompt_type]
        
        results = []
        for idx, sample in enumerate(tqdm(samples, desc=f"Rank {rank}", position=rank)):
            try:
                question = sample['instruction']
                response = sample.get('response', sample.get('prediction', ''))
                
                if not response:
                    results.append((sample['original_idx'], None))
                    continue
                
                templated_prompt = prompt_fn(question, tokenizer)
                probs = compute_confidence_scores(model, tokenizer, templated_prompt, response, device, max_len=max_len)
                
                if not probs: # Handle empty result from truncation
                     results.append((sample['original_idx'], None))
                     continue

                formatted_scores = [[p, p] for p in probs]
                
                output_sample = {
                    "instruction": question,
                    "response": response,
                    "scores": formatted_scores
                }
                results.append((sample['original_idx'], output_sample))
                
            except Exception as e:
                print(f"[Rank {rank}] Error processing sample {sample.get('original_idx')}: {e}")
                results.append((sample['original_idx'], None))
        
        output_queue.put(results)
        
    except Exception as e:
        print(f"[Rank {rank}] Critical error: {e}")
        output_queue.put([])

def process_dataset_confidence_multigpu(model_path, data_file, output_file, prompt_type, max_samples=None, gpu_ids=[0], max_len=None):
    
    if prompt_type not in PROMPT_MAP:
        raise ValueError(f"Invalid prompt_type: {prompt_type}. Choose from {list(PROMPT_MAP.keys())}")

    print(f"Loading data from {data_file}...")
    with open(data_file, 'r') as f:
        samples = [json.loads(line) for line in f if line.strip()]
    
    if max_samples:
        samples = samples[:max_samples]
    
    # Add original index to preserve order
    for i, sample in enumerate(samples):
        sample['original_idx'] = i
        
    print(f"Processing {len(samples)} samples on GPUs {gpu_ids}...")
    
    # Split data into chunks
    num_gpus = len(gpu_ids)
    chunk_size = math.ceil(len(samples) / num_gpus)
    chunks = [samples[i:i + chunk_size] for i in range(0, len(samples), chunk_size)]
    
    # Create output queue
    output_queue = mp.Queue()
    
    # Start processes
    processes = []
    for rank, (gpu_id, chunk) in enumerate(zip(gpu_ids, chunks)):
        p = mp.Process(target=worker_process, args=(rank, gpu_id, model_path, prompt_type, chunk, output_queue, max_len))
        p.start()
        processes.append(p)
    
    # Collect results
    all_results = []
    for _ in range(num_gpus):
        all_results.extend(output_queue.get())
        
    # Wait for all processes to finish
    for p in processes:
        p.join()
        
    # Sort results by original index to maintain order
    all_results.sort(key=lambda x: x[0])
    
    # Write to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f_out:
        count = 0
        for _, result in all_results:
            if result:
                f_out.write(json.dumps(result) + '\n')
                count += 1
                
    print(f"Done! Processed {count}/{len(samples)} samples. Saved to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Teacher Model Path")
    parser.add_argument("--data-file", type=str, required=True, help="Input raw MetaMathQA jsonl")
    parser.add_argument("--output-file", type=str, required=True, help="Output file path")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--prompt-type", type=str, default=None, 
                        choices=["llama31-8b","qwen25", "deepseek-math","llama32-3b"],
                        help="Prompt template type")
    parser.add_argument("--max-len", type=int, default=2048, help="Max sequence length for truncation")
    args = parser.parse_args()

    # Set start method to spawn for CUDA compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    process_dataset_confidence_multigpu(
        args.model_path,
        args.data_file,
        args.output_file,
        args.prompt_type,
        args.max_samples,
        gpu_ids,
        args.max_len
    )

if __name__ == "__main__":
    main()