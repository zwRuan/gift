from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_model", required=True)
parser.add_argument("--adapter", required=True)
parser.add_argument("--output_path", required=True)
parser.add_argument("--chat_template", help="path to llama3_template.txt",default=None)
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, args.adapter)
model = model.merge_and_unload()
model.save_pretrained(args.output_path)

tokenizer = AutoTokenizer.from_pretrained(args.base_model)
if args.chat_template:
    tokenizer.chat_template = Path(args.chat_template).read_text()
tokenizer.save_pretrained(args.output_path)