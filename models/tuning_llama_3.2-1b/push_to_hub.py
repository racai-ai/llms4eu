from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

#trained_model_path = "./tuned/COROLA-Meta-Llama-3.2-1B/merged/e2"
trained_model_path = "./tuned/COROLA-INSTRUCT-Meta-Llama-3.2-1B/merged/e1"

hub_path="racai/corola-instruct-llama-3.2-1b-e2"

tokenizer = AutoTokenizer.from_pretrained(trained_model_path, safetensors=True)
tokenizer.push_to_hub(hub_path)

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=trained_model_path,
    return_dict=True,
    torch_dtype=torch.bfloat16
)
model.push_to_hub(hub_path)

