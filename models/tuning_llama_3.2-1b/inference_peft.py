# from https://docs.pytorch.org/torchtune/stable/tutorials/e2e_flow.html

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

#TODO: update it to your chosen epoch
#trained_model_path = "./tuned/Meta-Llama-3.2-1B-Instruct/lora_single_device/epoch_3"
#trained_model_path = "./tuned/Meta-Llama-3.2-1B/lora_single_device/epoch_3"
#trained_model_path = "./tuned/Meta-Llama-3.2-1B-Instruct/lora_single_device/epoch_3"
#trained_model_path = "./models/Meta-Llama-3.2-1B-Instruct/"
#trained_model_path = "./tuned/COROLA-Meta-Llama-3.2-1B/qlora_single_device/epoch_2"
trained_model_path = "./tuned/COROLA-INSTRUCT-Meta-Llama-3.2-1B/qlora_single_device/epoch_1"

#original_model_name = "./models/Meta-Llama-3.2-1B-Instruct"
#original_model_name = "./models/Meta-Llama-3.2-1B"
original_model_name = "./tuned/COROLA-INSTRUCT-Meta-Llama-3.2-1B/merged/e1"

model = AutoModelForCausalLM.from_pretrained(original_model_name, torch_dtype=torch.bfloat16, load_in_4bit=True).to("cuda")
#peft_model = PeftModel.from_pretrained(model, trained_model_path, torch_dtype=torch.bfloat16, load_in_4bit=True).to("cuda")
peft_model=model

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(original_model_name)


# Function to generate text
def generate_text(model, tokenizer, prompt, max_length=2048):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    #print(inputs)
    outputs = model.generate(
         **inputs, 
         #max_length=max_length,
         temperature=0.2,
         top_p=0.9,
         repetition_penalty=1.5,
         early_stopping=True,
         num_beams=4,
         eos_token_id=tokenizer.eos_token_id,
         max_new_tokens=50
    )
    #print(outputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


prompt = "Complete the sentence: 'Once upon a time...'"
print("Base model output:", generate_text(peft_model, tokenizer, prompt))

prompt = "patologia tumorală este "
print("Base model output:", generate_text(peft_model, tokenizer, prompt))

prompt = "ce este patologia tumorală ?"
print("Base model output:", generate_text(peft_model, tokenizer, prompt))


#prompt = "Ce este patologia tumorală ?"
#print("Base model output:", generate_text(peft_model, tokenizer, prompt))

#prompt = "patologia tumorală "
#print("Base model output:", generate_text(peft_model, tokenizer, prompt))

#prompt = """<|start_header_id|>system<|end_header_id|><|eot_id|><|start_header_id|>user<|end_header_id|>Ce este tumora?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
prompt = """<|start_header_id|>user<|end_header_id|>ce este patologia tumorală?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
print("Base model output:", generate_text(peft_model, tokenizer, prompt))
