from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

#trained_model_path = "./tuned/COROLA-Meta-Llama-3.2-1B/merged/e2"
trained_model_path = "./tuned/COROLA-INSTRUCT-Meta-Llama-3.2-1B/merged/e1"
doChat=True

tokenizer = AutoTokenizer.from_pretrained(trained_model_path, safetensors=True)

if tokenizer.chat_template is None: doChat=False

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=trained_model_path,
    return_dict=True,
    torch_dtype=torch.bfloat16
).to("cuda")


# Function to generate text
def generate_text(model, tokenizer, prompt, max_new_tokens=100, skip_special_tokens=True):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    #print(inputs)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=skip_special_tokens)


prompt = "Once upon a time"
print("Prompt: [{}]\nModel output: [{}]".format(prompt,generate_text(model, tokenizer, prompt)))

prompt = "Patologia tumorală este"
print("Prompt: [{}]\nModel output: [{}]".format(prompt,generate_text(model, tokenizer, prompt)))

if doChat:
    messages=[
        {"role":"system", "content":"Ești un cercetător român."},
        {"role":"user", "content": "Ce este patologia tumorală?"}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print("Prompt: [{}]\nModel output: [{}]".format(prompt,generate_text(model, tokenizer, prompt, skip_special_tokens=False)))
    