import torch
import transformers
import json
import textwrap

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

from huggingface_hub import notebook_login

notebook_login()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")#use_auth_token = True)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",device_map = "auto",torch_dtype = torch.float16)#use_auth_token = True)

pipeline = pipeline("text-generation",
                    model = model,
                    tokenizer = tokenizer,
                    torch_dtype = torch.bfloat16,
                    device_map = "auto",
                    do_sample = True,
                    top_k = 30,
                    num_return_sequences = 1,
                    eos_token_id = tokenizer.eos_token_id)


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS

def get_prompt(instruction):
    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

def cut_off_text(text,prompt):
    cutoff_phrase = prompt
    index = text.find(cutoff_phrase)
    if index != -1:
        return text[:index]
    else:
        return text
    
def remove_substring(string,substring):
    return string.replace(substring, "")

def generate(text):
    prompt = get_prompt(text)
    with torch.autocast('cuda', dtype = torch.bfloat16):
        inputs = tokenizer(prompt,return_tensors="pt").to('cuda')
        outputs = model.generate(**inputs,max_new_tokens = 512,eos_token_id = tokenizer.eos_token_id,pad_token_id = tokenizer.eos_token_id)
        final_outputs = tokenizer.batch_decode(outputs,skip_special_tokens=True)[0]
        final_outputs = cut_off_text(final_outputs, "</s>")
        final_outputs = remove_substring(final_outputs, prompt)

    return final_outputs

def parse_text(text):
    wrapped_text = textwrap.fill(text,width = 100)
    print(wrapped_text + '\n\n')


prompt = 'What are the differences between alpacas, vicunas and llamas?'
generated_text = generate(prompt)
parse_text(generated_text)

