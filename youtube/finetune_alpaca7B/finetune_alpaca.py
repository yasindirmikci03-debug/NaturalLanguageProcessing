import torch
import torch.nn as nn
import bitsandbytes as bnb
import transformers

from datasets import load_dataset
from transformers import LlamaTokenizer,AutoTokenizer,AutoConfig,LlamaForCausalLM,LlamaTokenizer
from peft import prepare_model_for_int8_training,LoraConfig,get_peft_model

tokenizer = LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf',add_eos_token = True)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

data = load_dataset('json',data_files = "alpaca_data.json")

def generate_prompt(data_point):
    
    if data_point['instruction']:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""
    

data = data.map(lambda data_point : {"prompt" :tokenizer(generate_prompt(data_point))})


# Setting for A100 
MICRO_BATCH_SIZE = 8    # change to 4 for 3090
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 2      # paper uses 3
LEARNING_RATE = 2e-5
CUTOFF_LEN = 256
LORA_R = 4
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf",load_in_8bit = True,device_map = "auto")

tokenizer = LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf',add_eos_token= True)

model = prepare_model_for_int8_training(model)

config = LoraConfig(
    r=LORA_R,
    lora_alpha = LORA_ALPHA,
    target_modules= ['q_proj',"v_proj"],
    lora_dropout = LORA_DROPOUT,
    bias="none",
    task_type = "CAUSAL_LM"
)

model = get_peft_model(model,config)
tokenizer.pad_token_id = 0
data = load_dataset('json',data_files='alpaca_data.json')

data = data.shuffle().map(
    lambda data_point : tokenizer(
        generate_prompt(data_point),
        truncation = True,
        max_length = CUTOFF_LEN,
        padding = "max_length"
    )
)

trainer = transformers.Trainer(
    model = model,
    train_dataset=data['train'],
    args = transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate = LEARNING_RATE,
        fp16=True,
        logging_steps = 1,
        output_dir = "lora-alpaca",
        save_total_limit = 3,
    ),
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer,mlm=False)
)

model.config.use_cache = False
trainer.train(resume_from_checkpoint = False)

model.save_pretrained("lora-alpaca")

# Generation

from peft import peft_model
from transformers import GenerationConfig

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

model = LlamaForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    load_in_8bit = True,
    device_map = "auto"
)

model = peft_model.from_pretrained(model, "samwit/alpaca7B-lora")

PROMPT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Tell me something about alpacas.
### Response:"""

inputs = tokenizer(PROMPT,return_tensors = "pt")

input_ids = inputs["input_ids"].cuda()

generation_config = GenerationConfig(
    temperature = 0.6,
    top_p = 0.95,
    repetition_penalty = 1.15
)

generation_output = model.generate(
    input_ids = input_ids,
    generation_config = generation_config,
    return_dict_in_generate = True,
    output_scores = True,
    max_new_tokens = 128
)

for s in generation_output.sequences:
    print(tokenizer.decode(s))

