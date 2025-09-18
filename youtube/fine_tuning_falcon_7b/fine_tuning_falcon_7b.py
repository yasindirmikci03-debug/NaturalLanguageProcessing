import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,TrainingArguments
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer

dataset_name = load_dataset('timdettmers/openassistant-guanaco')

dataset = load_dataset(dataset_name,split = "train")

model_name = "ybelkada/falcon-7b-sharded-bf16"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config = bnb_config,
    trust_remote_code = True
)

model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)

tokenizer.pad_token = tokenizer.eos_token

lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha = lora_alpha,
    lora_dropout=lora_dropout,
    lora_r = lora_r,
    bias = "none",
    task_type = "CASUAL_LM",
    target_modules = ["query_key_value"] #,"dense","dense_h_to_4h","dense_4h_to_h"]
)

output_dir = "./results"
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 10
logging_steps = 10
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 500
warmup_ratio = 0.02
lr_scheduler_type = "constant"

training_arguments = TrainingArguments(
    output_dir = output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim = optim,
    save_steps = save_steps,
    logging_steps = logging_steps,
    learning_rate = learning_rate,
    max_grad_norm = max_grad_norm,
    max_steps = max_steps,
    warmup_ratio = warmup_ratio,
    lr_scheduler_type = lr_scheduler_type,
    fp16 = True,
    group_by_length = True,
)

max_seq_length = 512

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    peft_config = peft_config,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    tokenizer = tokenizer,
    args = training_arguments
)

trainer.train()

