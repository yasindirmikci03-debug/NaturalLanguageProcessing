import torch
import json
import textwrap

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",device_map = "auto",torch_dtype = torch.float16)

pipeline = pipeline("text-generation",
                    model = model,
                    tokenizer = tokenizer,
                    torch_dtype = torch.bfloat16,
                    device_map = "auto",
                    max_new_tokens = 512,
                    do_sample = True,
                    top_k = 30,
                    num_return_sequences = 1,
                    eos_token_id = tokenizer.eos_token_id)


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT ):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

def cut_off_text(text, prompt):
    cutoff_phrase = prompt
    index = text.find(cutoff_phrase)
    if index != -1:
        return text[:index]
    else:
        return text

def remove_substring(string, substring):
    return string.replace(substring, "")



def generate(text):
    prompt = get_prompt(text)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
        outputs = model.generate(**inputs,
                                 max_new_tokens=512,
                                 eos_token_id=tokenizer.eos_token_id,
                                 pad_token_id=tokenizer.eos_token_id,
                                 )
        final_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        final_outputs = cut_off_text(final_outputs, '</s>')
        final_outputs = remove_substring(final_outputs, prompt)

    return final_outputs#, outputs

def parse_text(text):
        wrapped_text = textwrap.fill(text, width=100)
        print(wrapped_text +'\n\n')


llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {"temperature" : 0})

system_prompt = "You are an advanced assistant that excels at translation. "
instruction = "Convert the following text from English to French:\n\n {text}"

template = get_prompt(instruction, system_prompt)

prompt = PromptTemplate(template = template, input_variables = ["text"])
llm_chain = LLMChain(prompt = prompt , llm = llm)

### Summarization
instruction = "Summarize the following article for me {text}"
system_prompt = "You are an expert and summarization and expressing key ideas succintly"

template = get_prompt(instruction, system_prompt)

prompt = PromptTemplate(template = template, input_variables = ["text"])
llm_chain = LLMChain(prompt = prompt , llm = llm)

### Simple Chatbot
from langchain.memory import ConversationBufferMemory

instruction = "Chat History:\n\n{chat_history} \n\nUser: {user_input}"
system_prompt = "You are a helpful assistant, you always only answer for the assistant then you stop. read the chat history to get context"

template = get_prompt(instruction, system_prompt)

prompt = PromptTemplate(input_variables=["chat_history", "user_input"], template=template)
memory = ConversationBufferMemory(memory_key="chat_history")


llm_chain = LLMChain(llm = llm, prompt = prompt, verbose = True, memory = memory)
print(llm_chain.predict(user_input="Hi, my name is Sam"))

print(llm_chain.predict(user_input="Can you tell me about yourself."))
