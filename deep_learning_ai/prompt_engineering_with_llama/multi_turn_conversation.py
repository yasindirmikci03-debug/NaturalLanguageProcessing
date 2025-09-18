from utils import llama

prompt = """
    What are fun activities I can do this weekend?
"""

response = llama(prompt)
print(response)

prompt_2 = """Which of the these would be good for my health?"""

response2 = llama(prompt_2)
print(response2)

# Constructing multi-turn prompts

prompt_1 = """
    What are fun activities I can do this weekend?
"""
response_1 = llama(prompt_1)

prompt_2 = """
Which of these would be good for my health?
"""

chat_prompt = f"""
<s>[INST] {prompt_1} [/INST]
{response_1}
</s>
<s>[INST] {prompt_2} [/INST]
"""
print(chat_prompt)

response_2 = llama(chat_prompt,
                 add_inst=False,
                 verbose=True)

# Use llama chat helper function 

from utils import llama_chat

prompt_1 = """
    What are fun activities I can do this weekend?
"""
response_1 = llama(prompt_1)

prompt_2 = """
Which of these would be good for my health?
"""

prompts = [prompt_1,prompt_2]
responses = [response_1]

# Pass prompts and responses to llama_chat function
response_2 = llama_chat(prompts,responses,verbose=True)

