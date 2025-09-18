
from utils import llama

# define the prompt
prompt = "Help me write a birthday card for my dear friend Andrew."

# pass prompt to the llama function, store output as 'response' then print
response = llama(prompt)
print(response)

# Set verbose to true to see the full prompt that is passed to the model.
prompt = "Help me write a birthday card for my dear friend Andrew."
response = llama(prompt,verbose = True)


# Chat vs base models

### chat model
prompt = "What is the capital of France?"
response = llama(prompt,verbose = True,model = "togethercomputer/llama-2-7b-chat")

print(response)

### base model
prompt = "What is the capital of France ?"
response = llama(prompt,verbose = True,model ="togethercomputer/llama-2-7b")

## Using Llama 3 chat models

response = llama(response,verbose = True,model = "META-LLAMA/LLAMA-3-8B-CHAT-HF",add_inst = False)

print(response)

#### Changing the temperature setting
prompt = """
Help me write a birthday card for my dear friend Andrew.
Here are details about my friend:
He likes long walks on the beach and reading in the bookstore.
His hobbies include reading research papers and speaking at conferences.
His favorite color is light blue.
He likes pandas.
"""
response = llama(prompt, temperature=0.0)
print(response)

prompt = """
Help me write a birthday card for my dear friend Andrew.
Here are details about my friend:
He likes long walks on the beach and reading in the bookstore.
His hobbies include reading research papers and speaking at conferences.
His favorite color is light blue.
He likes pandas.
"""
response = llama(prompt, temperature=0.9)
print(response)

# Changing the max tokens setting 
prompt = """
Help me write a birthday card for my dear friend Andrew.
Here are details about my friend:
He likes long walks on the beach and reading in the bookstore.
His hobbies include reading research papers and speaking at conferences.
His favorite color is light blue.
He likes pandas.
"""
response = llama(prompt, max_tokens = 20)
print(response)

# Asking a follow up question

prompt = """
Help me write a birthday card for my dear friend Andrew.
Here are details about my friend:
He likes long walks on the beach and reading in the bookstore.
His hobbies include reading research papers and speaking at conferences.
His favorite color is light blue.
He likes pandas.
"""
response = llama(prompt)
print(response)

prompt_2 = """
Oh, he also likes teaching. Can you rewrite it to include that?
"""
response_2 = llama(prompt_2)
print(response_2)
