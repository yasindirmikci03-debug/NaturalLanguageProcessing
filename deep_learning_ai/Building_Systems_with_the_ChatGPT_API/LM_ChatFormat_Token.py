import os
import openai
import tiktoken
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

def get_completion(prompt,model = "gpt-3.5-turbo"):
    messages = [{"role" : "user","content" : prompt}]
    response = openai.ChatCompletion.create(model = model,messages = messages, temperature = 0)

    return response.choices[0].message["content"]

response = get_completion("What is the capital of France?")
print(response)

### TOKENS

response = get_completion("Take the letters in lollipop and reverse them")
print(response)

### HELPER FUNCTION

def get_completion_from_messages(messages, model = "gpt-3.5-turbo",max_tokens = 500):
    response = openai.ChatCompletion.create(model = model, messages = messages, temperature = 0,max_tokens = max_tokens)

    return response.choices[0].message["content"]

messages = [
    {"role" : "system","content" : "You are an assistant who responds in the style of Dr Seuss."},
    {"role" : "user", "content" : "Write me a very short poem about a happy carrot"}
]

response = get_completion_from_messages(messages)
print(response)

# LENGTH
messages = [
    {"role" : "system","content" : "All your responses must be one sentence long."},
    {"role" : "user", "content" : "Write me a very short poem about a happy carrot"}
]

response = get_completion_from_messages(messages)
print(response)

# COMBINED
messages = [
    {"role" : "system", "content" : "All your responses must be one sentence long. You are an assistant who responds in the style of Dr Seuss."},
    {"role":  "user", "content" : "Write me a very short poem about a happy carrot"}
]

response = get_completion_from_messages(messages)
print(response)

def get_completion_and_token_count(messages,model = "gpt-3.5-turbo",max_tokens = 500):
    response = openai.ChatCompletion.create(model = model,messages = messages, max_tokens = max_tokens)

    content =  response.choices[0].message["content"]

    token_dict = {
        "prompt_tokens" : response['usage']['prompt_tokens'],
        "completion_tokens" : response['usage']['completion_tokens'],
        "total_tokens" : response['usage']['total_tokens']
    }

    return content,token_dict

messages = [
    {'role':'system', 'content':"""You are an assistant who responds in the style of Dr Seuss."""},    
    {'role':'user','content':"""write me a very short poem about a happy carrot"""},  
] 

response, token_dict = get_completion_and_token_count(messages)
print(response)
print(token_dict)