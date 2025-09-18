import os
from utils import get_api_key
import google.generativeai as palm
from google.api_core import client_options as client_options_lib

palm.configure(
    api_key=get_api_key(),
    transport="rest",
    client_options=client_options_lib.ClientOptions(
        api_endpoint=os.getenv("GOOGLE_API_BASE"),
    )
)

models = [m for m in palm.list_models if 'generateText' in m.supported_generation_methods]
model_bison = models[0]

from google.api_core import retry
@retry.Retry()
def generate_text(prompt,model = model_bison,temperature = 0.0):
    return palm.generate_text(prompt = prompt,model=model,temperature=0.0)

prompt_template = """
{priming}

{question}

{decorator}

Your solution:
"""

priming_text = "You are an expert at writing clear, concise, Python code."

question = "create a doubly linked list"

decorator = "Insert comments for each line of code."

prompt = prompt_template.format(priming = priming_text,question=question,decorator=decorator)

print(prompt)

