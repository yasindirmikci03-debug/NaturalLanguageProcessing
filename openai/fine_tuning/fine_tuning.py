###### Fine-tuning lets you get more out of the models available through the API by providing:

# Higher quality results than prompting
# Ability to train on more examples than can fit in a prompt
# Token savings due to shorter prompts
# Lower latency requests

# At a high level, fine-tuning involves the following steps:

# Prepare and upload training data
# Train a new fine-tuned model
# Evaluate results and go back to step 1 if needed
# Use your fine-tuned mode


###### When to use fine tuning ? 

# There are many tasks at which our models may not initally appear to perform well, but results can be improved
# with the right prompts -thus fine-tuning may not be necessary

# Iterating over prompts and other tactics has a much faster feedback loop than iterating with fine-tuning,
# which requires creating datasets and running training jobs

# In cases where fine-tuning is still necessary, initial prompt engineering work is not wasted - we typically see best results 
# when using a good prompt in the fine-tuning data (or combining prompt chaining/ tool use with fine-tuning)

###### Common use cases

# Setting the style, tone, format, or other qualitative aspects
# Improving reliability at producing a desired output
# Correcting failures to follow complex prompts
# Handling many edge cases in specific ways
# Performing a new skill or task that’s hard to articulate in a prompt



#### Preparing your dataset

# Example format

{"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."}]}
{"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "Who wrote 'Romeo and Juliet'?"}, {"role": "assistant", "content": "Oh, just some guy named William Shakespeare. Ever heard of him?"}]}
{"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "How far is the Moon from Earth?"}, {"role": "assistant", "content": "Around 384,400 kilometers. Give or take a few, like that really matters."}]}

{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}

# Multi-turn chat examples

{"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "Paris", "weight": 0}, {"role": "user", "content": "Can you be more sarcastic?"}, {"role": "assistant", "content": "Paris, as if everyone doesn't know that already.", "weight": 1}]}
{"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "Who wrote 'Romeo and Juliet'?"}, {"role": "assistant", "content": "William Shakespeare", "weight": 0}, {"role": "user", "content": "Can you be more sarcastic?"}, {"role": "assistant", "content": "Oh, just some guy named William Shakespeare. Ever heard of him?", "weight": 1}]}
{"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "How far is the Moon from Earth?"}, {"role": "assistant", "content": "384,400 kilometers", "weight": 0}, {"role": "user", "content": "Can you be more sarcastic?"}, {"role": "assistant", "content": "Around 384,400 kilometers. Give or take a few, like that really matters.", "weight": 1}]}

# Upload a training file

from openai import OpenAI
 
client = OpenAI()

client.files.create(
    file = open("mydata.jsonl","rb"),
    purpose = "fine-tune"
)

# Create a fine-tuned model

client.fine_tuning.jobs.create(
    training_file="file-abc123",
    model = "gpt-3.5-turbo"
)

# Use a fine-tuned model

completion = client.chat.completions.create(
    model = "ft:gpt-3.5-turbo:my-org:custom_suffix:id",
    messages = [
        {"role" : "system","content" : "You are a helpful assistant."},
        {"role" : "user","content" : "Hello!"}
    ]
)