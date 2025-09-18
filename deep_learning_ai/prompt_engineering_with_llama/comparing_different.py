from utils import llama,llama_chat

# Task 1 : Sentiment Classification

prompt = '''
Message: Hi Amit, thanks for the thoughtful birthday card!
Sentiment: Positive
Message: Hi Dad, you're 20 minutes late to my piano recital!
Sentiment: Negative
Message: Can't wait to order pizza for dinner tonight!
Sentiment: ?

Give a one word response.
'''

response = llama(prompt, model = "META-LLAMA/LLAMA-2-7B-CHAT-HF")
print(response)

response = llama(prompt,model = "META-LLAMA/LLAMA-2-70B-CHAT-HF")
print(response)

response = llama(prompt,model = "META-LLAMA/Llama-3-8B-CHAT-HF")
print(response)

response = llama(prompt,model = "META-LLAMA/Llama-3-70B-CHAT-HF")
print(response)

# Task 2 : Summarization

email = """
Dear Amit,

An increasing variety of large language models (LLMs) are open source, or close to it. The proliferation of models with relatively permissive licenses gives developers more options for building applications.

Here are some different ways to build applications based on LLMs, in increasing order of cost/complexity:

Prompting. Giving a pretrained LLM instructions lets you build a prototype in minutes or hours without a training set. Earlier this year, I saw a lot of people start experimenting with prompting, and that momentum continues unabated. Several of our short courses teach best practices for this approach.
One-shot or few-shot prompting. In addition to a prompt, giving the LLM a handful of examples of how to carry out a task — the input and the desired output — sometimes yields better results.
Fine-tuning. An LLM that has been pretrained on a lot of text can be fine-tuned to your task by training it further on a small dataset of your own. The tools for fine-tuning are maturing, making it accessible to more developers.
Pretraining. Pretraining your own LLM from scratch takes a lot of resources, so very few teams do it. In addition to general-purpose models pretrained on diverse topics, this approach has led to specialized models like BloombergGPT, which knows about finance, and Med-PaLM 2, which is focused on medicine.
For most teams, I recommend starting with prompting, since that allows you to get an application working quickly. If you’re unsatisfied with the quality of the output, ease into the more complex techniques gradually. Start one-shot or few-shot prompting with a handful of examples. If that doesn’t work well enough, perhaps use RAG (retrieval augmented generation) to further improve prompts with key information the LLM needs to generate high-quality outputs. If that still doesn’t deliver the performance you want, then try fine-tuning — but this represents a significantly greater level of complexity and may require hundreds or thousands more examples. To gain an in-depth understanding of these options, I highly recommend the course Generative AI with Large Language Models, created by AWS and DeepLearning.AI.

(Fun fact: A member of the DeepLearning.AI team has been trying to fine-tune Llama-2-7B to sound like me. I wonder if my job is at risk? 😜)

Additional complexity arises if you want to move to fine-tuning after prompting a proprietary model, such as GPT-4, that’s not available for fine-tuning. Is fine-tuning a much smaller model likely to yield superior results than prompting a larger, more capable model? The answer often depends on your application. If your goal is to change the style of an LLM’s output, then fine-tuning a smaller model can work well. However, if your application has been prompting GPT-4 to perform complex reasoning — in which GPT-4 surpasses current open models — it can be difficult to fine-tune a smaller model to deliver superior results.

Beyond choosing a development approach, it’s also necessary to choose a specific model. Smaller models require less processing power and work well for many applications, but larger models tend to have more knowledge about the world and better reasoning ability. I’ll talk about how to make this choice in a future letter.

Keep learning!

Andrew
"""

prompt = f"""
Summarize this email and extract some key points.

What did the author say about llama models?
```
{email}
```
"""

response_7b = llama(prompt,model="META-LLAMA/Llama-2-7B-CHAT-HF")
print(response_7b)

response_13b = llama(prompt,model="META-LLAMA/Llama-2-13B-CHAT-HF")
print(response_13b)


# Task 3 : Reasoning 

context = """
Jeff and Tommy are neighbors

Tommy and Eddy are not neighbors
"""

query = """
Are Jeff and Eddy neighbors ?
"""

prompt = f"""
Given this context : ```{context}```,

and the following query:
```{query}```

Please answer the questions in the query and explain your reasoning.
If there is not enough information to answer, please say
"I do not have enough information to answer this questions."
"""

response_7b_chat = llama(prompt,model="META-LLAMA/Llama-2-7B-CHAT-HF")
print(response_7b_chat)

response_13b_chat = llama(prompt,model="META-LLAMA/Llama-2-13B-CHAT-HF")
print(response_13b_chat)



