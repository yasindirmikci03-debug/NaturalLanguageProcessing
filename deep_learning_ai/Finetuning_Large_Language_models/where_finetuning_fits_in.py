import jsonlines
import itertools
import pandas as pd
from pprint import pprint

import datasets
from datasets import load_dataset

# Pretraining data

pretrained_data = load_dataset("c4","en",split="train",streaming=True)

n=5
top_n = itertools.islice(pretrained_data,n)
for i in top_n:
    print(i)

# Contrast with company finetuning dataset you will be using
filename = "lamini_docs.jsonl"
instruction_dataset_df = pd.read_json(filename,lines=True)

# Various ways of formatting your data
examples = instruction_dataset_df.to_dict()

text = examples["question"][0] + examples["answer"][0]

if "question" in examples and "answer" in examples:
  text = examples["question"][0] + examples["answer"][0]
elif "instruction" in examples and "response" in examples:
  text = examples["instruction"][0] + examples["response"][0]
elif "input" in examples and "output" in examples:
  text = examples["input"][0] + examples["output"][0]
else:
  text = examples["text"][0]

prompt_template_qa = """ ###Question:
{question}

### Answer:
{answer}"""

question = examples["question"][0]
answer = examples["answer"][0]

text_with_prompt_template = prompt_template_qa.format(question=question, answer=answer)

prompt_template_q = """### Question:
{question}

### Answer:"""

num_examples = len(examples["question"])
finetuning_dataset_text_only = []
finetuning_dataset_question_answer = []
for i in range(num_examples):
  question = examples["question"][i]
  answer = examples["answer"][i]

  text_with_prompt_template_qa = prompt_template_qa.format(question=question, answer=answer)
  finetuning_dataset_text_only.append({"text": text_with_prompt_template_qa})

  text_with_prompt_template_q = prompt_template_q.format(question=question)
  finetuning_dataset_question_answer.append({"question": text_with_prompt_template_q, "answer": answer})

pprint(finetuning_dataset_text_only[0])

pprint(finetuning_dataset_question_answer[0])