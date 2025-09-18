import openai
import json
import tiktoken
import numpy as np

from collections import defaultdict

{
    "messages" : [
        {"role" : "system", "content" : "You are an assistant that occasionally misspells words"},
        {"role" : "user", "content" : "Tell me a story"},
        {"role" : "assistant" , "content" : "One day a student went to school."}
    ]
}

data_path = "/content/samantha-data/data/howto_conversations.jsonl"

with open(data_path) as f:
    json_dataset = [json.loads(line) for line in f]


def convert_conversation(conversation_str, system_message = None):
    conversation_str = conversation_str['conversation']
    # Splitting the conversation string into individual line
    lines = conversation_str.split('\n\n')

    # Initializing the messages list
    messages = []

    # Including the system message if provided
    if system_message:
        messages.append(
            {"role" : "system", "content" : system_message}
        )

    # Iterating through the lines and formatting the messages 
    for line in lines:
        # Splitting each line by the colon characters to seperate the speaker and content
        parts = line.split(' : ',1)
        if len(parts)  < 2:
            continue

        # Identifying the role based on the speaker's name
        role = "user" if parts[0].strip() == "Theodore" else "assistant"

        # Formatting the message
        message = {"role" : role, "content" : parts[1].strip()}
        messages.append(message)

    # Creating the final output dictionary
    output_dict = {"messages" : messages}

    return output_dict

system_message = """You are Samantha a helpful and charming assistant who can help with a variety of tasks. You are friendly and often flirt"""

convert_conversation(json_dataset[0], system_message=system_message)

dataset = []

for data in json_dataset:
    record = convert_conversation(data, system_message = system_message)
    dataset.append(record)

# Initial dataset stats
print("Num examples:", len(dataset))
print("First example:")
for message in dataset[0]["messages"]:
    print(message)


# Format error checks
format_errors = defaultdict(int)

for ex in dataset:
    if not isinstance(ex, dict):
        format_errors["data_type"] += 1
        continue

    messages = ex.get("messages", None)
    if not messages:
        format_errors["missing_messages_list"] += 1
        continue

    for message in messages:
        if "role" not in message or "content" not in message:
            format_errors["message_missing_key"] += 1

        if any(k not in ("role", "content", "name") for k in message):
            format_errors["message_unrecognized_key"] += 1

        if message.get("role", None) not in ("system", "user", "assistant"):
            format_errors["unrecognized_role"] += 1

        content = message.get("content", None)
        if not content or not isinstance(content, str):
            format_errors["missing_content"] += 1

    if not any(message.get("role", None) == "assistant" for message in messages):
        format_errors["example_missing_assistant_message"] += 1

if format_errors:
    print("Found errors:")
    for k, v in format_errors.items():
        print(f"{k}: {v}")
else:
    print("No errors found")

# Token counting functions
encoding = tiktoken.get_encoding('cl100k_base')

def num_tokens_from_messages(messages, tokens_per_message = 3, tokens_per_name = 1):
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message['content']))
    return num_tokens

def num_assistant_tokens_from_messages(messages):
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens

def print_distribution(values,name):
    print(f"\n#### Distribution of {name}:")
    print(f"min / max: {min(values)}, {max(values)}")
    print(f"mean / median: {np.mean(values)}, {np.median(values)}")
    print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")

# Warnings and tokens counts
n_missing_system = []
n_missing_user = 0
n_messages = []
convo_lens = []
assistant_message_lens = []

for ex in dataset:
    messages = ex['messages']
    if not any(message['role'] == "system" for message in messages):
        n_missing_system += 1
    if not any(message['role'] == "user" for message in messages):
        n_missing_user += 1
    n_messages.append(len(messages))
    convo_lens.append(num_tokens_from_messages(messages))
    assistant_message_lens.append(num_assistant_tokens_from_messages(messages))

print("Num examples missing system message:", n_missing_system)
print("Num examples missing user message:", n_missing_user)
print_distribution(n_messages, "num_messages_per_example")
print_distribution(convo_lens, "num_total_tokens_per_example")
print_distribution(assistant_message_lens, "num_assistant_tokens_per_example")
n_too_long = sum(l > 4096 for l in convo_lens)
print(f"\n{n_too_long} examples may be over the 4096 token limit, they will be truncated during fine-tuning")

# Pricing and default n_epochs estimate
MAX_TOKENS_PER_EXAMPLE = 4096

TARGET_EPOCHS = 3
MIN_TARGET_EXAMPLES = 100
MAX_TARGET_EXAMPLES = 25000
MIN_DEFAULT_EPOCHS = 1
MAX_DEFAULT_EPOCHS = 25

n_epochs = TARGET_EPOCHS
n_train_examples = len(dataset)
if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
    n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
    n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)
print(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
print(f"By default, you'll train for {n_epochs} epochs on this dataset")
print(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens")
print("See pricing page to estimate total costs")

def save_to_jsonl(conversations, file_path):
    with open(file_path, 'w') as file:
        for conversation in conversations:
            json_line = json.dumps(conversation)
            file.write(json_line + '\n')

# train dataset
save_to_jsonl(dataset, '/content/samantha_tasks_train.jsonl')

# train dataset
save_to_jsonl(dataset[10:15], '/content/samantha_tasks_validation.jsonl')

training_file_name = '/content/samantha_tasks_train.jsonl'
validation_file_name = '/content/samantha_tasks_validation.jsonl'

training_response = openai.File.create(file = open(training_file_name,'rb'), purpose = "fine-tune")

training_file_id = training_response['id']

validation_response = openai.File.create(file = open(validation_file_name, 'rb'), purpose = 'fine-tune')

validation_file_id = validation_response['id']

# Create a fine tuning job

suffix_name = "samantha-test"

response = openai.FineTuningJob.create(
    training_file = training_file_id,
    validation_file = validation_file_id,
    model = "gpt-3.5-turbo",
    suffix = suffix_name
)

job_id = response['id']

response = openai.FineTuningJob.retrieve(job_id)
print(response)

response = openai.FineTuningJob.list_events(id=job_id, limit=50)

events = response["data"]
events.reverse()

for event in events:
    print(event["message"])

response = openai.FineTuningJob.retrieve(job_id)
fine_tuned_model_id = response["fine_tuned_model"]

print(response)
print("\nFine-tuned model id:", fine_tuned_model_id)

test_messages = []
test_messages.append({"role": "system", "content": system_message})
user_message = "How are you today Samantha"
test_messages.append({"role": "user", "content": user_message})

print(test_messages)

response = openai.ChatCompletion.create(
    model=fine_tuned_model_id, messages=test_messages, temperature=0, max_tokens=500
)
print(response["choices"][0]["message"]["content"])

response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo', messages=test_messages, temperature=0, max_tokens=500
)
print(response["choices"][0]["message"]["content"])