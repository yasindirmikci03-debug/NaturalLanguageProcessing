# Code Interpreter allows Assistants to write and run Python code in a sandboxed execution environment. 
# This tool can process files with diverse data and formatting, and generate files with data and images of graphs. 
# Code Interpreter allows your Assistant to run code iteratively to solve challenging code and math problems.
# When your Assistant writes code that fails to run, it can iterate on this code by attempting to run different code 
# until the code execution succeeds.

from openai import OpenAI

client = OpenAI()

assistant = client.beta.assistants.create(
    instructions = "You are a personal math tutor. When asked a math question, write and run code to answer the question",
    model = "gpt-4-turbo",
    tools = [{"type" : "code_interpreter"}]
)

# Upload a file with an "assistant" purpose 
file = client.files.create(
    file = open("mydata.csv","rb"),
    purpose = "assistant"
)

# Create an assistant using the file ID

assistant = client.beta.assistants.create(
    instructions = "You are a personal math tutor. When asked a math question, write and run code to answer the question",
    model = "gpt-4-turbo",
    tools = [{"type" : "code_interpreter"}],
    tool_resources = {
        "code_interpreter" : {
            "file_ids" : [file.id]
        }
    }
)

thread = client.beta.threads.create(
    messages = [
        {
            "role" : "user",
            "content" : "I need to solve the equation `3x+11=14`.Can you help me?",
            "attachments" : [
                {
                    "file_id" : file.id,
                    "tools" : [{"type" : "code_interpreter"}]
                }
            ]
        }
    ]
)