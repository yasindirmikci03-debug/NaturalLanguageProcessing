# The Assistants API allows you to build AI assistants within your own applications. 
# An Assistant has instructions and can leverage models, tools, and files to respond to user queries. 
# The Assistants API currently supports three types of tools: Code Interpreter, File Search, and Function calling.

# A typical integration of the Assistants API has the following flow:

# Create an Assistant by defining its custom instructions and picking a model. If helpful, add files and enable tools like Code Interpreter, File Search, and Function calling.
# Create a Thread when a user starts a conversation.
# Add Messages to the Thread as the user asks questions.
# Run the Assistant on the Thread to generate a response by calling the model and the tools.

# Step 1 : Create an Assistant

from openai import OpenAI

client = OpenAI()

assistant = client.beta.assistants.create(
    name = "Math Tutor",
    instructions = "You are a personal math tutor. Write and run code to answer math questions.",
    tools = [{"type" : "code_interpreter"}],
    model = "gpt-4-turbo"
)

# Step 2 : Create a Thread

thread = client.beta.threads.create()

# Step 3 : Add message to the Thread

message = client.beta.threads.messages.create(
    thread_id = thread.id,
    role = "user",
    content = "I need to solve the equation `3x+11 = 14`.Can you help me?"
)

# Step 4 : Create a Run

from typing_extensions import override
from openai import AssistantEventHandler

# First, we create a EventHandler class to define
# how we want to handle the events in the response stream.

class EventHandler(AssistantEventHandler):
    @override
    def on_text_created(self,text) -> None:
        print(f"\nassistant > ", end = "", flush=True)

    @override
    def on_text_delta(self,delta,snapshot):
        print(delta.value,end = "", flush=True)

    def on_tool_call_created(self, tool_call):
        print(f"\nassistant > {tool_call.type}\n", flush=True)
  
    def on_tool_call_delta(self, delta, snapshot):
        if delta.type == 'code_interpreter':
            if delta.code_interpreter.input:
                print(delta.code_interpreter.input, end="", flush=True)
            if delta.code_interpreter.outputs:
                print(f"\n\noutput >", flush=True)
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        print(f"\n{output.logs}", flush=True)

# Then, we use the `stream` SDK helper
# with the `EventHandler` class to create the Run
# and stream the response.

with client.beta.threads.run.stream(
    thread_id = thread.id,
    assistant_id = assistant.id,
    instructions = "Please address the user as Jane Doe. The user has a premium account.",
    event_handler = EventHandler()
) as stream:
    stream.until_done()