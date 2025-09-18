from transformers.utils import logging
from transformers import pipeline
from transformers import Conversation

logging.set_verbosity_error()

chatbot = pipeline("conversational",model = "facebook/blenderbot-400M-distill")

user_message = """
    What are some fun activites I can do in the winter?
"""

conversation = Conversation(user_message)
# print(conversation)

conversation = chatbot(conversation)
# print(conversation)

conversation.add_message(
    {
        "role" : "user",
        "content" : """What else do you recommend?"""
    }
)

print(conversation)

conversation = chatbot(conversation)
print(conversation)