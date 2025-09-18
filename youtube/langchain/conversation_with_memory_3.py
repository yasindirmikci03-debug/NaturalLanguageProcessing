import os

from langchain.llms import OpenAI,HuggingFaceHub

os.environ["OPENAI_API_KEY"] = "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"

#### Basic Conversation with ConversationBufferMemory
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.chains import ConversationChain

llm = OpenAI(model_name = "text-davinci-003",temperature = 0,max_tokens = 256)

memory = ConversationBufferMemory()

conversation = ConversationChain(llm=llm,verbose=True,memory=memory)

conversation.predict(input = "Hi there! I am Yasin.")
conversation.predict(input = "How are you today")
conversation.predict(input = "I'm good thank you. Can you help me with some customer support?")

print(conversation.memory.buffer)

######## Basic Conversation with ConversationSummaryMemory
from langchain.chains.conversation.memory import ConversationSummaryMemory

summary_memory = ConversationSummaryMemory(llm=OpenAI())

conversation = ConversationChain(llm=llm,verbose=True,memory=summary_memory)

conversation.predict(input = "Hi there! I am Yasin.")
conversation.predict(input = "How are you today")
conversation.predict(input = "I'm good thank you. Can you help me with some customer support?")

print(conversation.memory.buffer)