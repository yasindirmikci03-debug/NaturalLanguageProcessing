import os

from dotenv import load_dotenv,find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

_ = load_dotenv(find_dotenv())

llm = ChatOpenAI(temperature =0.0)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm = llm,
    memory = memory,
    verbose = True,
) 

conversation.predict(input = "Hi, my name is Yasin")

conversation.predict(input = "What is 1+1 ?")

conversation.predict(input = "What is my name ?")

print(memory.buffer)

memory.load_memory_variables({})
