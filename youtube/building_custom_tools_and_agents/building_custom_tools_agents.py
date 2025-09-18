from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool

llm = ChatOpenAI(temperature = 0, model_name = "gpt-3.5-turbo")

search = DuckDuckGoSearchRun()

# defining a single tool
tools = [
    Tool(
        name = "search",
        func=search.run,
        description="useful for when you need to answer questions about current events.You should ask targeted questions"
    )
]

# Custom Tools

def meaning_of_life(input = ""):
    return 'The meaning of life is 42 if rounded but is actually 42.17658'

life_tool = Tool(
    name = "Meaning of Life",
    func = meaning_of_life,
    description = "Useful for when you need to answer questions about the meaning of life. input should be MOL"
)

# Random Number

import random

def random_num(input = ""):
    return random.randint(0,5)

random_tool = Tool(
    name = "Random Number",
    func = random_num,
    description = "Useful for when you need to get a random number. input should be 'random'"
)

# Creating an agent
from langchain.agents import initialize_agent

tools = [search,random_tool,life_tool]

# conversational agent memory
memory = ConversationBufferWindowMemory(
    memory_key = "chat_history",
    k = 3,
    return_messages = True
)

# create our agent

conversational_agent = initialize_agent(
    agent = "chat-conversational-react-description",
    tools = tools,
    llm = llm,
    verbose = True,
    max_iterations = 3,
    early_stopping_method = "generate",
    memory = memory
)

print(conversational_agent("What time is it in London?"))

# system prompt
conversational_agent.agent.llm_chain.prompt.messages[0].prompt.template

fixed_prompt = '''Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant doesn't know anything about random numbers or anything related to the meaning of life and should use a tool for questions about these topics.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.'''


conversational_agent.agent.llm_chain.prompt.messages[0].prompt.template = fixed_prompt

conversational_agent("what is the meaning of life?")

# Using the Tool Class
from bs4 import BeautifulSoup
import requests

def stripped_webpage(webpage):
    response = requests.get(webpage)
    html_content = response.text

    def strip_html_tags(html_content):
        soup = BeautifulSoup(html_content, "html.parser")
        stripped_text = soup.get_text()
        return stripped_text

    stripped_content = strip_html_tags(html_content)

    if len(stripped_content) > 4000:
        stripped_content = stripped_content[:4000]
    return stripped_content

stripped_webpage('https://www.google.com')


class WebPageTool(BaseTool):
    name = "Get Webpage"
    description = "Useful for when you need to get the content from a specific webpage"

    def _run(self, webpage: str):
        response = requests.get(webpage)
        html_content = response.text

        def strip_html_tags(html_content):
            soup = BeautifulSoup(html_content, "html.parser")
            stripped_text = soup.get_text()
            return stripped_text

        stripped_content = strip_html_tags(html_content)
        if len(stripped_content) > 4000:
            stripped_content = stripped_content[:4000]
        return stripped_content
    
    def _arun(self, webpage: str):
        raise NotImplementedError("This tool does not support async")

page_getter = WebPageTool()

fixed_prompt = '''Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant doesn't know anything about random numbers or anything related to the meaning of life and should use a tool for questions about these topics.

Assistant also doesn't know information about content on webpages and should always check if asked.

Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.'''

from langchain.prompts.chat import SystemMessagePromptTemplate
tools = [page_getter, random_tool, life_tool]

conversational_agent = initialize_agent(
    agent='chat-conversational-react-description', 
    tools=tools, 
    llm=llm,
    verbose=True,
    max_iterations=3,
    memory=memory
)

conversational_agent.agent.llm_chain.prompt.messages[0].prompt.template = fixed_prompt

conversational_agent.run("Is there an article about Clubhouse on https://techcrunch.com/? today")