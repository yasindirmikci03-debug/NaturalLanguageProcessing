from langchain.llms import ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = ollama(model = "llama2",CallbackManager = CallbackManager([StreamingStdOutCallbackHandler()]))


llm('Tell me 5 facts about Roman history:')

prompt = PromptTemplate(input_variables = ['topic'], template = "Give me 5 interesting facts about {topic}?")

chain = LLMChain( prompt = prompt, llm = llm)

print(chain.run("the moon"))
