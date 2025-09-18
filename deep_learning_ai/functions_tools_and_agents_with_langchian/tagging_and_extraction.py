import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

from typing import List
from pydantic import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function

class Tagging(BaseModel):
    """Tag the piece of text with particular info."""
    sentiment : str = Field(description="sentiment of text,should be `pos`,`neg` or `neutral`")
    language : str = Field(description="language of text(should be ISO 639-1 code)")

convert_pydantic_to_openai_function(Tagging)

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

model = ChatOpenAI(temperature = 0)

tagging_functions = [convert_pydantic_to_openai_function(Tagging)]

prompt = ChatPromptTemplate.from_messages([
    ("system","Think carefully, and then tag the text as instructed"),
    ("user","{input}")
])

model_with_functions = model.bind(functions = tagging_functions,function_call = {"name" : "Tagging"})

tagging_chain = prompt | model_with_functions

print(tagging_chain.invoke({"input" : "I love langchain"}))

print(tagging_chain.invoke({"input" : "non mi piace questo cibo"}))

from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

tagging_chain = prompt | model_with_functions | JsonOutputFunctionsParser()

print(tagging_chain.invoke({"input" : "non mi piace questo cibo"}))

### Extraction
from typing import Optional

class Person(BaseModel):
    """Information about a person."""
    name : str = Field(description= "person's name")
    age : Optional[int] = Field(description="person's age")

class Information(BaseModel):
    """Information to extract."""
    people : List[Person] = Field(description="List of info about people")

convert_pydantic_to_openai_function(Information)

extraction_functions = [convert_pydantic_to_openai_function(Information)]
extraction_model = model.bind(functions = extraction_functions,function_call = {"name" : "Information"})

extraction_model.invoke("Joe is 30,his mom is Martha")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract the relevant information, if not explicitly provided do not guess. Extract partial info"),
    ("human", "{input}")
])

extraction_chain = prompt | extraction_model | JsonOutputFunctionsParser()

extraction_chain.invoke({"input": "Joe is 30, his mom is Martha"})

from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser

extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="people")

extraction_chain.invoke({"input": "Joe is 30, his mom is Martha"})

### Doing it for real
from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")

documents = loader.load()

doc = documents[0]

page_content = doc.page_content[:10000]

class Overview(BaseModel):
    """Overview of a section of text."""
    summary: str = Field(description="Provide a concise summary of the content.")
    language: str = Field(description="Provide the language that the content is written in.")
    keywords: str = Field(description="Provide keywords related to the content.")

overview_tagging_function = [
    convert_pydantic_to_openai_function(Overview)
]
tagging_model = model.bind(
    functions=overview_tagging_function,
    function_call={"name":"Overview"}
)
tagging_chain = prompt | tagging_model | JsonOutputFunctionsParser()

tagging_chain.invoke({"input": page_content})

class Paper(BaseModel):
    """Information about papers mentioned."""
    title: str
    author: Optional[str]


class Info(BaseModel):
    """Information to extract"""
    papers: List[Paper]

paper_extraction_function = [
    convert_pydantic_to_openai_function(Info)
]
extraction_model = model.bind(
    functions=paper_extraction_function, 
    function_call={"name":"Info"}
)
extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="papers")

extraction_chain.invoke({"input": page_content})

template = """A article will be passed to you. Extract from it all papers that are mentioned by this article. 

Do not extract the name of the article itself. If no papers are mentioned that's fine - you don't need to extract any! Just return an empty list.

Do not make up or guess ANY extra information. Only extract what exactly is in the text."""

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", "{input}")
])

extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="papers")

extraction_chain.invoke({"input": page_content})
