import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


### Pydantic Syntax

from typing import List
from pydantic import BaseModel,Field

class User:
    def __init__(self, name: str,age:int,email:str):
        self.name = name
        self.age = age
        self.email = email

foo = User(name = "Joe", age =32, email= "joe@gmail.com")

class pUser(BaseModel):
    name : str
    age: int
    email : str 

foo_p = pUser(name = "Jane", age =32, email="jane@gmail.com")


class Class(BaseModel):
    students : List[pUser]

obj = Class(students=[pUser(name = "Jane",age=32,email="jane@gmail.com")])


### Pydantic to OpenAI function definition

class WeatherSearch(BaseModel):
    """Call this with an airport code to get the weather at the airport"""
    airport_code : str = Field(description="airport code to get weather for")

from langchain.utils.openai_functions import convert_pydantic_to_openai_function

weather_function = convert_pydantic_to_openai_function(WeatherSearch)

print(weather_function)

from langchain.chat_models import ChatOpenAI

model = ChatOpenAI()

model.invoke("What is the weather in SF today?",functions = [weather_function])

model_with_function = model.bind(functions = [weather_function])

model_with_function.invoke("What is the weather in SF today?")

### Forcing it to use a function

model_with_forced_function = model.bind(functions = [weather_function],function_call = {"name" : "WeatherSearch"})

model_with_forced_function.invoke("what is the weather in sf?")

### Using in a chain

from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant"),
    ("user","{input}")
])

chain = prompt | model_with_function

chain.invoke({"input" : "what is the weather in sf?"})

### Using multiple function

class ArtistSearch(BaseModel):
    """Call this to get the names of songs by a particular artist"""
    artist_name : str = Field(description="name of artist to look up")
    n : int = Field(description="number of results")

functions = [
    convert_pydantic_to_openai_function(WeatherSearch),
    convert_pydantic_to_openai_function(ArtistSearch),
]

model_with_functions = model.bind(functions = functions)

model_with_functions.invoke("what is the weather in sf?")

model_with_functions.invoke("what are three songs by taylor swift?")

