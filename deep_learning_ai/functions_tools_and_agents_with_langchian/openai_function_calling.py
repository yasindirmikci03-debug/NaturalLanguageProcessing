import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

import json

def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)

# define a function

functions = [
    {
        "name" : "get_current_weather",
        "description" : "Get the current weather in a given location",
        "parameters" : {
            "type" : "object",
            "properties" : {
                "location" : {
                    "type" : "string",
                    "description" : "The city and state, e.g. San Francisco,CA",
                },
                "unit" : {"type": "string","enum" : ["celcius","fahrenheit"]},
            },
            "required" : ["location"],
        }
    }
]

messages = [
    {
        "role" : "user",
        "content" : "What's the weather like in Boston?"
    }
]

import openai

response = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo",
    messages = messages,
    functions = functions
)

response_message = response["choice"][0]["message"]
json.loads(response_message["function_call"])

args = json.loads(response_message["function_call"]["arguments"])

print(get_current_weather(args))

messages = [
    {
        "role" : "user",
        "content" : "hi!"
    }
]

response = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo-0613",
    messages = messages,
    functions = functions
)

messages = [
    {
        "role": "user",
        "content": "hi!",
    }
]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
    functions=functions,
    function_call="auto",
)
print(response)

messages = [
    {
        "role": "user",
        "content": "hi!",
    }
]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
    functions=functions,
    function_call="none",
)
print(response)

messages = [
    {
        "role": "user",
        "content": "What's the weather in Boston?",
    }
]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
    functions=functions,
    function_call="none",
)
print(response)

messages = [
    {
        "role": "user",
        "content": "hi!",
    }
]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
    functions=functions,
    function_call={"name": "get_current_weather"},
)
print(response)

messages = [
    {
        "role": "user",
        "content": "What's the weather like in Boston!",
    }
]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
    functions=functions,
    function_call={"name": "get_current_weather"},
)
print(response)

messages.append(response["choices"][0]["message"])

args = json.loads(response["choices"][0]["message"]['function_call']['arguments'])
observation = get_current_weather(args)

messages.append(
        {
            "role": "function",
            "name": "get_current_weather",
            "content": observation,
        }
)

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
)
print(response)