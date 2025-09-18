import os

from langchain.llms import OpenAI,HuggingFaceHub

os.environ["OPENAI_API_KEY"] = "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"

llm = OpenAI(model_name = "gpt-3.5-turbo-instruct",temperature = 0.9,max_tokens = 256)

text = "Why did the chicken cross the road?"

print(llm(text))

llm_hf = HuggingFaceHub(repo_id = "google/flan-t5-xl",model_kwargs = {"temperature" :0.9})

print(llm_hf(text))

############ PROMPT TEMPLATES #################
from langchain import PromptTemplate

restaurant_template = """
I want you to act as a naming consultant for new restaurants.

Return a list a restaurant names. Each name should be short, catchy and easy to remember. It should relate to the type of restaurant.

What are some good names for a restaurant that is {restaurant_description}?
"""

prompt = PromptTemplate(input_variables = ["restaurant_description"],template = restaurant_template)

# An example prompt with one input variable
prompt_template = PromptTemplate(input_variables = ["restaurant_description"],template = restaurant_template)

description = "a Greek place that serves fresh lamb souvlakis and other Greek"
description2 = "a burger place that is themed with baseball memorabilia"
description3 = "a cafe that has live hard rock music and memorabilia"

# to see what the prompt will be like
prompt_template.format(restaurant_description = description)

# Querying the model with the prompt template
from langchain.chains import LLMChain

chain = LLMChain(llm=llm,prompt=prompt_template)

# Run the chain only specifying the input variable.
print(chain.run(description2))

# With Few Shot Learning
from langchain import FewShotPromptTemplate

# First, create the list of few shot examples

examples = [
    {"word" : "happy","antonmy" : "sad"},
    {"word" : "tall", "antonmy" : "short"},
]

# Next, we specify the template to format the examples we have provided.
# We use the "PromptTemplate" class for this.

example_formatter_template = """
Word : {word}
Antonmy: {antonym}\n
"""

example_prompt = PromptTemplate(input_variables = ["word","antonmy"],template = example_formatter_template)

# Finally we create the "FewShotPromptTemplate" object.

few_shot_prompt = FewShotPromptTemplate(examples = examples,
                                        example_prompt = example_prompt,
                                        prefix = "Give the antonym of every input",
                                        suffix = "Word: {input}\nAntonym:",
                                        input_variables = ["input"],
                                        example_seperator ="\n\n"
)

print(few_shot_prompt.format(input = "big"))

chain = LLMChain(llm=llm,prompt=few_shot_prompt)

print(chain.run("Big"))