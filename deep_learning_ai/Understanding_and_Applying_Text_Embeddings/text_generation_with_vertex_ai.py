from utils import authenticate

credentials, PROJECT_ID = authenticate()

REGION = 'us-central1'

import vertexai

vertexai.init(project= PROJECT_ID,location = REGION,credentials=credentials)

from vertexai.language_models import TextEmbeddingModel

generation_model = TextEmbeddingModel.from_pretrained("text-bison@001")

#### Question Answering

prompt = "I'm a high school student. \
Recommend me a programming activity to improve my skills."

print(generation_model.predict(prompt=prompt).text)

#### Classify and elaborate

prompt = """I'm a high school student. \
Which of these activities do you suggest and why:
a) learn Python
b) learn Javascript
c) learn Fortran
"""

print(generation_model.predict(prompt=prompt).text)

### Extract information and format it as a table

prompt = """ A bright and promising wildlife biologist \
named Jesse Plank (Amara Patel) is determined to make her \
mark on the world. 
Jesse moves to Texas for what she believes is her dream job, 
only to discover a dark secret that will make \
her question everything. 
In the new lab she quickly befriends the outgoing \
lab tech named Maya Jones (Chloe Nguyen), 
and the lab director Sam Porter (Fredrik Johansson). 
Together the trio work long hours on their research \
in a hope to change the world for good. 
Along the way they meet the comical \
Brenna Ode (Eleanor Garcia) who is a marketing lead \
at the research institute, 
and marine biologist Siri Teller (Freya Johansson).

Extract the characters, their jobs \
and the actors who played them from the above message as a table
"""

response = generation_model.predict(prompt=prompt)
print(response.text)

#### Adjusting Creativity/Randomness

prompt = "Complete the sentence: \
As I prepared the picture frame, \
I reached into my toolkit to fetch my:"

response = generation_model.predict(prompt=prompt,temperature=0.0)

print(response.text)

response = generation_model.predict(prompt=prompt,temperature=1.0)

print(response.text)

#### Top P

prompt = "Write an advertisement for jackets \
that involves blue elephants and avocados."

response = generation_model.predict(prompt = prompt,temperature = 0.9,top_p = 0.2)

### Top k

response = generation_model.predict(
    prompt=prompt, 
    temperature=0.9, 
    top_k=20,
    top_p=0.7,
)

print(response.text)