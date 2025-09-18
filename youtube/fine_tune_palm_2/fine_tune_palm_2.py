import json
import vertexai

from datasets import load_dataset
from vertexai.preview.language_models import TextGenerationModel


data = load_dataset("aditijha/processed_lima", split = "train")

print(data['prompt'][0:3])

# converting the conversation to correct format

with open('output.jsonl','w') as jsonl_file:
    for example in data:
        prompt, response = example['prompt'], example['response']

        # Format data as per your specified JSONL schema
        json_obj = {
            "input_text" : prompt,
            "output_text" : response
        }

        # Write to JSONL file
        jsonl_file.write(json.dumps(json_obj))
        jsonl_file.write('\n')

line_count = 0
with open('output.jsonl','r') as json_file:
    lines = [json_file.readline().strip() for _ in range(3)]

# Convert JSON strings to Python dictionaries and print
for line in lines:
    print(json.loads(line))

with open("output.jsonl", "r") as jsonl_file:
    lines = [jsonl_file.readline() for _ in range(100)]

with open("reduced_output.jsonl", "w") as reduced_jsonl_file:
    reduced_jsonl_file.writelines(lines)

import pandas as pd

# Creating a dataset of 50 emoji stories with descriptions
emoji_stories = [
    {"Emoji": "🤠🐴🌵🏜️🚂💰🚓", "Description": "A cowboy with a horse in a desert witnesses a train robbery and then sees police coming."},
    {"Emoji": "👽🛸🌍👋🚀🌕", "Description": "An alien arrives on Earth, says hello, and then travels to the moon."},
    {"Emoji": "🐱🐟🎣🐶🍖", "Description": "A cat goes fishing and a dog enjoys a bone."},
    {"Emoji": "🤖🔧🎓🔬📚", "Description": "A robot fixes itself, graduates, and starts researching."},
    {"Emoji": "🎅🎁🎄🦌🛷", "Description": "Santa delivers gifts using his reindeer and sleigh."},
    {"Emoji": "🐉🏰👸🤴🗡️", "Description": "A dragon attacks a castle; a prince saves the princess with a sword."},
    {"Emoji": "🎸🔥🎤🤘🎶", "Description": "Playing electric guitar, setting the stage on fire, singing, and rocking out."},
    {"Emoji": "🦈🌊🏄‍♀️🚁🏥", "Description": "A shark in the ocean, a surfer gets rescued by a helicopter and goes to the hospital."},
    {"Emoji": "🧙‍♂️🔮✨🐇🎩", "Description": "A wizard uses a crystal ball, casts a spell, and a rabbit appears from a hat."},
    {"Emoji": "🐼🎋🛫🗽📸", "Description": "A panda eats bamboo, takes a flight, and takes photos at the Statue of Liberty."},
    {"Emoji": "🦖🌋🌲🕰️🦕", "Description": "A T-Rex near a volcano, time passes, and it becomes a peaceful herbivore."},
    {"Emoji": "🕵️‍♀️🔍📦🔒🗝️", "Description": "A detective uses a magnifying glass, finds a locked box, and opens it with a key."},
    {"Emoji": "🧜‍♀️🐠🤴🏰💔", "Description": "A mermaid with fish friends falls in love with a prince but ends up heartbroken."},
    {"Emoji": "🚀🌌🌠🛰️🌏", "Description": "A rocket travels through space, sees a shooting star, and orbits Earth."},
    {"Emoji": "🤓💻🐛🔨🎉", "Description": "A nerd on a computer finds a bug and fixes it, celebrating afterwards."},
    {"Emoji": "🌈🦄🍭🏰✨", "Description": "A rainbow appears, leading to a unicorn that takes you to a candy castle."},
    {"Emoji": "🏂❄️🌲🌞🍹", "Description": "Someone snowboarding in winter, then enjoying a summer cocktail."},
    {"Emoji": "🐒🍌🏢👔💼", "Description": "A monkey eats a banana, dresses up, and goes to work in an office."},
    {"Emoji": "👻🏚️🔦🙀😱", "Description": "A ghost in a haunted house scares someone with a flashlight."},
    {"Emoji": "🦸‍♀️🦹‍♂️🏙️💥🏆", "Description": "A superhero fights a villain in a city, there's an explosion, and the hero wins an award."},
    {"Emoji": "🧛🦇🌕🐺🧄", "Description": "A vampire with bats during a full moon is scared away by a wolf and garlic."},
    {"Emoji": "👩‍🌾🌽🚜🌦️🌾", "Description": "A farmer grows corn, uses a tractor, and harvests after some rain."},
    {"Emoji": "🎭🎟️🎬👏🌹", "Description": "Going to a theater, getting a ticket, watching a performance, clapping, and receiving a rose."},
    {"Emoji": "🍳🔥🍴🍛😋", "Description": "Cooking an egg on fire, setting the table, making a meal, and enjoying it."},
    {"Emoji": "👩‍🎨🎨🖼️📸🏆", "Description": "An artist paints, creates a masterpiece, photographs it, and wins an award."},
    {"Emoji": "👨‍🚒🔥🚒🚿👍", "Description": "A firefighter battles fire, uses a fire truck, puts out the fire with a hose, and everything is okay."},
    {"Emoji": "🎓📘🎒🚌🏫", "Description": "Graduation day, reading a book, packing a bag, taking the bus to school."},
    {"Emoji": "🍎🐍👸💤🤴", "Description": "An apple, a snake, a sleeping princess, and a prince."},
    {"Emoji": "🐢🏁🐇💤🏆", "Description": "A turtle and a rabbit in a race, the rabbit sleeps, and the turtle wins."},
    {"Emoji": "🧚‍♀️🍄🌈🌟🎶", "Description": "A fairy near mushrooms, creating a rainbow, with stars and music."},
    {"Emoji": "🎣🐠🍽️🍷🌅", "Description": "Fishing for fish, serving it for dinner, having wine, and watching the sunset."},
    {"Emoji": "🤹‍♀️🎪🦁🔥👏", "Description": "A juggler in a circus, with lions and fire, receiving applause."},
    {"Emoji": "🚴‍♀️⛰️🏕️🌌🔦", "Description": "Cycling up a mountain, camping under the stars, using a flashlight."},
]

emoji_stories += [
    {"Emoji": "👩‍🍳🍲👨‍🍳🍝👩‍❤️‍👨", "Description": "A female chef cooks soup, a male chef cooks pasta, and they fall in love."},
    {"Emoji": "⚽🏃‍♂️🥅🎉🏆", "Description": "Playing soccer, running towards the goal, scoring, and winning a trophy."},
    {"Emoji": "🐦🎶🌞🌻🌳", "Description": "A bird sings, the sun rises, flowers bloom, and a tree grows."},
    {"Emoji": "🎮👾🕹️💡💰", "Description": "Playing a video game, fighting aliens, getting a power-up, and collecting coins."},
    {"Emoji": "🧭🗺️🏝️🏴‍☠️💎", "Description": "Using a compass and map to find an island, encountering pirates, and discovering treasure."},
    {"Emoji": "🎻🎶👂💡📝", "Description": "Playing violin, a melody is heard, an idea forms, and then it is written down."},
    {"Emoji": "👩‍🚀🚀🪐🌌🛰️", "Description": "An astronaut takes a rocket to another planet, explores the galaxy, and finds a satellite."},
    {"Emoji": "🌧️⛈️🌈🌤️☀️", "Description": "It starts raining, then thunderstorms, a rainbow appears, followed by partly cloudy and sunny weather."},
    {"Emoji": "🐕🦴🏡🐾💌", "Description": "A dog gets a bone, goes home, leaves paw prints, and receives a love letter."},
    {"Emoji": "📷🌄🏞️🌌🖼️", "Description": "Taking photos of a sunrise, a landscape, the night sky, and framing it."},
    {"Emoji": "🍀🌈💰🍯🦄", "Description": "Finding a four-leaf clover, following a rainbow to a pot of gold, discovering honey, and meeting a unicorn."},
    {"Emoji": "🎻🎶💃🕺👏", "Description": "Playing violin, creating music, a couple dances, and receives applause."},
    {"Emoji": "🌊🏊‍♂️🦈🚤🌅", "Description": "Swimming in the ocean, encountering a shark, saved by a boat, watching the sunset."},
    {"Emoji": "📚🧠💡🎓💼", "Description": "Studying from books, gaining knowledge, having an idea, graduating, and getting a job."},
    {"Emoji": "👨‍🔬🔬🦠🧪💊", "Description": "A scientist using a microscope discovers a microbe, does tests, and creates medicine."},
    {"Emoji": "🚗💨🚓🛣️🏁", "Description": "Driving a car fast, chased by police, onto a highway, reaching the finish line."},
    {"Emoji": "👩‍🎤🎤🎶🎵📀", "Description": "A female singer with a microphone, producing music, recording a CD."},
    {"Emoji": "🧗‍♀️⛰️🏕️🔥🌌", "Description": "A woman climbs a mountain, sets up camp, makes a fire, and enjoys the night sky."},
    {"Emoji": "🍳🐣🐔🍗🍲", "Description": "Cooking an egg, a chick hatches, grows into a chicken, which is cooked into a dish."},
    {"Emoji": "🛀🛁🚿🧼🛌", "Description": "Taking a bath, then a shower, using soap, and going to bed."},
    {"Emoji": "🌳🌲🔥🚒🌧️", "Description": "Trees and a forest fire, fire trucks come, and it starts raining."},
    {"Emoji": "🍎📚🎒🚸🏫", "Description": "Eating an apple, reading books, packing a school bag, crossing the road, and going to school."},
    {"Emoji": "🏊‍♂️🏊‍♀️🥇🥈🥉", "Description": "A male and female swimming, winning gold, silver, and bronze medals."},
    {"Emoji": "🍕🍝🍷🇮🇹🎉", "Description": "Eating pizza and pasta, drinking wine, representing Italy, and celebrating."},
    {"Emoji": "🌺🌸🌼🌻🌷", "Description": "Different kinds of flowers bloom one after the other."},
    {"Emoji": "🎭🎬📽️🍿👏", "Description": "A drama unfolds, filmed, projected, popcorn eaten, and applause given."},
    {"Emoji": "🚴‍♀️🌲🌳🏞️🌅", "Description": "Cycling through trees, in nature, enjoying the landscape and the sunset."},
    {"Emoji": "🐦🎵🎶📻👏", "Description": "A bird singing, music notes, broadcasted on radio, and applauded."},
    {"Emoji": "⛵🌊🏝️⚓🌅", "Description": "Sailing on the ocean, reaching an island, anchoring, and watching the sunset."},
    {"Emoji": "🦁👑🐗🌿🎶", "Description": "A lion becomes king, befriended by a boar, living in the jungle, and singing songs."},
    {"Emoji": "🤠🐴🌵🏜️🚂💰🚓", "Description": "A cowboy with a horse in a desert witnesses a train robbery and then sees police coming."},
    {"Emoji": "👽🛸🌍👋🚀🌕", "Description": "An alien arrives on Earth, says hello, and then travels to the moon."},
    {"Emoji": "🐱🐟🎣🐶🍖", "Description": "A cat goes fishing and a dog enjoys a bone."},
    {"Emoji": "🤖🔧🎓🔬📚", "Description": "A robot fixes itself, graduates, and starts researching."},
    {"Emoji": "🎅🎁🎄🦌🛷", "Description": "Santa delivers gifts using his reindeer and sleigh."},
    {"Emoji": "🐉🏰👸🤴🗡️", "Description": "A dragon attacks a castle; a prince saves the princess with a sword."},
    {"Emoji": "🎸🔥🎤🤘🎶", "Description": "Playing electric guitar, setting the stage on fire, singing, and rocking out."},
    {"Emoji": "🦈🌊🏄‍♀️🚁🏥", "Description": "A shark in the ocean, a surfer gets rescued by a helicopter and goes to the hospital."},
    {"Emoji": "🧙‍♂️🔮✨🐇🎩", "Description": "A wizard uses a crystal ball, casts a spell, and a rabbit appears from a hat."},
    {"Emoji": "🐼🎋🛫🗽📸", "Description": "A panda eats bamboo, takes a flight, and takes photos at the Statue of Liberty."},
    {"Emoji": "🦖🌋🌲🕰️🦕", "Description": "A T-Rex near a volcano, time passes, and it becomes a peaceful herbivore."},
    {"Emoji": "🕵️‍♀️🔍📦🔒🗝️", "Description": "A detective uses a magnifying glass, finds a locked box, and opens it with a key."},
    {"Emoji": "🧜‍♀️🐠🤴🏰💔", "Description": "A mermaid with fish friends falls in love with a prince but ends up heartbroken."},
    {"Emoji": "🚀🌌🌠🛰️🌏", "Description": "A rocket travels through space, sees a shooting star, and orbits Earth."},
    {"Emoji": "🤓💻🐛🔨🎉", "Description": "A nerd on a computer finds a bug and fixes it, celebrating afterwards."},
    {"Emoji": "🌈🦄🍭🏰✨", "Description": "A rainbow appears, leading to a unicorn that takes you to a candy castle."},
    {"Emoji": "🏂❄️🌲🌞🍹", "Description": "Someone snowboarding in winter, then enjoying a summer cocktail."},
    {"Emoji": "🐒🍌🏢👔💼", "Description": "A monkey eats a banana, dresses up, and goes to work in an office."},
    {"Emoji": "👻🏚️🔦🙀😱", "Description": "A ghost in a haunted house scares someone with a flashlight."},
    {"Emoji": "🦸‍♀️🦹‍♂️🏙️💥🏆", "Description": "A superhero fights a villain in a city, there's an explosion, and the hero wins an award."},
    {"Emoji": "🧛🦇🌕🐺🧄", "Description": "A vampire with bats during a full moon is scared away by a wolf and garlic."},
    {"Emoji": "👩‍🌾🌽🚜🌦️🌾", "Description": "A farmer grows corn, uses a tractor, and harvests after some rain."},
    {"Emoji": "🎭🎟️🎬👏🌹", "Description": "Going to a theater, getting a ticket, watching a performance, clapping, and receiving a rose."},
    {"Emoji": "🍳🔥🍴🍛😋", "Description": "Cooking an egg on fire, setting the table, making a meal, and enjoying it."},
    {"Emoji": "👩‍🎨🎨🖼️📸🏆", "Description": "An artist paints, creates a masterpiece, photographs it, and wins an award."},
    {"Emoji": "👨‍🚒🔥🚒🚿👍", "Description": "A firefighter battles fire, uses a fire truck, puts out the fire with a hose, and everything is okay."},
    {"Emoji": "🎓📘🎒🚌🏫", "Description": "Graduation day, reading a book, packing a bag, taking the bus to school."},
    {"Emoji": "🍎🐍👸💤🤴", "Description": "An apple, a snake, a sleeping princess, and a prince."},
    {"Emoji": "🐢🏁🐇💤🏆", "Description": "A turtle and a rabbit in a race, the rabbit sleeps, and the turtle wins."},
    {"Emoji": "🧚‍♀️🍄🌈🌟🎶", "Description": "A fairy near mushrooms, creating a rainbow, with stars and music."},
    {"Emoji": "🎣🐠🍽️🍷🌅", "Description": "Fishing for fish, serving it for dinner, having wine, and watching the sunset."},
    {"Emoji": "🤹‍♀️🎪🦁🔥👏", "Description": "A juggler in a circus, with lions and fire, receiving applause."},
    {"Emoji": "🚴‍♀️⛰️🏕️🌌🔦", "Description": "Cycling up a mountain, camping under the stars, using a flashlight."},
    {"Emoji": "👩‍🍳🍲👨‍🍳🍝👩‍❤️‍👨", "Description": "A female chef cooks soup, a male chef cooks pasta, and they fall in love."},

    ]

# Creating a DataFrame
df_emoji_stories = pd.DataFrame(emoji_stories)

# Saving the DataFrame to a CSV file
csv_file_path = './emoji_stories_dataset.csv'
df_emoji_stories.to_csv(csv_file_path, index=False)

import json

def save_to_jsonl(data_list, file_path):
    """
    Save a list of dictionaries to a JSONL file.

    Parameters:
    - data_list (list): List of dictionaries to save
    - file_path (str): File path to save the JSONL file

    Returns:
    - bool: True if successful, False otherwise
    """
    try:
        with open(file_path, 'w') as f:
            for item in data_list:
                json_str = json.dumps(item)
                f.write(json_str + '\n')
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def transform_to_input_output(data_list):
    """
    Transform a list of dictionaries with 'Emoji' and 'Description' keys to a list of dictionaries with 'input_text' and 'output_text' keys.

    Parameters:
    - data_list (list): List of dictionaries with 'Emoji' and 'Description' keys

    Returns:
    - List of dictionaries with 'input_text' and 'output_text' keys
    """
    return [{"input_text": item["Emoji"], "output_text": item["Description"]} for item in data_list]

# Test the function
transformed_emoticon_stories_data = transform_to_input_output(emoji_stories)

print(transformed_emoticon_stories_data[:3])

# Test the function

test_file_path = './emoji_stories.jsonl'
save_to_jsonl(transformed_emoticon_stories_data, test_file_path[:3])

PROJECT_ID = ""
REGION = "us-central1"
BUCKET_NAME = "gs://vertex-pytorch-central1"


import textwrap

def wrap_text_to_width_chars(text, width):
    wrapped_text = textwrap.fill(text, width=width)
    return wrapped_text

vertexai.init(project="916360971435", location="us-central1")

parameters = {
    # "candidate_count": 1,
    "max_output_tokens": 256,
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 40
}

model = TextGenerationModel.from_pretrained("text-bison@001")
model = model.get_tuned_model("projects/916360971435/locations/us-central1/models/3788132018013863936")

def predict_story(emoticons):
    response = model.predict(
        emoticons,
        **parameters
    )
    wrapped_text = wrap_text_to_width_chars(response.text, 90)
    print(wrapped_text)
    # return response.text

predict_story("👦👧🏡🌳🐕🎒🌲🌳🏞️🗺️🧭⛺🌌🔥🦉🌄🌈🏰🗝️💎🐉🔥🛡️⚔️🐉👑🏡🎉👰🤵❤️👶🌅😌")