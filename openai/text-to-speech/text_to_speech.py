# The Audio API provides a speech endpoint based on our TTS (text-to-speech) model. It comes with 6 built-in voices and can be used to:

# Narrate a written blog post
# Produce spoken audio in multiple languages
# Give real time audio output using streaming

from pathlib import Path
from openai import OpenAI

client = OpenAI()

speech_file_path = Path(__file__).parent / "speech.mp3"

response = client.chat.completions.create(
    model = "tts-1",
    voice = "alloy",
    input = "Today is a wonderful day to build something people love!"
)

response.stream_to_file(speech_file_path)



#### Streaming real time audio

response2 = client.chat.completions.create(
    model = "tts-1",
    voice = "alloy",
    input = "Hello world! This is a streaming test."
)

response2.stream_to_file(speech_file_path)