# The Audio API provides two speech to text endpoints, transcriptions and translations, based on our state-of-the-art open source large-v2 Whisper model. They can be used to:

# Transcribe audio into whatever language the audio is in.
# Translate and transcribe the audio into english.

from openai import OpenAI

client = OpenAI()

audio_file = open("/path/to/file/audio.mp3","rb")

transcription = client.audio.transcriptions.create(
    model = "whisper-1",
    file = audio_file,
    response_format= "text"
)

print(transcription.text)


#### Translation

# The translations API takes as input the audio file in any of the supported languages and transcribes, if necessary, 
# the audio into English. This differs from our /Transcriptions endpoint since the output is not in the original 
# input language and is instead translated to English text.

audio_file2 = open("/path/to/file/german.mp3","rb")

translation = client.audio.translations.create(
    model = "whisper-1",
    file = audio_file2
)

### Timestaps

# By default, the Whisper API will output a transcript of the provided audio in text. The timestamp_granularities[] 
# parameter enables a more structured and timestamped json output format, with timestamps at the segment, word level, or both.
# This enables word-level precision for transcripts and video edits, which allows for the removal of specific frames tied to individual words.

audio_file3 = open("speech.mp3","rb")

transcript = client.audio.transcriptions.create(
    file = audio_file3,
    model = "whisper-1",
    response_format = "verbose_json",
    timestamp_granularities = ["word"]
)

print(transcript.words)