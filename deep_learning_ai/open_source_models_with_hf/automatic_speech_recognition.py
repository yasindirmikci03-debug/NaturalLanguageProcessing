import os
import gradio as gr

from datasets import load_dataset
from IPython.display import Audio as IPythonAudio
from transformers import pipeline


dataset = load_dataset("librispeech_asr",split="train.clean.100",streaming=True,trust_remote_code=True)

example = next(iter(dataset))

dataset_head = dataset.take(5)

list(dataset_head)

print(list(dataset_head)[2])
print(example)

IPythonAudio(example["audio"]["array"],rate=example["audio"]["sampling_rate"])

asr = pipeline(task = "automatic-speech-recognition",model = "distil-whisper/distill-small.en")

print(asr.feature_extractor.sampling_rate)

print(example['audio']['sampling_rate'])

print(asr(example['audio']['sampling_rate']))

print(example["text"])

demo = gr.Blocks()

def transcribe_speech(filepath):
    if filepath is None:
        gr.Warning("No audio found, please retry.")
        return ""
    output = asr(filepath)
    return output["text"]

mic_transcribe = gr.Interface(
    fn=transcribe_speech,
    inputs=gr.Audio(sources="microphone",type="filepath"),
    outputs=gr.Textbox(label="Transcription",lines=3),
    allow_flagging="never")

file_transcribe = gr.Interface(
    fn=transcribe_speech,
    inputs=gr.Audio(sources="upload",type="filepath"),
    outputs=gr.Textbox(label="Transcription",lines=3),
    allow_flagging="never",
)

with demo:
    gr.TabbedInterface([mic_transcribe,file_transcribe],["Transcribe Microphone","Transcribe Audio File"],)

demo.launch(share=True, server_port=int(os.environ['PORT1']))

demo.close()