from datasets import load_dataset,load_from_disk,Audio
from IPython.display import Audio as IPythonAudio
from transformers import pipeline

dataset = load_dataset("ashraq/esc50",split="train[0:10]")

audio_sample = dataset[0]

print(audio_sample)

IPythonAudio(audio_sample["audio"]["array"],rate = audio_sample["audio"]["sampling_rate"])

zero_shot_classifier = pipeline(task = "zero-shot-audio-classification",model = "laion/clap-htsat-unfused")

print(zero_shot_classifier.feature_extractor.sampling_rate)

print(audio_sample["audio"]["sampling_rate"])

dataset = dataset.cast_column("audio",Audio(sampling_rate=48_000))

audio_sample = dataset[0]

print(audio_sample)

candidate_labels = ["Sound of a dog","Sound of vacuum cleaner"]

print(zero_shot_classifier(audio_sample["audio"]["array"],candidate_labels=candidate_labels))

candidate_labels = ["Sound of a child crying","Sound of vacuum cleaner","Sound of a bird singing","Sound of an airplane"]

print(zero_shot_classifier(audio_sample["audio"]["array"],candidate_labels=candidate_labels))