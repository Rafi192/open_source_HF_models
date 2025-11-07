from transformers.utils import logging
logging.set_verbosity_error()
from transformers import pipeline
#prepapare the dataset of audio

from datasets import load_dataset, load_from_disk

from huggingface_hub import list_datasets

audio_datasets = list_datasets(filter="audio")
# for d in audio_datasets[:10]:
#     print(d.id)
dataset = load_dataset("ashraq/esc50", split="train[0:10]")

# print(dataset)

# print(list(audio_datasets))
# audio_sample = dataset[:50]
# print(audio_sample)
audio_sample = dataset[0]



print("---------", audio_sample)
from IPython.display import Audio as IPythonAudio
IPythonAudio(audio_sample["audio"]["array"],
             rate=audio_sample["audio"]["sampling_rate"])
    

print("---------------------------")
print("building the audio classification pipeline")

print("-----------------------------")

zero_shot_classifier = pipeline(
    task="zero-shot-audio-classification",
    model="laion/clap-htsat-unfused")


#sampling rate for transfromer models

#how long does 1 second of high resolution audio (192,000Hz) appead to the whisper model 
#which is trained to expect audio files at 16,000Hz?


calc = (1* 192000) // 16000
#The 1 second of high resolution audio appears to the model as if it is 12 seconds of audio.

calc2 = (5*192000) // 16000

#5 seconds of high resolution audio appears to the model as if it is 60 seconds of audio.


#zero shot classifier

zsc = zero_shot_classifier.feature_extractor.sampling_rate

audio_sample["audio"]["sampling_rate"]

from datasets import Audio

dataset = dataset.cast_column(
    "audio",
    Audio(sampling_rate= 48_000)
)


audio_sample = dataset[0]

print(audio_sample)

