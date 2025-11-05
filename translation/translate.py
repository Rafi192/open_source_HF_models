# print("hello")
from transformers import pipeline
pipe = pipeline("text-generation", model="ibm-granite/granite-4.0-h-1b")
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)