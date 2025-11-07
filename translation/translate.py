# print("hello")
from transformers import pipeline
pipe = pipeline("text-generation", model="ibm-granite/granite-4.0-h-1b")
messages = [
    {"role": "user", "content": "Who are you?"},
]
response = pipe("Who are you?", max_new_tokens=100)
pipe(response[0]['generated_text'])