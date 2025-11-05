# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("translation", model="yanolja/YanoljaNEXT-Rosetta-27B-2511")


pipe("my name is Rafi san")
