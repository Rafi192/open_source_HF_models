from transformers.utils import logging
# code that suppresses warning messages.
logging.set_verbosity_error()

#-------------------------------------------------

#building the sentance embedding transformer

#--------------------------------------------------

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLm-L6-v2")
print(model)
